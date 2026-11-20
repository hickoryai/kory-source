"""One-time import of explicit, filtered session-history bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from mira.registry import decision_llm
from mira.registry.access import (
    AccessContext,
    AccessDenied,
    require_import,
    session_subject,
)
from mira.registry.attribution import Attribution, tool_attribution
from mira.registry.decision_passages import Passage, nominate
from mira.registry.imports import (
    IMPORT_ACTOR_SUB,
    ImportRuns,
    materialize,
    record_cursor,
)
from mira.registry.model import RegistryError
from mira.registry.provenance import ProvenanceStore, QuoteMismatch, SourceRevision, evidence_id_for

IMPORTER = "session_history"
VERSION = "1"
SYSTEMS = ("kory_code_session", "agent_memory", "offbox_session")
EVENT_TYPES = ("user", "text")
FILE_DIGEST = re.compile(r"[0-9a-f]{64}")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=True) + "\n").encode()


def _object(data: bytes) -> dict:
    obj = json.loads(data)
    if not isinstance(obj, dict):
        raise ValueError("expected an object")
    return obj


def _label(value: Any) -> str:
    return value if isinstance(value, str) and value.strip() else ""


def _timestamp(value: str) -> float:
    if value:
        try:
            instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if instant.tzinfo is not None:
                return instant.timestamp()
        except ValueError:
            pass
    return 0.0


def _events(data: bytes) -> tuple[list[dict], int]:
    events = []
    excluded = 0
    for line in data.splitlines():
        try:
            event = _object(line)
        except ValueError:
            excluded += 1
            continue
        kind, content = event.get("type"), event.get("content")
        if kind not in EVENT_TYPES or not isinstance(content, str) or "\x00" in content:
            excluded += 1
            continue
        try:
            content.encode()
        except UnicodeEncodeError:
            excluded += 1
            continue
        events.append(
            {
                "type": kind,
                "content": content,
                "ts": _label(event.get("ts")),
                "origin": _label(event.get("origin")),
            }
        )
    return events, excluded


def _sidecar(path: Path) -> dict[str, str]:
    sidecar = path.with_suffix(".meta.json")
    if sidecar.is_symlink() or not sidecar.is_file():
        return {}
    try:
        meta = _object(sidecar.read_bytes())
    except ValueError:
        return {}
    links = meta.get("links")
    ticket = links.get("ticket") if isinstance(links, dict) else None
    fields = {
        "session": _label(meta.get("id")) or _label(meta.get("sessionId")),
        "model": _label(meta.get("model")),
        "branch": _label(meta.get("branch")),
        "ticket": _label(ticket.get("id") if isinstance(ticket, dict) else ticket),
    }
    return {key: value for key, value in fields.items() if value}


def export_bundle(
    bundle: Path,
    *,
    user: str,
    sources: list[tuple[str, Path]],
    unavailable: list[str],
) -> dict:
    """Copy only selected file types from explicit roots; never copy native state."""
    user = session_subject(user)
    if not user:
        raise RegistryError("a valid session subject is required")
    bundle.mkdir(mode=0o700)
    files = []
    for system, root in sources:
        if system not in SYSTEMS or root.is_symlink() or not root.is_dir():
            raise RegistryError("each source must be a supported type and an explicit directory")
        if bundle.resolve().is_relative_to(root.resolve()):
            raise RegistryError("the bundle must be outside the source directories")
        # Memory roots may have nested projects. Transcript roots are flat.
        root_id = f"{root.name}-{digest(str(root.resolve()).encode())[:16]}"
        paths = root.rglob("*.md") if system == "agent_memory" else root.glob("*.jsonl")
        for path in sorted(paths):
            if (
                not path.is_file()
                or path.is_symlink()
                or not path.resolve().is_relative_to(root.resolve())
            ):
                continue
            raw = path.read_bytes()
            excluded = 0
            meta = {}
            if system == "agent_memory":
                if "\x00" in raw.decode("utf-8"):
                    raise RegistryError("memory file contains a NUL character")
                data = raw
            else:
                events, excluded = _events(raw)
                data = b"".join(_json_bytes(event) for event in events)
                meta = _sidecar(path)
            source_id = f"{root_id}/{path.relative_to(root).as_posix()}"
            checksum = digest(data)
            name = f"{checksum}.data"
            (bundle / name).write_bytes(data)
            files.append(
                {
                    "system": system,
                    "source_id": source_id,
                    "sha256": checksum,
                    "source_sha256": digest(raw),
                    "excluded": excluded,
                    "meta": meta,
                }
            )
    manifest = {
        "schema": VERSION,
        "user": user,
        "files": files,
        "unavailable": sorted(set(unavailable)),
    }
    (bundle / "manifest.json").write_bytes(_json_bytes(manifest))
    return {
        "files_read": len(files),
        "excluded_events": sum(entry["excluded"] for entry in files),
        "unavailable": manifest["unavailable"],
    }


@dataclass(frozen=True)
class EventSpan:
    start: int
    end: int
    kind: str
    timestamp: float
    origin: str


class SessionRevision(SourceRevision):
    @property
    def observation_sha256(self) -> str:
        return self.source_version


@dataclass(frozen=True)
class BundleEntry:
    system: str
    source_id: str
    sha256: str
    source_sha256: str
    excluded: int
    meta: dict

    def __post_init__(self) -> None:
        name = PurePosixPath(self.source_id)
        if (
            self.system not in SYSTEMS
            or name.is_absolute()
            or ".." in name.parts
            or len(name.parts) < 2
            or not FILE_DIGEST.fullmatch(self.sha256)
            or not FILE_DIGEST.fullmatch(self.source_sha256)
            or type(self.excluded) is not int
            or self.excluded < 0
            or not isinstance(self.meta, dict)
            or set(self.meta) - {"session", "model", "branch", "ticket"}
            or not all(_label(value) for value in self.meta.values())
        ):
            raise ValueError("invalid file entry")

    @property
    def source_type(self) -> str:
        return "memory_file" if self.system == "agent_memory" else "transcript"

    def evidence_id(self, namespace: str) -> str:
        return evidence_id_for(
            self.system, self.source_type, f"{namespace}/{self.source_id}", self.source_sha256
        )


@dataclass(frozen=True)
class BundleFile:
    revision: SourceRevision
    meta: dict
    spans: list[EventSpan]
    passages: list[Passage]
    excluded: int

    @classmethod
    def load(cls, bundle: Path, entry: BundleEntry, *, namespace: str) -> BundleFile:
        meta = entry.meta
        excluded = entry.excluded
        path = bundle / f"{entry.sha256}.data"
        if path.is_symlink():
            raise ValueError("bundle files must be regular files")
        raw = path.read_bytes()
        if digest(raw) != entry.sha256:
            raise ValueError("file digest mismatch")
        spans = []
        passages = []
        if entry.system == "agent_memory":
            text = raw.decode()
            if "\x00" in text:
                raise ValueError("invalid memory text")
            passages = nominate(text, markdown=True)
        else:
            events, excluded_in_bundle = _events(raw)
            excluded += excluded_in_bundle
            pieces = []
            offset = 0
            for event in events:
                content, kind = event["content"], event["type"]
                timestamp = _timestamp(event["ts"])
                spans.append(
                    EventSpan(offset, offset + len(content), kind, timestamp, event["origin"])
                )
                passages.extend(nominate(content, section=kind, offset=offset))
                pieces.append(content)
                offset += len(content) + 2
            text = "\n\n".join(pieces)
        # The registry namespaces immutable revisions by source, file and corpus.
        # Source bytes determine the revision within that file.
        revision = SessionRevision(
            entry.system,
            entry.source_type,
            f"{namespace}/{entry.source_id}",
            text,
            source_version=entry.source_sha256,
            source_labels={
                "branch": meta.get("branch"),
                "branch_state": "stated" if meta.get("branch") else "unknown",
            },
            source_created_at=next((span.timestamp for span in spans if span.timestamp), 0.0),
        )
        return cls(revision, meta, spans, passages, excluded)

    def author(self, span: EventSpan | None) -> Attribution:
        author = Attribution("decision_author", self.revision.evidence_id)
        if span:
            if span.kind == "text":
                author.set("actor_kind", "bot", "stated", pointer="event.type")
            elif span.origin in ("human", "relay"):
                kind = "human" if span.origin == "human" else "automation"
                author.set("actor_kind", kind, "stated", pointer="event.origin")
            if span.timestamp:
                author.set("occurred_at", span.timestamp, "stated", pointer="event.ts")
        self._session_fields(author, include_model=bool(span and span.kind == "text"))
        return author

    def _session_fields(self, row: Attribution, *, include_model: bool) -> None:
        if self.meta.get("session"):
            row.set("source_session_id", self.meta["session"], "stated", pointer="meta.id")
        if include_model and self.meta.get("model"):
            row.set("original_model", self.meta["model"], "stated", pointer="meta.model")


class SessionHistoryImporter:
    def __init__(
        self,
        registry: Any,
        *,
        access: AccessContext,
        user: str,
        bundle: str | Path,
        mark_cutover: bool = False,
        classifier: Any = None,
    ):
        self.registry = registry
        self.access = access
        self.user = session_subject(user)
        self.bundle = Path(bundle)
        self.mark_cutover = mark_cutover
        self.classifier = classifier
        self.runs = ImportRuns(registry)
        self.store = ProvenanceStore(registry)

    def _policy(self):
        policy = require_import(self.access)
        if self.user not in policy.session_subjects:
            raise AccessDenied("the session subject is not admitted by the corpus policy")
        return policy

    @contextmanager
    def _lock(self, namespace: str) -> Iterator[None]:
        db = self.registry.db
        with db.transaction():
            if db.postgres:
                key = int(digest(f"{IMPORTER}:{namespace}".encode())[:15], 16)
                db.execute("SELECT pg_advisory_xact_lock(?)", (key,))
            self._policy()
            yield

    def _cutover(self, namespace: str) -> bool:
        return bool(
            self.registry.db.execute(
                "SELECT 1 FROM registry_sync_state WHERE source_system = ? AND stream = ?",
                (IMPORTER, namespace),
            )
        )

    def _new_files(self, files: list[BundleEntry], namespace: str) -> list[BundleEntry]:
        ids = [file.evidence_id(namespace) for file in files]
        existing = set()
        for start in range(0, len(ids), 500):
            batch = ids[start : start + 500]
            marks = ", ".join("?" for _ in batch)
            existing.update(
                row[0]
                for row in self.registry.db.execute(
                    f"SELECT evidence_id FROM registry_source_revisions WHERE evidence_id IN ({marks})",
                    batch,
                )
            )
        pending = [
            file
            for file, evidence_id in zip(files, ids, strict=True)
            if evidence_id not in existing
        ]
        if pending and self._cutover(namespace):
            raise RegistryError("session cutover is complete; the bundle contains a new revision")
        return pending

    def _load(self) -> tuple[list[BundleEntry], list[str], str]:
        try:
            path = self.bundle / "manifest.json"
            if path.is_symlink():
                raise ValueError("manifest must be a regular file")
            raw = path.read_bytes()
            manifest = _object(raw)
            if manifest["schema"] != VERSION or manifest["user"] != self.user:
                raise ValueError("manifest does not match this import")
            unavailable = manifest["unavailable"]
            if not isinstance(unavailable, list) or not all(_label(x) for x in unavailable):
                raise ValueError("invalid unavailable sources")
            if not isinstance(manifest["files"], list):
                raise ValueError("invalid file list")
            files = [BundleEntry(**entry) for entry in manifest["files"]]
            if len({(file.system, file.source_id) for file in files}) != len(files):
                raise ValueError("duplicate source file")
        except (OSError, ValueError, KeyError, TypeError):
            raise RegistryError("invalid session-history bundle") from None
        return files, unavailable, digest(raw)

    def run(self, *, apply: bool = False) -> dict:
        policy = self._policy()
        if self.mark_cutover and not apply:
            raise RegistryError("marking cutover requires apply")
        namespace = digest(_json_bytes([policy.corpus_id, self.user]))
        files, unavailable, checksum = self._load()
        counts: Counter[str] = Counter(
            files_in_bundle=len(files),
            files_read=0,
            excluded_events=0,
            stored_evidence=0,
            proposed=0,
        )
        out = {"apply": apply, "unavailable": unavailable, "counts": counts}
        if not apply:
            pending = self._new_files(files, namespace)
            candidate_passages = 0
            for entry in pending:
                file = BundleFile.load(self.bundle, entry, namespace=namespace)
                counts["files_read"] += 1
                counts["excluded_events"] += file.excluded
                candidate_passages += len(file.passages)
            return {**out, "new_revisions": len(pending), "candidate_passages": candidate_passages}

        run = None
        try:
            with self._lock(namespace):
                pending = self._new_files(files, namespace)
                if not pending:
                    if self.mark_cutover and files and not self._cutover(namespace):
                        record_cursor(
                            self.registry,
                            IMPORTER,
                            namespace,
                            watermark=self.registry.clock(),
                            cursor=checksum,
                        )
                    return {**out, "cutover": self._cutover(namespace)}
                classifier = self.classifier or decision_llm.default_classifier(
                    self.registry, corpus_id=policy.corpus_id
                )
                if classifier.corpus_id != policy.corpus_id:
                    raise AccessDenied("classifier cache belongs to another corpus")
                classifier.validate_configuration()
                run = self.runs.start(
                    self.access,
                    source_system=IMPORTER,
                    stream=f"session:{self.user}",
                    importer=IMPORTER,
                    importer_version=VERSION,
                    scope={
                        "bundle_sha256": checksum,
                        "unavailable": unavailable,
                        "model": classifier.model,
                        "policy_sha256": policy.manifest_sha256,
                        "operator": self.access.principal,
                    },
                    extractor=decision_llm.EXTRACTOR,
                    extractor_version=decision_llm.EXTRACTOR_VERSION,
                    extractor_model=classifier.model,
                )
            classifier.import_run_id = run["run_id"]
            for entry in pending:
                with self._lock(namespace):
                    if not self._new_files([entry], namespace):
                        continue
                file = BundleFile.load(self.bundle, entry, namespace=namespace)
                candidates = classifier.classify(file.passages)
                if getattr(classifier, "skipped", None):
                    raise RegistryError("classification incomplete for a session-history file")
                with self._lock(namespace):
                    if not self._new_files([entry], namespace):
                        continue
                    run = self.runs.get(run["run_id"])
                    with self.runs.page(run, self.access) as page_counts:
                        page_counts["files_read"] += 1
                        page_counts["excluded_events"] += file.excluded
                        self._record(run, file, candidates, page_counts)
                        run["next_checkpoint"] = {"evidence_id": file.revision.evidence_id}
                    counts.update(page_counts)
            with self._lock(namespace):
                run = self.runs.get(run["run_id"])
                self.runs.finish(run, complete=True)
                if self.mark_cutover and not self._cutover(namespace):
                    record_cursor(
                        self.registry,
                        IMPORTER,
                        namespace,
                        watermark=self.registry.clock(),
                        cursor=checksum,
                    )
            return {**out, "cutover": self._cutover(namespace)}
        except Exception:
            if run is not None:
                with self.registry.db.transaction():
                    current = self.runs.get(run["run_id"])
                    if current["status"] == "running":
                        self.runs.finish(
                            current,
                            complete=False,
                            errors=["session history import interrupted"],
                        )
            raise

    def _record(self, run: dict, file: BundleFile, candidates: list, counts: Counter[str]) -> None:
        corpus = (run["corpus_id"], run["policy_version"])
        rev = file.revision
        eid, created = self.store.record_revision(rev, corpus=corpus, run_id=run["run_id"])
        if not created:
            return
        counts["stored_evidence"] += 1
        attribution = Attribution("source_account", eid)
        file._session_fields(attribution, include_model=True)
        self.store.record_attributions(
            [
                attribution,
                tool_attribution("importer", eid, identity=IMPORT_ACTOR_SUB, version=VERSION),
            ],
            corpus_id=corpus[0],
        )
        for candidate in candidates:
            self.store.validate_quote(eid, candidate.quote, candidate.start, candidate.end)
            span = next(
                (
                    span
                    for span in file.spans
                    if span.start <= candidate.start < candidate.end <= span.end
                ),
                None,
            )
            if file.spans and span is None:
                raise QuoteMismatch("quote must lie within one transcript event")
            author = file.author(span)
            ids = materialize(
                self.registry,
                run=run,
                ctx=self.access,
                source_system=rev.source_system,
                evidence_id=eid,
                source_url="",
                repo="",
                candidates=[candidate],
                attributions=[author],
                counts=counts,
                identity=rev.source_id,
                occurred_at=span.timestamp if span else 0,
                ticket=file.meta.get("ticket", ""),
                session_history=True,
            )
            if not ids:
                raise RegistryError("could not materialize a session decision")
            author.object_id = ids[0]
            author.decision_revision = 1
            self.store.record_attributions([author], corpus_id=corpus[0])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mira.registry.session_history")
    commands = parser.add_subparsers(dest="command", required=True)
    exporter = commands.add_parser("export")
    importer = commands.add_parser("import")
    for command in (exporter, importer):
        command.add_argument("--user", required=True)
        command.add_argument("--bundle", required=True, type=Path)
    exporter.add_argument("--source", action="append", required=True, metavar="TYPE=DIR")
    exporter.add_argument("--unavailable", action="append", default=[])
    importer.add_argument("--apply", action="store_true")
    importer.add_argument("--mark-cutover", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            sources = []
            for value in args.source:
                kind, separator, path = value.partition("=")
                if not separator or not path:
                    parser.error("--source requires TYPE=DIR")
                sources.append((kind, Path(path)))
            out = export_bundle(
                args.bundle,
                user=args.user,
                sources=sources,
                unavailable=args.unavailable,
            )
        else:
            from mira.registry import get_registry
            from mira.registry.access import operator_context

            access = operator_context()
            if session_subject(args.user) not in require_import(access).session_subjects:
                raise AccessDenied("the session subject is not admitted by the corpus policy")
            settings = {
                "access": access,
                "user": args.user,
                "bundle": args.bundle,
                "mark_cutover": args.mark_cutover,
            }
            if args.apply:
                registry = get_registry()
                assert registry is not None
                out = SessionHistoryImporter(registry, **settings).run(apply=True)
            else:
                from mira.registry.db import RegistryDB, default_location
                from mira.registry.service import Registry

                url, path = default_location()
                with closing(RegistryDB(url=url, path=path, read_only=True)) as db:
                    out = SessionHistoryImporter(Registry(db), **settings).run()
    except (RegistryError, ValueError, OSError):
        # A raw exception can contain private paths, quotes or source content.
        parser.exit(1, "session history operation failed; check bundle, policy and run state\n")
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

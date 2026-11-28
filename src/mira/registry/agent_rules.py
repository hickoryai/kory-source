"""Mirror AGENTS.md hard rules into the registry as rules anchored to hook files.

A bullet in any ``AGENTS.md`` of a configured repository that names an existing
file under the configured hook directory becomes one proposed ``rule``, with
one path anchor per named hook file. Every other bullet is listed in the run
report as an exception and is not written. AGENTS.md is the source of truth.

Configuration is runtime only; with any of the first two unset this is off:

- ``REGISTRY_AGENT_RULES_REPOS``: space-separated ``owner/name`` list.
- ``REGISTRY_AGENT_RULES_HOOK_DIR``: hook directory inside each repository.
- ``REGISTRY_AGENT_RULES_MODE``: ``dry-run`` (default) stores only the report;
  ``apply`` writes registry rows.

Runs under the corpus policy's import operator (``REGISTRY_IMPORT_OPERATOR``).
A repository outside the manifest is refused before any GitHub call. GitHub is
read with REST only (head commit, one recursive tree, AGENTS.md blobs), using a
read-only installation token for that one repository. No clone, no model call.

The hourly registry job and the GitHub push webhook both call ``sync_repo``. A
run whose head commit and mode match the last finished run reads nothing more.

A bullet "names" a hook file when a backticked span is the hook's path, its
file name, or its file name without ``.py``. Files whose name starts with ``_``
are helpers, not hooks.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import os
import posixpath
import re
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from mira.registry.access import AccessContext, AccessDenied, require_import
from mira.registry.model import Actor, NotFound

logger = logging.getLogger(__name__)

SOURCE_SYSTEM = "agents_md_rule"
REPOS_ENV = "REGISTRY_AGENT_RULES_REPOS"
HOOK_DIR_ENV = "REGISTRY_AGENT_RULES_HOOK_DIR"
MODE_ENV = "REGISTRY_AGENT_RULES_MODE"
MODES = ("dry-run", "apply")
ACTOR = Actor("agent", "agents-md-mirror", model="none")
AGENTS_FILE = "AGENTS.md"

_REPO = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_BULLET = re.compile(r"^(\s*)(?:[-*+]|\d{1,3}[.)])\s+(.*\S)\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_HEADING = re.compile(r"^\s{0,3}#")
_TICK = re.compile(r"`([^`\n]+)`")
_SYNC_LOCK = threading.Lock()
KEEP_RUNS = 20  # runs kept per repository; reports can be large


class ReadError(RuntimeError):
    """GitHub returned less than a complete read. The run writes nothing."""


# ── configuration ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class Settings:
    repos: tuple[str, ...]
    hook_dir: str
    mode: str = "dry-run"


def settings(env: Mapping[str, str] | None = None) -> Settings | None:
    """The runtime configuration, or None when the mirror is off."""
    env = os.environ if env is None else env
    repos = tuple(r for r in env.get(REPOS_ENV, "").split() if r)
    hook_dir = env.get(HOOK_DIR_ENV, "").strip().strip("/")
    if not repos or not hook_dir:
        return None
    mode = env.get(MODE_ENV, "").strip() or "dry-run"
    if mode not in MODES:
        logger.warning("%s=%r is not one of %s; using dry-run", MODE_ENV, mode, MODES)
        mode = "dry-run"
    return Settings(repos=repos, hook_dir=hook_dir, mode=mode)


# ── reading ─────────────────────────────────────────────────────────────


class Reader(Protocol):
    def head(self, repo: str) -> str: ...
    def tree(self, repo: str, commit: str) -> dict[str, str]: ...
    def blob(self, repo: str, sha: str) -> str: ...


class GitHubReader:
    """GitHub REST reads with one repository-scoped, read-only installation token."""

    def __init__(
        self,
        token_for: Callable[[str], str] | None = None,
        *,
        api: str = "https://api.github.com",
        http: httpx.Client | None = None,
    ) -> None:
        self._token_for = token_for or _scoped_token
        self._api = api
        self._http = http
        self._tokens: dict[str, str] = {}

    def _get(self, repo: str, path: str, accept: str) -> httpx.Response:
        token = self._tokens.get(repo)
        if token is None:
            token = self._tokens[repo] = self._token_for(repo)
        headers = {"Authorization": f"Bearer {token}", "Accept": accept}
        try:
            with nullcontext(self._http) if self._http else httpx.Client(timeout=30.0) as c:
                resp = c.get(f"{self._api}/repos/{repo}{path}", headers=headers)
        except httpx.HTTPError as exc:
            raise ReadError(f"GitHub read failed ({type(exc).__name__})") from None
        if resp.status_code != 200:
            raise ReadError(f"GitHub returned HTTP {resp.status_code}")
        return resp

    def head(self, repo: str) -> str:
        meta = self._get(repo, "", "application/vnd.github+json").json()
        branch = str(meta.get("default_branch") or "")
        if not branch:
            raise ReadError("repository has no default branch")
        sha = self._get(repo, f"/commits/{branch}", "application/vnd.github.sha").text.strip()
        if not re.fullmatch(r"[0-9a-f]{40,64}", sha):
            raise ReadError("GitHub returned no head commit")
        return sha

    def tree(self, repo: str, commit: str) -> dict[str, str]:
        data = self._get(
            repo, f"/git/trees/{commit}?recursive=1", "application/vnd.github+json"
        ).json()
        if data.get("truncated") or not isinstance(data.get("tree"), list):
            raise ReadError("GitHub returned a truncated tree")
        return {e["path"]: e["sha"] for e in data["tree"] if e.get("type") == "blob"}

    def blob(self, repo: str, sha: str) -> str:
        return self._get(repo, f"/git/blobs/{sha}", "application/vnd.github.raw").text


def _scoped_token(repo: str) -> str:
    from mira.registry.checkout import scoped_read_token

    return scoped_read_token(
        repo,
        os.environ.get("MIRA_GITHUB_APP_ID", ""),
        os.environ.get("MIRA_GITHUB_PRIVATE_KEY", ""),
    )


# ── parsing ─────────────────────────────────────────────────────────────


@dataclass
class Bullet:
    agents_path: str
    line: int
    text: str
    hooks: tuple[str, ...] = ()
    fingerprint: str = ""

    @property
    def content_hash(self) -> str:
        return _sha(self.text + "\n" + "\n".join(self.hooks))


@dataclass
class Parsed:
    bullets: list[Bullet] = field(default_factory=list)  # bullets naming a hook
    exceptions: list[Bullet] = field(default_factory=list)
    paragraph_mentions: list[dict] = field(default_factory=list)
    bullet_count: int = 0


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def hook_files(tree: Iterable[str], hook_dir: str) -> list[str]:
    prefix = hook_dir.strip("/") + "/"
    return sorted(
        p
        for p in tree
        if p.startswith(prefix) and p.endswith(".py") and not posixpath.basename(p).startswith("_")
    )


def agents_files(tree: Iterable[str]) -> list[str]:
    return sorted(p for p in tree if posixpath.basename(p) == AGENTS_FILE)


class HookNames:
    """Which hook files a backticked span names: path, file name, or stem."""

    def __init__(self, hooks: Iterable[str]) -> None:
        self.paths = set(hooks)
        by_name: dict[str, set[str]] = {}
        for path in self.paths:
            base = posixpath.basename(path)
            for name in (base, base.removesuffix(".py")):
                by_name.setdefault(name, set()).add(path)
        # A file name two hooks share names neither; only the full path does.
        self.by_name = {n: next(iter(p)) for n, p in by_name.items() if len(p) == 1}

    def resolve(self, span: str) -> str | None:
        token = span.strip().removeprefix("./")
        if token in self.paths:
            return token
        return self.by_name.get(token)

    def scan(self, text: str) -> tuple[tuple[str, ...], str]:
        """Named hook paths (sorted, unique) and the text with each hook name masked."""
        found: set[str] = set()

        def mask(m: re.Match) -> str:
            path = self.resolve(m.group(1))
            if path is None:
                return m.group(0)
            found.add(path)
            return "`<hook>`"

        masked = _TICK.sub(mask, text)
        return tuple(sorted(found)), masked


def parse(agents_path: str, text: str, names: HookNames) -> Parsed:
    """Bullets and paragraph lines of one AGENTS.md, outside fenced code."""
    out = Parsed()
    items: list[tuple[int, list[str]]] = []
    paragraphs: list[tuple[int, str]] = []
    in_fence = False
    current: tuple[int, int, list[str]] | None = None  # (line, indent, parts)

    def close() -> None:
        nonlocal current
        if current is not None:
            items.append((current[0], current[2]))
            current = None

    for i, raw in enumerate(text.splitlines(), start=1):
        if _FENCE.match(raw):
            close()
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if not raw.strip():
            close()
            continue
        bullet = _BULLET.match(raw)
        if bullet:
            close()
            current = (i, len(bullet.group(1)), [bullet.group(2)])
            continue
        indent = len(raw) - len(raw.lstrip())
        if current is not None and indent > current[1] and not _HEADING.match(raw):
            current[2].append(raw.strip())
            continue
        close()
        if not _HEADING.match(raw):
            paragraphs.append((i, raw.strip()))
    close()

    for line, parts in items:
        body = " ".join(parts)
        hooks, masked = names.scan(body)
        b = Bullet(agents_path, line, body, hooks, _sha(" ".join(masked.lower().split())))
        out.bullet_count += 1
        (out.bullets if hooks else out.exceptions).append(b)
    for line, body in paragraphs:
        hooks, _ = names.scan(body)
        if hooks:
            out.paragraph_mentions.append(
                {"agents_path": agents_path, "line": line, "hooks": list(hooks)}
            )
    return out


# ── matching ────────────────────────────────────────────────────────────


@dataclass
class Identity:
    identity_id: str
    repo: str
    object_id: str
    agents_path: str
    fingerprint: str
    body: str
    hook_paths: tuple[str, ...]
    hook_blobs: dict[str, str]
    content_hash: str
    version: int
    state: str
    prior_status: str


_ID_COLS = (
    "identity_id, repo, object_id, agents_path, fingerprint, body, hook_paths, hook_blobs, "
    "content_hash, version, state, prior_status"
)


def _identity(row: tuple) -> Identity:
    return Identity(
        row[0],
        row[1],
        row[2],
        row[3],
        row[4],
        row[5],
        tuple(json.loads(row[6] or "[]")),
        json.loads(row[7] or "{}"),
        row[8],
        int(row[9]),
        row[10],
        row[11],
    )


@dataclass
class Match:
    bullet: Bullet
    identity: Identity | None  # None = add
    how: str  # unchanged | edit | rename | move | agents_moved | reappear | add


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def match(
    bullets: list[Bullet], identities: list[Identity], tree: Mapping[str, str]
) -> tuple[list[Match], list[Identity]]:
    """Pair current bullets with stored identities. Returns matches and unmatched active ids.

    Order: unchanged; same file and same hooks (edit); same file and same
    masked text (rename); old hook paths gone with their blobs at the named
    paths (move); same text and hooks with the old AGENTS.md gone (file
    moved); a removed identity with the same file, text and hooks
    (reappear). Anything else is an add.
    """
    active = [i for i in identities if i.state == "active"]
    removed = [i for i in identities if i.state != "active"]
    taken: set[str] = set()
    out: dict[int, Match] = {}

    def run(how: str, pool: list[Identity], ok: Callable[[Bullet, Identity], bool]) -> None:
        for n, b in enumerate(bullets):
            if n in out:
                continue
            cands = [i for i in pool if i.identity_id not in taken and ok(b, i)]
            if not cands:
                continue
            best = max(cands, key=lambda i: _similarity(b.text, i.body))
            taken.add(best.identity_id)
            out[n] = Match(b, best, how)

    run(
        "unchanged",
        active,
        lambda b, i: (
            b.agents_path == i.agents_path
            and b.fingerprint == i.fingerprint
            and b.hooks == i.hook_paths
        ),
    )
    run("edit", active, lambda b, i: b.agents_path == i.agents_path and b.hooks == i.hook_paths)
    run(
        "rename",
        active,
        lambda b, i: b.agents_path == i.agents_path and b.fingerprint == i.fingerprint,
    )

    def moved(b: Bullet, i: Identity) -> bool:
        gone = [p for p in i.hook_paths if p not in tree]
        if not gone or b.agents_path != i.agents_path:
            return False
        new = [p for p in b.hooks if p not in i.hook_paths]
        return sorted(i.hook_blobs.get(p, "") for p in gone) == sorted(tree[p] for p in new)

    run("move", active, moved)
    run(
        "agents_moved",
        active,
        lambda b, i: (
            i.agents_path not in tree and b.fingerprint == i.fingerprint and b.hooks == i.hook_paths
        ),
    )
    run(
        "reappear",
        removed,
        lambda b, i: (
            b.agents_path == i.agents_path
            and b.fingerprint == i.fingerprint
            and b.hooks == i.hook_paths
        ),
    )
    matches = [out.get(n) or Match(b, None, "add") for n, b in enumerate(bullets)]
    gone = [i for i in active if i.identity_id not in taken]
    return matches, gone


def _changed(ident: Identity, b: Bullet) -> bool:
    """A new version: other text, other hook paths, or a moved AGENTS.md."""
    return ident.content_hash != b.content_hash or ident.agents_path != b.agents_path


def identity_id(repo: str, bullet: Bullet) -> str:
    """Deterministic, so two concurrent applies of one bullet write one row."""
    key = "\n".join((repo.lower(), bullet.agents_path, bullet.fingerprint, *bullet.hooks))
    return _sha(key)[:32]


def _new_identity_id(registry: Any, repo: str, bullet: Bullet) -> str:
    """The deterministic id, unless an older rule already holds it.

    An identity keeps the id of its first text through edits, renames and
    moves, so a new bullet with that first text must not take it over.
    """
    candidate = identity_id(repo, bullet)
    taken = registry.db.execute(
        "SELECT 1 FROM registry_agent_rule_identities WHERE identity_id = ?", (candidate,)
    )
    return uuid.uuid4().hex if taken else candidate


# ── sync ────────────────────────────────────────────────────────────────


def _short(b: Bullet) -> dict:
    return {
        "agents_path": b.agents_path,
        "line": b.line,
        "text": b.text,
        "hooks": list(b.hooks),
    }


def last_finished(registry: Any, repo: str) -> dict | None:
    rows = registry.db.execute(
        "SELECT run_id, head_commit, mode, report FROM registry_agent_rule_runs "
        "WHERE repo = ? AND status = 'finished' ORDER BY finished_at DESC, run_id LIMIT 1",
        (repo,),
    )
    if not rows:
        return None
    try:
        hook_dir = str(json.loads(rows[0][3] or "{}").get("hook_dir") or "")
    except ValueError:
        hook_dir = ""
    return {"run_id": rows[0][0], "head": rows[0][1], "mode": rows[0][2], "hook_dir": hook_dir}


def _save_run(registry: Any, run: dict) -> None:
    with registry.db.transaction():
        registry.db.execute(
            "DELETE FROM registry_agent_rule_runs WHERE run_id = ?", (run["run_id"],)
        )
        registry.db.execute(
            "INSERT INTO registry_agent_rule_runs (run_id, repo, corpus_id, head_commit, mode, "
            "trigger_name, status, error, report, started_at, finished_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run["run_id"],
                run["repo"],
                run["corpus_id"],
                run["head"],
                run["mode"],
                run["trigger"],
                run["status"],
                run["error"],
                json.dumps(run["report"], sort_keys=True),
                run["started_at"],
                run["finished_at"],
            ),
        )
        old = registry.db.execute(
            "SELECT run_id FROM registry_agent_rule_runs WHERE repo = ? "
            "ORDER BY started_at DESC, run_id",
            (run["repo"],),
        )[KEEP_RUNS:]
        for (run_id,) in old:
            registry.db.execute("DELETE FROM registry_agent_rule_runs WHERE run_id = ?", (run_id,))


def sync_repo(
    registry: Any,
    repo: str,
    *,
    access: AccessContext,
    conf: Settings,
    reader: Reader | None = None,
    trigger: str = "hourly",
) -> dict:
    """Mirror one repository's AGENTS.md rules. Both triggers call this.

    Returns the run summary. Never retires or writes anything after a partial
    read: the run is saved as failed and the next trigger retries.
    """
    with _SYNC_LOCK:
        return _sync_repo(registry, repo, access, conf, reader, trigger)


def _sync_repo(
    registry: Any,
    repo: str,
    access: AccessContext,
    conf: Settings,
    reader: Reader | None,
    trigger: str,
) -> dict:
    clock = getattr(registry, "clock", time.time)
    run: dict[str, Any] = {
        "run_id": uuid.uuid4().hex,
        "repo": repo,
        "corpus_id": "",
        "head": "",
        "mode": conf.mode,
        "trigger": trigger,
        "status": "failed",
        "error": "",
        "report": {},
        "started_at": clock(),
        "finished_at": 0.0,
    }
    try:
        if not _REPO.fullmatch(repo):
            raise AccessDenied("repository must be owner/name")
        policy = require_import(access)
        if not policy.admits_repository(repo):
            raise AccessDenied("the repository is not in the approved corpus manifest")
        run["corpus_id"] = policy.corpus_id
        corpus = (policy.corpus_id, policy.policy_version)
        reader = reader or GitHubReader()
        run["head"] = reader.head(repo)
        last = last_finished(registry, repo)
        if last and (last["head"], last["mode"], last["hook_dir"]) == (
            run["head"],
            conf.mode,
            conf.hook_dir,
        ):
            return {**_summary(run), "status": "unchanged", "run_id": last["run_id"]}
        tree = reader.tree(repo, run["head"])
        report = _compute(registry, repo, run["head"], tree, reader, conf)
        if conf.mode == "apply":
            _apply(registry, repo, run["head"], tree, report.pop("_plan"), corpus, clock)
        else:
            report.pop("_plan")
        run.update(status="finished", report=report, finished_at=clock())
    except (ReadError, AccessDenied) as exc:
        run.update(error=str(exc), finished_at=clock())
        logger.warning("AGENTS.md rule sync of %s failed: %s", repo, exc)
    except Exception as exc:
        run.update(error=f"{type(exc).__name__}", finished_at=clock())
        logger.warning("AGENTS.md rule sync of %s failed: %s", repo, type(exc).__name__)
    _save_run(registry, run)
    return _summary(run)


def _summary(run: dict) -> dict:
    return {
        "run_id": run["run_id"],
        "repo": run["repo"],
        "head": run["head"],
        "mode": run["mode"],
        "status": run["status"],
        "error": run["error"],
        "counts": (run["report"] or {}).get("counts", {}),
    }


def _load_identities(registry: Any, repo: str) -> list[Identity]:
    rows = registry.db.execute(
        f"SELECT {_ID_COLS} FROM registry_agent_rule_identities WHERE repo = ?",  # noqa: S608
        (repo,),
    )
    return [_identity(r) for r in rows]


def _compute(
    registry: Any,
    repo: str,
    head: str,
    tree: dict[str, str],
    reader: Reader,
    conf: Settings,
) -> dict:
    hooks = hook_files(tree, conf.hook_dir)
    names = HookNames(hooks)
    parsed = Parsed()
    files = agents_files(tree)
    for path in files:
        one = parse(path, reader.blob(repo, tree[path]), names)
        parsed.bullets += one.bullets
        parsed.exceptions += one.exceptions
        parsed.paragraph_mentions += one.paragraph_mentions
        parsed.bullet_count += one.bullet_count
    unique: dict[tuple, Bullet] = {}
    for b in parsed.bullets:  # the same rule twice in one file is one rule
        unique.setdefault((b.agents_path, b.fingerprint, b.hooks), b)
    duplicates = len(parsed.bullets) - len(unique)
    parsed.bullets = list(unique.values())
    identities = _load_identities(registry, repo)
    matches, gone = match(parsed.bullets, identities, tree)
    named = {h for b in parsed.bullets for h in b.hooks}
    groups: dict[str, list[dict]] = {
        "create": [],
        "update": [],
        "re_anchor": [],
        "restore": [],
        "retire": [],
    }
    for m in matches:
        entry = _short(m.bullet)
        if m.identity is not None:
            entry["object_id"] = m.identity.object_id
            entry["how"] = m.how
        if m.how == "add":
            groups["create"].append(entry)
        elif m.how == "reappear":
            groups["restore"].append(entry)
        elif m.identity is not None and _changed(m.identity, m.bullet):
            if m.bullet.hooks != m.identity.hook_paths:
                groups["re_anchor"].append(entry)
            else:
                groups["update"].append(entry)
    for ident in gone:
        groups["retire"].append(
            {
                "object_id": ident.object_id,
                "agents_path": ident.agents_path,
                "text": ident.body,
                "hooks": list(ident.hook_paths),
            }
        )
    return {
        "head": head,
        "mode": conf.mode,
        "hook_dir": conf.hook_dir,
        "counts": {
            "agents_files": len(files),
            "bullets": parsed.bullet_count,
            "rule_bullets": len(parsed.bullets),
            "duplicate_rule_bullets": duplicates,
            "exceptions": len(parsed.exceptions),
            "paragraph_mentions": len(parsed.paragraph_mentions),
            "hooks": len(hooks),
            "hooks_named": len(named),
            "hooks_unnamed": len(hooks) - len(named),
            **{k: len(v) for k, v in groups.items()},
        },
        **groups,
        "exceptions": [_short(b) for b in parsed.exceptions],
        "paragraph_mentions": parsed.paragraph_mentions,
        "unnamed_hooks": [h for h in hooks if h not in named],
        "_plan": (matches, gone),
    }


def _apply(
    registry: Any,
    repo: str,
    head: str,
    tree: dict[str, str],
    plan: tuple[list[Match], list[Identity]],
    corpus: tuple[str, str],
    clock: Callable[[], float],
) -> None:
    matches, gone = plan
    for m in matches:
        _apply_match(registry, repo, head, tree, m, corpus, clock)
    for ident in gone:
        _retire(registry, ident, clock)


def _apply_match(
    registry: Any,
    repo: str,
    head: str,
    tree: dict[str, str],
    m: Match,
    corpus: tuple[str, str],
    clock: Callable[[], float],
) -> None:
    b = m.bullet
    ident = m.identity
    changed = ident is None or _changed(ident, b)
    if not changed and m.how != "reappear":
        return
    version = 1 if ident is None else ident.version + int(changed)
    if ident is None:
        ident = Identity(
            _new_identity_id(registry, repo, b),
            repo,
            "",
            b.agents_path,
            "",
            "",
            (),
            {},
            "",
            0,
            "active",
            "",
        )
    item = {
        "source_id": f"{repo}/agents-md/{ident.identity_id}",
        "source_version": version,
        "kind": "rule",
        "title": b.text[:120],
        "body": b.text,
        "summary": f"AGENTS.md rule in {b.agents_path}",
        "repo": repo,
        "scope": "repo",
        "source_url": f"https://github.com/{repo}/blob/{head}/{b.agents_path}",
    }
    [res] = registry.import_items(
        SOURCE_SYSTEM, [item], ACTOR, status="proposed", basis="statement", corpus=corpus
    )
    if "object_id" not in res:
        raise RuntimeError(f"registry import failed: {res.get('error', '')}")
    object_id = res["object_id"]
    ident.object_id = object_id
    with registry._object_lock(object_id):
        obj = registry.get(object_id)
        if (
            m.how == "reappear"
            and obj["status"] == "retired"
            and (obj["status_reason"] == "source_removed")
        ):
            obj.update(
                status=ident.prior_status or "proposed", status_reason="", sync_status="synced"
            )
            registry._write(obj)
        registry.db.execute(
            "DELETE FROM registry_anchors WHERE object_id = ? AND qualified_name = ''",
            (object_id,),
        )
        for path in b.hooks:
            registry.db.execute(
                "INSERT INTO registry_anchors (object_id, repo, path, qualified_name, "
                "symbol_kind, anchor_id, created_at, corpus_id) VALUES (?, ?, ?, '', '', '', ?, ?)",
                (object_id, repo, path, clock(), obj["corpus_id"]),
            )
        registry.db.execute(
            "DELETE FROM registry_agent_rule_versions WHERE object_id = ? AND version = ?",
            (object_id, version),
        )
        registry.db.execute(
            "INSERT INTO registry_agent_rule_versions (object_id, version, body, hook_paths, "
            "agents_path, head_commit, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (object_id, version, b.text, json.dumps(list(b.hooks)), b.agents_path, head, clock()),
        )
        ident.state = "active"
        _store_identity(registry, ident, b, tree, head, version, clock)


def _store_identity(
    registry: Any,
    ident: Identity,
    b: Bullet,
    tree: Mapping[str, str],
    head: str,
    version: int,
    clock: Callable[[], float],
) -> None:
    with registry.db.transaction():
        registry.db.execute(
            "DELETE FROM registry_agent_rule_identities WHERE identity_id = ?",
            (ident.identity_id,),
        )
        registry.db.execute(
            f"INSERT INTO registry_agent_rule_identities ({_ID_COLS}, head_commit, updated_at) "  # noqa: S608
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ident.identity_id,
                ident.repo,
                ident.object_id,
                b.agents_path,
                b.fingerprint,
                b.text,
                json.dumps(list(b.hooks)),
                json.dumps({p: tree.get(p, "") for p in b.hooks}, sort_keys=True),
                b.content_hash,
                version,
                "active",
                ident.prior_status,
                head,
                clock(),
            ),
        )


def _retire(registry: Any, ident: Identity, clock: Callable[[], float]) -> None:
    """The bullet or its hook file is gone. A human rejection stays a rejection."""
    with registry._object_lock(ident.object_id):
        try:
            obj = registry.get(ident.object_id)
        except NotFound:
            obj = {"status": ""}
        prior = ident.prior_status
        if obj["status"] in ("proposed", "approved"):
            prior = obj["status"]
            obj.update(
                status="retired", status_reason="source_removed", sync_status="source_removed"
            )
            registry._write(obj)
        registry.db.execute(
            "UPDATE registry_agent_rule_identities SET state = 'removed', prior_status = ?, "
            "updated_at = ? WHERE identity_id = ?",
            (prior, clock(), ident.identity_id),
        )


# ── entry points ────────────────────────────────────────────────────────


def sync_configured(
    registry: Any,
    *,
    trigger: str = "hourly",
    repos: Iterable[str] | None = None,
    env: Mapping[str, str] | None = None,
    reader: Reader | None = None,
    access: AccessContext | None = None,
) -> list[dict]:
    """Run ``sync_repo`` for each configured repository (or the given subset).

    Off (returns ``[]``) when the mirror is not configured.
    """
    conf = settings(env)
    if conf is None:
        return []
    wanted = list(conf.repos)
    if repos is not None:
        named = {r.lower() for r in repos}
        wanted = [r for r in wanted if r.lower() in named]
    if not wanted:
        return []
    if access is None:
        from mira.registry.access import operator_context

        try:
            access = operator_context()
        except AccessDenied as exc:
            logger.warning("AGENTS.md rule sync skipped: %s", exc)
            return [{"repo": r, "status": "failed", "error": str(exc)} for r in wanted]
    return [
        sync_repo(registry, r, access=access, conf=conf, reader=reader, trigger=trigger)
        for r in wanted
    ]


def on_push(registry: Any, repo: str, ref: str, default_branch: str) -> list[dict]:
    """Push-webhook trigger: a push to a configured repository's default branch."""
    if not default_branch or ref != f"refs/heads/{default_branch}":
        return []
    return sync_configured(registry, trigger="push", repos=[repo])


def latest_reports(registry: Any, access: AccessContext) -> list[dict]:
    """The newest run per repository whose corpus the caller may read."""
    rows = registry.db.execute(
        "SELECT run_id, repo, corpus_id, head_commit, mode, trigger_name, status, error, "
        "report, started_at, finished_at FROM registry_agent_rule_runs ORDER BY started_at DESC"
    )
    seen: set[str] = set()
    out: list[dict] = []
    for r in rows:
        if r[1] in seen or not access.can_read(r[2]):
            continue
        seen.add(r[1])
        out.append(
            {
                "run_id": r[0],
                "repo": r[1],
                "head": r[3],
                "mode": r[4],
                "trigger": r[5],
                "status": r[6],
                "error": r[7],
                "report": json.loads(r[8] or "{}"),
                "started_at": r[9],
                "finished_at": r[10],
            }
        )
    return out

"""MySQL backend: a psycopg-shaped connection over PyMySQL.

The stores keep one SQL dialect for shared servers, the one their Postgres
branch already speaks. This module gives MySQL a connection whose cursor accepts
that SQL, translates it, and returns rows, row counts and errors the way the
Postgres branch expects. Postgres and SQLite SQL is untouched.

What the translation covers:

- ``CREATE TABLE``: ``TEXT`` becomes ``LONGTEXT``, ``SERIAL`` becomes
  ``AUTO_INCREMENT``, ``TIMESTAMPTZ`` becomes ``DATETIME(6)`` in UTC. Tables use
  ``utf8mb4`` with the binary ``utf8mb4_0900_bin`` collation, so equality,
  uniqueness and ``LIKE`` are case-sensitive as on Postgres.
- Keys: InnoDB cannot index unbounded text. A primary or unique key with a text
  column is enforced on a stored, invisible generated column holding
  ``SHA2(JSON_ARRAY(columns))``, so no new length limit applies. A prefix
  index on the same columns serves lookups. Foreign keys over text columns
  reference the parent's hashed key the same way.
- ``CREATE INDEX IF NOT EXISTS`` and ``ADD COLUMN [IF NOT EXISTS]`` run behind
  an ``information_schema`` check, so a rerun after a partial failure skips
  what already exists. A partial unique index keys only the rows its
  ``WHERE`` admits.
- ``ON CONFLICT .. DO UPDATE`` becomes ``ON DUPLICATE KEY UPDATE`` with the
  ``excluded`` row alias. ``DO NOTHING`` is a plain insert whose duplicate-key
  error is caught (``INSERT IGNORE`` would also hide truncation and NOT NULL
  errors). ``INSERT .. RETURNING`` returns the supplied key, or the new
  ``AUTO_INCREMENT`` id.
- ``pg_advisory_xact_lock`` becomes ``GET_LOCK``, released when the
  transaction ends; ``pg_try_advisory_lock`` becomes ``GET_LOCK(.., 0)``.
- ``ILIKE``, ``EXTRACT(EPOCH FROM ..)``, ``to_timestamp`` and ``NOW()`` are
  rewritten.
  ``||`` concatenates through the session's ``PIPES_AS_CONCAT`` mode.

Anything else Postgres-only fails closed with :class:`UnsupportedSQL` before it
reaches the server. Conditional upserts (``DO UPDATE .. WHERE``) and
``UPDATE .. RETURNING`` have no MySQL equivalent, so their call sites branch.
"""

from __future__ import annotations

import datetime as dt
import functools
import hashlib
import re
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

# Row alias in upserts needs 8.0.19, INVISIBLE columns 8.0.23.
MIN_VERSION = (8, 0, 23)
CHARSET = "utf8mb4"
COLLATION = "utf8mb4_0900_bin"
TABLE_OPTIONS = f"ENGINE=InnoDB DEFAULT CHARSET={CHARSET} COLLATE={COLLATION}"
SQL_MODE = (
    "STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,"
    "NO_ENGINE_SUBSTITUTION,ONLY_FULL_GROUP_BY,PIPES_AS_CONCAT,ANSI_QUOTES"
)
SESSION_SETUP = (
    f"SET SESSION sql_mode = '{SQL_MODE}'",
    "SET SESSION time_zone = '+00:00'",
    "SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED",
    f"SET NAMES {CHARSET} COLLATE {COLLATION}",
)
# InnoDB caps an index key at 3072 bytes; utf8mb4 is up to 4 bytes a character.
INDEX_KEY_BYTES = 3072
MAX_PREFIX_CHARS = 255
# Column names in the schemas that MySQL reserves. Quoted in translated DML.
RESERVED_COLUMNS = ("key", "signal")

DUPLICATE_KEY = 1062
LOST_CONNECTION = frozenset({0, 2006, 2013, 2055})

_LOCK_PREFIX = "mira:"
_TEXT_TYPES = frozenset({"TEXT", "LONGTEXT"})


class UnsupportedSQL(ValueError):
    """SQL that has no MySQL translation here. The caller must branch."""


# ---------------------------------------------------------------------------
# Connecting


def connect_kwargs(url: str) -> dict[str, Any]:
    """PyMySQL keyword arguments for ``mysql://`` or ``mysql+pymysql://`` URLs.

    TLS parameters are described in :func:`tls_settings`. Certificate files
    are read here, so they must exist.
    """
    from pymysql.constants import CLIENT

    parts = urlsplit(url)
    if parts.scheme not in ("mysql", "mysql+pymysql"):
        raise ValueError("not a MySQL URL")
    database = unquote(parts.path.lstrip("/"))
    if not database:
        raise ValueError("the MySQL URL must name a database")
    query = {k.lower().replace("-", "_"): v[-1] for k, v in parse_qs(parts.query).items()}
    kwargs: dict[str, Any] = {
        "host": parts.hostname or "localhost",
        "port": parts.port or 3306,
        "user": unquote(parts.username or ""),
        "password": unquote(parts.password or ""),
        "database": database,
        "charset": CHARSET,
        "collation": COLLATION,
        "autocommit": True,
        # UPDATE row counts report matched rows, as on Postgres and SQLite.
        "client_flag": CLIENT.FOUND_ROWS,
        "conv": _decoders(),
    }
    if "connect_timeout" in query:
        kwargs["connect_timeout"] = int(query["connect_timeout"])
    tls = tls_settings(query)
    if tls is DISABLED:
        kwargs["ssl_disabled"] = True
    elif isinstance(tls, TLS):
        # A ready SSLContext is used as given. PyMySQL otherwise rebuilds its
        # TLS settings from separate arguments and can drop the CA, the client
        # certificate, or the requirement to encrypt at all.
        kwargs["ssl"] = tls.context()
    return kwargs


@dataclass(frozen=True)
class TLS:
    """TLS the connection must use. Without it the driver's default applies."""

    ca: str | None = None
    cert: str | None = None
    key: str | None = None
    verify_cert: bool = False
    verify_identity: bool = False

    def context(self) -> Any:
        import ssl

        ctx = ssl.create_default_context(cafile=self.ca)
        # Python 3.13 enables strict X.509 checks that MySQL's auto-generated
        # self-signed certificates fail; the CA and host checks below remain.
        ctx.verify_flags &= ~getattr(ssl, "VERIFY_X509_STRICT", 0)
        ctx.check_hostname = self.verify_identity
        ctx.verify_mode = ssl.CERT_REQUIRED if self.verify_cert else ssl.CERT_NONE
        if self.cert:
            ctx.load_cert_chain(self.cert, keyfile=self.key)
        return ctx


DISABLED = "disabled"


def tls_settings(query: dict[str, str]) -> TLS | str | None:
    """TLS requirement from URL query parameters.

    ``ssl_ca`` verifies the server certificate against that CA. ``ssl_cert`` and
    ``ssl_key`` present a client certificate. ``ssl_mode=REQUIRED`` encrypts
    (and verifies only when ``ssl_ca`` is given), ``VERIFY_CA`` verifies
    against ``ssl_ca`` or the system trust store, ``VERIFY_IDENTITY`` also
    checks the host name, ``DISABLED`` turns TLS off. Any of these makes TLS
    mandatory: the connection fails rather than continuing in plaintext.
    With none of them, TLS is not required.
    """
    ca, cert, key = (query.get(f"ssl_{name}") or None for name in ("ca", "cert", "key"))
    mode = query.get("ssl_mode", "").upper()
    if mode == "PREFERRED":
        # MySQL's own clients fall back to plaintext when the server lacks TLS;
        # a URL that asks for TLS here must get it, so this mode is refused.
        raise ValueError("ssl_mode=PREFERRED is not supported; use REQUIRED or VERIFY_IDENTITY")
    if mode == "DISABLED":
        if ca or cert or key:
            raise ValueError("ssl_mode=DISABLED conflicts with ssl_ca, ssl_cert or ssl_key")
        return DISABLED
    if mode not in ("", "REQUIRED", "VERIFY_CA", "VERIFY_IDENTITY"):
        raise ValueError(f"unknown ssl_mode {mode!r}")
    if key and not cert:
        raise ValueError("ssl_key needs ssl_cert")
    if not (mode or ca or cert):
        return None
    return TLS(
        ca=ca,
        cert=cert,
        key=key,
        verify_cert=bool(ca) or mode in ("VERIFY_CA", "VERIFY_IDENTITY"),
        verify_identity=mode == "VERIFY_IDENTITY",
    )


def connect(url: str) -> MySQLConnection:
    """Open an autocommit connection with the session settings applied."""
    import pymysql

    raw = pymysql.connect(**connect_kwargs(url))
    try:
        check_server(raw.get_server_info())
        with raw.cursor() as cur:
            for statement in SESSION_SETUP:
                cur.execute(statement)
    except BaseException:
        with suppress(Exception):
            raw.close()
        raise
    return MySQLConnection(raw)


def check_server(version: str) -> None:
    """Refuse MariaDB and MySQL older than 8.0.23 with a clear message."""
    if "mariadb" in version.lower():
        raise RuntimeError(f"MariaDB is not supported (server reports {version})")
    numbers = tuple(int(n) for n in re.findall(r"\d+", version)[:3])
    if numbers < MIN_VERSION:
        need = ".".join(map(str, MIN_VERSION))
        raise RuntimeError(f"MySQL {need} or newer is required (server reports {version})")


def _decoders() -> dict[Any, Any]:
    from pymysql.constants import FIELD_TYPE
    from pymysql.converters import conversions, convert_datetime

    def as_number(value: str | bytes) -> int | float:
        text = value.decode() if isinstance(value, bytes) else value
        return float(text) if "." in text or "e" in text.lower() else int(text)

    def as_utc(value: str | bytes) -> Any:
        parsed = convert_datetime(value)
        if isinstance(parsed, dt.datetime):
            return parsed.replace(tzinfo=dt.UTC)
        return parsed

    conv = dict(conversions)
    # SUM and AVG return DECIMAL; the stores expect plain numbers.
    conv[FIELD_TYPE.DECIMAL] = as_number
    conv[FIELD_TYPE.NEWDECIMAL] = as_number
    # Only BOOLEAN columns are declared TINYINT, so this matches Postgres bool.
    conv[FIELD_TYPE.TINY] = lambda value: bool(int(value))
    # The session runs in UTC; return aware datetimes like TIMESTAMPTZ does.
    conv[FIELD_TYPE.DATETIME] = as_utc
    conv[FIELD_TYPE.TIMESTAMP] = as_utc
    return conv


def _param(value: Any) -> Any:
    if isinstance(value, dt.datetime) and value.tzinfo is not None:
        return value.astimezone(dt.UTC).replace(tzinfo=None)
    return value


def _params(params: Sequence[Any] | None) -> tuple[Any, ...] | None:
    if params is None:
        return None
    return tuple(_param(v) for v in params)


def is_duplicate_key(exc: BaseException) -> bool:
    try:
        import pymysql
    except ImportError:  # pragma: no cover - only reached with the extra installed
        return False
    return isinstance(exc, pymysql.err.IntegrityError) and _errno(exc) == DUPLICATE_KEY


def is_lost_connection(exc: BaseException) -> bool:
    """True for a dropped connection. Deadlocks and lock waits are not retried."""
    try:
        import pymysql
    except ImportError:  # pragma: no cover
        return False
    if isinstance(exc, pymysql.err.InterfaceError):
        return True
    return isinstance(exc, pymysql.err.OperationalError) and _errno(exc) in LOST_CONNECTION


def _errno(exc: BaseException) -> int:
    args = getattr(exc, "args", ())
    return args[0] if args and isinstance(args[0], int) else 0


# ---------------------------------------------------------------------------
# Connection and cursor


class MySQLConnection:
    """The subset of the psycopg connection API the stores use.

    PyMySQL connections are not thread-safe. Every statement runs under one
    lock and buffers its rows, so threads may share the connection the way
    they share a psycopg connection.
    """

    def __init__(self, raw: Any) -> None:
        self.raw = raw
        self.lock = threading.RLock()
        self._xact_locks: list[str] = []
        self._in_transaction = False

    @property
    def closed(self) -> bool:
        return not self.raw.open

    @property
    def in_transaction(self) -> bool:
        return self._in_transaction

    def cursor(self) -> MySQLCursor:
        return MySQLCursor(self)

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> MySQLCursor:
        cur = self.cursor()
        cur.execute(sql, params)
        return cur

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """One transaction; locks from ``pg_advisory_xact_lock`` end with it.

        MySQL commits implicitly on DDL. A schema statement inside the block
        therefore commits the statements before it; the migration runner
        depends on per-statement guards, not on rollback, for DDL.
        """
        with self.lock:
            if self._in_transaction:
                yield
                return
            self.raw.begin()
            self._in_transaction = True
            try:
                yield
            except BaseException:
                with suppress(Exception):
                    self.raw.rollback()
                raise
            else:
                self.raw.commit()
            finally:
                self._in_transaction = False
                self.release_transaction_locks()

    def hold_transaction_lock(self, name: str) -> None:
        self._xact_locks.append(name)

    def release_transaction_locks(self) -> None:
        names, self._xact_locks = self._xact_locks, []
        if not names or not self.raw.open:
            return
        with suppress(Exception), self.raw.cursor() as cur:
            for name in reversed(names):
                cur.execute("SELECT RELEASE_LOCK(%s)", (name,))

    def commit(self) -> None:
        with self.lock:
            self.raw.commit()

    def rollback(self) -> None:
        with self.lock:
            self.raw.rollback()

    def close(self) -> None:
        with suppress(Exception):
            self.raw.close()

    def __enter__(self) -> MySQLConnection:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class MySQLCursor:
    """Translates and runs one statement at a time; rows are buffered."""

    def __init__(self, conn: MySQLConnection) -> None:
        self.connection = conn
        self._rows: list[tuple[Any, ...]] = []
        self._pos = 0
        self.description: tuple[Any, ...] | None = None
        self.rowcount = -1
        self.lastrowid: int | None = None

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> MySQLCursor:
        steps = translate(sql)
        bound = _params(params)
        conn = self.connection
        with conn.lock, conn.raw.cursor() as cur:
            self._reset()
            for step in steps:
                self._run(cur, step, bound)
        return self

    def executemany(self, sql: str, seq: Sequence[Sequence[Any]]) -> None:
        total = 0
        for params in seq:
            self.execute(sql, params)
            total += max(self.rowcount, 0)
        self.rowcount = total

    def _reset(self) -> None:
        self._rows, self._pos = [], 0
        self.description = None
        self.rowcount = -1
        self.lastrowid = None

    def _run(self, cur: Any, step: Step, params: tuple[Any, ...] | None) -> None:
        if step.guard is not None:
            cur.execute(step.guard, step.guard_args)
            if cur.fetchone() is not None:
                self._reset()
                self.rowcount = 0
                return
        if step.action == "index":
            sql = _index_sql(cur, step.index)
        elif step.action in ("xact_lock", "try_lock"):
            self._lock(cur, step, params)
            return
        else:
            sql = step.sql
        args = params if step.uses_params else None
        try:
            cur.execute(sql, args)
        except Exception as exc:
            if step.ignore_duplicate and is_duplicate_key(exc):
                self._reset()
                self.rowcount = 0
                return
            raise
        self.rowcount = cur.rowcount
        self.lastrowid = cur.lastrowid
        if step.returning is not None:
            self._returning(step.returning, params)
        elif cur.description:
            self.description = cur.description
            self._rows = [tuple(r) for r in cur.fetchall()]
        else:
            self.description = None
            self._rows = []

    def _returning(self, spec: Returning, params: tuple[Any, ...] | None) -> None:
        self.description = ((spec.column, None, None, None, None, None, None),)
        # 1 = inserted; an upsert that updated reports 2 (or 1 when unchanged).
        if self.rowcount < 1:
            self._rows = []
            return
        if spec.param_index is not None:
            value = (params or ())[spec.param_index]
        elif spec.has_literal:
            value = spec.literal
        else:
            value = self.lastrowid
        self._rows = [(value,)]

    def _lock(self, cur: Any, step: Step, params: tuple[Any, ...] | None) -> None:
        cur.execute(step.sql, params if step.uses_params else None)
        (name,) = cur.fetchone()
        if step.action == "try_lock":
            cur.execute("SELECT GET_LOCK(%s, 0)", (name,))
            (got,) = cur.fetchone()
            self.description = (("pg_try_advisory_lock", None, None, None, None, None, None),)
            self._rows = [(got == 1,)]
            self.rowcount = 1
            return
        cur.execute("SELECT GET_LOCK(%s, -1)", (name,))
        (got,) = cur.fetchone()
        if got != 1:
            raise RuntimeError(f"could not take MySQL lock {name}")
        if self.connection.in_transaction:
            self.connection.hold_transaction_lock(name)
        else:
            # Outside a transaction Postgres releases the lock at statement end.
            cur.execute("SELECT RELEASE_LOCK(%s)", (name,))
        self.description = (("pg_advisory_xact_lock", None, None, None, None, None, None),)
        self._rows = [(None,)]
        self.rowcount = 1

    def fetchone(self) -> tuple[Any, ...] | None:
        if self._pos >= len(self._rows):
            return None
        row = self._rows[self._pos]
        self._pos += 1
        return row

    def fetchall(self) -> list[tuple[Any, ...]]:
        rows = self._rows[self._pos :]
        self._pos = len(self._rows)
        return rows

    def fetchmany(self, size: int = 1) -> list[tuple[Any, ...]]:
        rows = self._rows[self._pos : self._pos + size]
        self._pos += len(rows)
        return rows

    def __iter__(self) -> Iterator[tuple[Any, ...]]:
        while (row := self.fetchone()) is not None:
            yield row

    def close(self) -> None:
        self._rows = []

    def __enter__(self) -> MySQLCursor:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Translation


@dataclass(frozen=True)
class Returning:
    column: str
    param_index: int | None = None
    has_literal: bool = False
    literal: Any = None


@dataclass(frozen=True)
class IndexSpec:
    name: str
    table: str
    columns: tuple[str, ...]
    unique: bool
    where: str | None


@dataclass(frozen=True)
class Step:
    sql: str = ""
    uses_params: bool = False
    action: str = "sql"  # sql | index | xact_lock | try_lock
    guard: str | None = None
    guard_args: tuple[Any, ...] = ()
    ignore_duplicate: bool = False
    returning: Returning | None = None
    index: IndexSpec | None = field(default=None)


_INDEX_EXISTS = (
    "SELECT 1 FROM information_schema.statistics "
    "WHERE table_schema = DATABASE() AND table_name = %s AND index_name = %s LIMIT 1"
)
_COLUMN_EXISTS = (
    "SELECT 1 FROM information_schema.columns "
    "WHERE table_schema = DATABASE() AND table_name = %s AND column_name = %s LIMIT 1"
)


@functools.lru_cache(maxsize=4096)
def translate(sql: str) -> tuple[Step, ...]:
    """Translate one Postgres-dialect statement (or a ``;``-separated script)."""
    steps: list[Step] = []
    statements = split_statements(sql)
    for statement in statements:
        steps.extend(_translate_one(statement, len(statements) == 1))
    return tuple(steps)


def split_statements(sql: str) -> list[str]:
    """Split on top-level ``;`` and drop ``--`` comments outside string literals."""
    out: list[str] = []
    buf: list[str] = []
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        if ch == "'":
            j = i + 1
            while j < n:
                if sql[j] == "'" and j + 1 < n and sql[j + 1] == "'":
                    j += 2
                    continue
                if sql[j] == "'":
                    break
                j += 1
            buf.append(sql[i : j + 1])
            i = j + 1
            continue
        if sql.startswith("--", i):
            end = sql.find("\n", i)
            i = n if end < 0 else end
            continue
        if sql.startswith("$$", i):
            end = sql.find("$$", i + 2)
            end = n if end < 0 else end + 2
            buf.append(sql[i:end])
            i = end
            continue
        if ch == ";":
            out.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    out.append("".join(buf))
    return [s.strip() for s in out if s.strip()]


def _translate_one(sql: str, single: bool) -> list[Step]:
    head = re.sub(r"\s+", " ", sql[:80]).upper().lstrip()
    if head.startswith("DO "):
        raise UnsupportedSQL("PL/pgSQL DO blocks have no MySQL translation; branch at the caller")
    if head.startswith("CREATE TABLE"):
        return _create_table(sql)
    if re.match(r"CREATE (UNIQUE )?INDEX", head):
        return [_create_index(sql)]
    if head.startswith("ALTER TABLE"):
        return [_add_column(sql)]
    if re.fullmatch(r"SET DEFAULT_TRANSACTION_READ_ONLY\s*(=|TO)\s*ON", head.strip()):
        return [Step(sql="SET SESSION TRANSACTION READ ONLY")]
    lock = re.fullmatch(
        r"SELECT\s+(pg_advisory_xact_lock|pg_try_advisory_lock)\((.*)\)",
        sql.strip(),
        flags=re.S | re.I,
    )
    if lock:
        return [_lock_step(lock.group(1).lower(), lock.group(2))]
    if re.match(r"(SELECT|INSERT|UPDATE|DELETE|WITH)\b", head):
        return [_dml(sql)]
    raise UnsupportedSQL(f"no MySQL translation for: {head[:40]}")


def _lock_step(kind: str, arg: str) -> Step:
    inner = arg.strip()
    hashed = re.fullmatch(r"hashtext\((.*)\)", inner, flags=re.S | re.I)
    if hashed:
        inner = hashed.group(1).strip()
        namespace = "h:"
    else:
        namespace = "k:"
    expr, uses = _placeholders(inner)
    # Lock names are capped at 64 characters; a digest keeps any key in range.
    name = f"SELECT CONCAT('{_LOCK_PREFIX}{namespace}', LEFT(SHA2(CAST({expr} AS CHAR), 256), 56))"
    return Step(
        sql=name,
        uses_params=uses,
        action="xact_lock" if kind == "pg_advisory_xact_lock" else "try_lock",
    )


def _placeholders(text: str) -> tuple[str, bool]:
    return text.replace("?", "%s"), "%s" in text or "?" in text


# --- DML --------------------------------------------------------------------

_LITERAL = re.compile(r"'(?:[^']|'')*'")


def _mask(sql: str) -> tuple[str, list[str]]:
    literals: list[str] = []

    def keep(m: re.Match[str]) -> str:
        literals.append(m.group(0))
        return f"\x00{len(literals) - 1}\x00"

    return _LITERAL.sub(keep, sql), literals


def _unmask(sql: str, literals: list[str]) -> str:
    return re.sub(r"\x00(\d+)\x00", lambda m: literals[int(m.group(1))], sql)


_CONFLICT = re.compile(
    r"\s+ON\s+CONFLICT\s*(?:\((?P<target>[^)]*)\))?\s*DO\s+"
    r"(?:(?P<nothing>NOTHING)|UPDATE\s+SET\s+(?P<set>.*?))"
    r"(?:\s+WHERE\s+(?P<where>.*?))?"
    r"(?:\s+RETURNING\s+(?P<ret>[\w\s,]+?))?\s*$",
    flags=re.S | re.I,
)
_RETURNING = re.compile(r"\s+RETURNING\s+(?P<ret>[\w\s,.]+?)\s*$", flags=re.S | re.I)
_INSERT = re.compile(
    r"^\s*INSERT\s+INTO\s+(?P<table>[\w`\"]+)\s*\((?P<cols>[^)]*)\)\s*VALUES\s*\((?P<vals>.*)\)\s*$",
    flags=re.S | re.I,
)
_LEFTOVER = re.compile(
    r"ON\s+CONFLICT|\bRETURNING\b|\bILIKE\b|::|\bpg_\w+|\$\$|\bhashtext\b|\bEXTRACT\s*\(\s*EPOCH"
    r"|\bto_timestamp\b|\bto_jsonb?\b|\bjsonb_\w+|\bSTRING_AGG\b|\bARRAY_AGG\b|\bDISTINCT\s+ON\b"
    r"|\bNULLS\s+(FIRST|LAST)\b|\bIS\s+(NOT\s+)?DISTINCT\s+FROM\b",
    flags=re.I,
)


def _dml(sql: str) -> Step:
    masked, literals = _mask(sql.strip())
    masked = _quote_reserved(masked)
    masked = re.sub(r"\bEXCLUDED\.", "excluded.", masked, flags=re.I)
    masked = re.sub(
        r"(?<![\w.])([\w.]+)\s+ILIKE\s+(%s|\?|\x00\d+\x00|[\w.]+)",
        r"LOWER(\1) LIKE LOWER(\2)",
        masked,
        flags=re.I,
    )
    masked = re.sub(r"\bNOW\(\)", "NOW(6)", masked, flags=re.I)
    # Epoch seconds to a timestamp; the session runs in UTC.
    masked = re.sub(r"\bto_timestamp\(", "FROM_UNIXTIME(", masked, flags=re.I)
    masked = _epoch(masked)

    ignore_duplicate = False
    returning: Returning | None = None
    conflict = _CONFLICT.search(masked)
    if conflict:
        if conflict.group("where"):
            raise UnsupportedSQL("conditional upsert (DO UPDATE .. WHERE); branch at the caller")
        head = masked[: conflict.start()]
        if not _INSERT.match(head):
            raise UnsupportedSQL("ON CONFLICT needs INSERT .. (columns) VALUES (..)")
        if conflict.group("nothing"):
            masked = head
            ignore_duplicate = True
        else:
            assignments = _split_top(conflict.group("set"))
            _check_assignment_order(assignments)
            masked = f"{head} AS excluded ON DUPLICATE KEY UPDATE {', '.join(assignments)}"
        if conflict.group("ret"):
            returning = _returning_spec(head, conflict.group("ret"), literals)
    else:
        ret = _RETURNING.search(masked)
        if ret:
            head = masked[: ret.start()]
            if not re.match(r"\s*INSERT\b", head, flags=re.I):
                raise UnsupportedSQL("UPDATE/DELETE .. RETURNING; branch at the caller")
            returning = _returning_spec(head, ret.group("ret"), literals)
            masked = head
    leftover = _LEFTOVER.search(masked)
    if leftover:
        raise UnsupportedSQL(f"Postgres-only SQL left after translation: {leftover.group(0)}")
    return Step(
        sql=_unmask(masked, literals),
        uses_params=True,
        ignore_duplicate=ignore_duplicate,
        returning=returning,
    )


def _quote_reserved(sql: str) -> str:
    for word in RESERVED_COLUMNS:
        sql = re.sub(
            rf"(?<![\w`\"])(?<!PRIMARY )(?<!FOREIGN )(?<!UNIQUE )(?<!DUPLICATE )({word})(?![\w`\"(])",
            r"`\1`",
            sql,
        )
    return sql


def _epoch(sql: str) -> str:
    out = []
    pos = 0
    for m in re.finditer(r"\bEXTRACT\s*\(\s*EPOCH\s+FROM\s+", sql, flags=re.I):
        if m.start() < pos:
            continue
        depth, i = 1, m.end()
        while i < len(sql) and depth:
            depth += {"(": 1, ")": -1}.get(sql[i], 0)
            i += 1
        out.append(sql[pos : m.start()])
        out.append(f"UNIX_TIMESTAMP({sql[m.end() : i - 1].strip()})")
        pos = i
    out.append(sql[pos:])
    return "".join(out)


def _split_top(text: str) -> list[str]:
    parts, depth, buf = [], 0, []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    if "".join(buf).strip():
        parts.append("".join(buf).strip())
    return parts


def _check_assignment_order(assignments: list[str]) -> None:
    """MySQL applies SET left to right; Postgres reads the old row throughout.

    The translation is only exact when no assignment reads a column that an
    earlier assignment in the same list already changed.
    """
    assigned: list[str] = []
    for item in assignments:
        target, _, expr = item.partition("=")
        name = target.strip().strip('`"').split(".")[-1].lower()
        refs = re.sub(r"\bexcluded\.[\w`]+", "", expr, flags=re.I)
        for earlier in assigned:
            if re.search(rf"(?<![\w`]){re.escape(earlier)}(?![\w`])", refs.replace("`", "")):
                raise UnsupportedSQL(
                    f"upsert assignment to {name} reads {earlier}, which an earlier assignment "
                    "changed; MySQL would read the new value"
                )
        assigned.append(name)


def _returning_spec(head: str, ret: str, literals: list[str]) -> Returning:
    column = ret.strip()
    if "," in column:
        raise UnsupportedSQL("RETURNING more than one column")
    column = column.split(".")[-1].strip('`"')
    insert = _INSERT.match(head)
    if not insert:
        raise UnsupportedSQL("RETURNING needs INSERT .. (columns) VALUES (..)")
    cols = [c.strip().strip('`"').lower() for c in insert.group("cols").split(",")]
    values = _split_top(insert.group("vals"))
    if column.lower() not in cols:
        return Returning(column=column)  # the AUTO_INCREMENT id
    value = values[cols.index(column.lower())]
    before = "".join(values[: cols.index(column.lower())])
    if value in ("%s", "?"):
        return Returning(column=column, param_index=before.count("%s") + before.count("?"))
    lit = re.fullmatch(r"\x00(\d+)\x00", value)
    if lit:
        return Returning(column=column, has_literal=True, literal=literals[int(lit.group(1))][1:-1])
    if re.fullmatch(r"-?\d+", value):
        return Returning(column=column, has_literal=True, literal=int(value))
    raise UnsupportedSQL(f"RETURNING {column} from an expression")


# --- DDL --------------------------------------------------------------------


@dataclass
class _Column:
    name: str
    type: str
    nullable: bool = True
    default: str | None = None
    auto_increment: bool = False

    @property
    def text(self) -> bool:
        return self.type in _TEXT_TYPES


_TYPE = re.compile(
    r"(DOUBLE PRECISION|BIGSERIAL|SERIAL|TEXT|INTEGER|BIGINT|SMALLINT|REAL|BOOLEAN|"
    r"TIMESTAMPTZ|INT)\b",
    flags=re.I,
)
_DEFAULT = re.compile(
    r"\bDEFAULT\s+('(?:[^']|'')*'|-?\d+(?:\.\d+)?|TRUE|FALSE|NULL|NOW\(\)|CURRENT_TIMESTAMP)",
    flags=re.I,
)
_REFERENCES = re.compile(
    r"\bREFERENCES\s+(\w+)\s*\(([^)]*)\)(\s+ON\s+DELETE\s+(CASCADE|RESTRICT|SET NULL|NO ACTION))?",
    flags=re.I,
)


def _q(name: str) -> str:
    return f"`{name}`"


def _parse_column(text: str) -> tuple[_Column, list[tuple[str, Any]]]:
    """One column definition; returns it and any inline key constraints."""
    m = re.match(r"\s*\"?(\w+)\"?\s+(.*)$", text, flags=re.S)
    if not m:
        raise UnsupportedSQL(f"unparsed column definition: {text.strip()[:60]}")
    name, rest = m.group(1), m.group(2)
    tm = _TYPE.match(rest)
    if not tm:
        raise UnsupportedSQL(f"unsupported column type in: {text.strip()[:60]}")
    pg_type = tm.group(1).upper()
    rest = rest[tm.end() :]
    constraints: list[tuple[str, Any]] = []
    col = _Column(name=name, type=_mysql_type(pg_type))
    if pg_type in ("SERIAL", "BIGSERIAL"):
        col.auto_increment = True
        col.nullable = False
    if re.search(r"\bNOT\s+NULL\b", rest, flags=re.I):
        col.nullable = False
    dm = _DEFAULT.search(rest)
    if dm:
        col.default = dm.group(1)
    if re.search(r"\bPRIMARY\s+KEY\b", rest, flags=re.I):
        constraints.append(("primary", (name,)))
        col.nullable = False
    if re.search(r"\bUNIQUE\b", rest, flags=re.I):
        constraints.append(("unique", (name,)))
    ref = _REFERENCES.search(rest)
    if ref:
        constraints.append(("foreign", ((name,), ref.group(1), ref.group(4))))
    leftover = _REFERENCES.sub("", _DEFAULT.sub("", rest))
    leftover = re.sub(r"\b(NOT\s+NULL|NULL|PRIMARY\s+KEY|UNIQUE)\b", "", leftover, flags=re.I)
    if leftover.strip():
        raise UnsupportedSQL(f"unsupported column clause {leftover.strip()!r} on {name}")
    return col, constraints


def _mysql_type(pg_type: str) -> str:
    return {
        "TEXT": "LONGTEXT",
        "SERIAL": "INTEGER",
        "BIGSERIAL": "BIGINT",
        "TIMESTAMPTZ": "DATETIME(6)",
        "BOOLEAN": "TINYINT(1)",
        "INT": "INTEGER",
        # Postgres REAL is float4; DOUBLE keeps at least that precision.
        "REAL": "DOUBLE",
    }.get(pg_type, pg_type)


def _column_sql(col: _Column) -> str:
    parts = [_q(col.name), col.type]
    if not col.nullable:
        parts.append("NOT NULL")
    if col.default is not None:
        parts.append(_default_sql(col))
    if col.auto_increment:
        parts.append("AUTO_INCREMENT")
    return " ".join(parts)


def _default_sql(col: _Column) -> str:
    value = col.default or ""
    upper = value.upper()
    if upper in ("NOW()", "CURRENT_TIMESTAMP"):
        return "DEFAULT CURRENT_TIMESTAMP(6)"
    if upper == "NULL":
        return "DEFAULT NULL"
    if col.text:
        # TEXT and BLOB columns take only expression defaults (8.0.13+).
        return f"DEFAULT ({value})"
    return f"DEFAULT {value}"


def _hash_expr(
    columns: Sequence[str], nullable: Sequence[str] = (), where: str | None = None
) -> str:
    digest = f"UNHEX(SHA2(JSON_ARRAY({', '.join(_q(c) for c in columns)}), 256))"
    conditions = [f"{_q(c)} IS NOT NULL" for c in nullable]
    if where:
        conditions.insert(0, f"({where})")
    if conditions:
        return f"(CASE WHEN {' AND '.join(conditions)} THEN {digest} END)"
    return f"({digest})"


def _name(base: str, *parts: str) -> str:
    name = "_".join((base, *parts))
    if len(name) <= 64:
        return name
    return name[:55] + "_" + hashlib.sha256(name.encode()).hexdigest()[:8]


def _prefix_columns(columns: Sequence[str], text: Sequence[str]) -> str:
    n_text = sum(1 for c in columns if c in text)
    other_bytes = 8 * (len(columns) - n_text)
    prefix = MAX_PREFIX_CHARS
    if n_text:
        prefix = max(1, min(MAX_PREFIX_CHARS, (INDEX_KEY_BYTES - other_bytes) // (4 * n_text)))
    return ", ".join(f"{_q(c)}({prefix})" if c in text else _q(c) for c in columns)


def _create_table(sql: str) -> list[Step]:
    m = re.match(
        r"\s*CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+\"?(\w+)\"?\s*\((.*)\)\s*$",
        sql,
        flags=re.S | re.I,
    )
    if not m:
        raise UnsupportedSQL("only CREATE TABLE IF NOT EXISTS name (..) is translated")
    table, body = m.group(1), m.group(2)
    columns: dict[str, _Column] = {}
    constraints: list[tuple[str, Any]] = []
    for item in _split_top(body):
        upper = item.upper()
        if upper.startswith("PRIMARY KEY"):
            constraints.append(("primary", _column_list(item)))
        elif upper.startswith("UNIQUE"):
            constraints.append(("unique", _column_list(item)))
        elif upper.startswith("FOREIGN KEY"):
            ref = _REFERENCES.search(item)
            if not ref:
                raise UnsupportedSQL(f"unparsed foreign key: {item[:60]}")
            constraints.append(("foreign", (_column_list(item), ref.group(1), ref.group(4))))
        elif re.match(r"(CONSTRAINT\s|CHECK\s*\(|EXCLUDE\s)", upper):
            raise UnsupportedSQL(f"unsupported table constraint: {item[:60]}")
        else:
            col, inline = _parse_column(item)
            columns[col.name] = col
            constraints.extend(inline)
    for kind, spec in constraints:
        if kind == "primary":
            for name in spec:
                columns[name].nullable = False

    defs = [_column_sql(c) for c in columns.values()]
    keys: list[str] = []
    text = [c.name for c in columns.values() if c.text]
    # InnoDB refuses a cascading foreign key on a base column of a STORED
    # generated column. Such a key is hashed into a VIRTUAL column instead,
    # which can carry a unique index but not the primary key.
    cascading = {
        c
        for kind, spec in constraints
        if kind == "foreign" and (spec[2] or "").upper() in ("CASCADE", "SET NULL")
        for c in spec[0]
    }
    unique_n = 0
    fk_n = 0
    for kind, spec in constraints:
        if kind in ("primary", "unique"):
            cols = tuple(spec)
            missing = [c for c in cols if c not in columns]
            if missing:
                raise UnsupportedSQL(f"key on unknown column {missing[0]} in {table}")
            hashed = any(columns[c].text for c in cols)
            if kind == "primary" and not hashed:
                keys.append(f"PRIMARY KEY ({', '.join(_q(c) for c in cols)})")
                continue
            if kind == "unique":
                unique_n += 1
                key_name = _name(table, "uk", str(unique_n))
            else:
                key_name = "PRIMARY"
            if not hashed:
                keys.append(f"UNIQUE KEY {_q(key_name)} ({', '.join(_q(c) for c in cols)})")
                continue
            gen = "mira_pk" if kind == "primary" else _name("mira_uk", str(unique_n))
            nullable = [c for c in cols if columns[c].nullable]
            null_sql = "" if nullable else " NOT NULL"
            storage = "VIRTUAL" if cascading & set(cols) else "STORED"
            defs.append(
                f"{_q(gen)} BINARY(32) AS {_hash_expr(cols, nullable)} {storage}{null_sql} INVISIBLE"
            )
            if kind == "primary" and storage == "STORED":
                keys.append(f"PRIMARY KEY ({_q(gen)})")
            elif kind == "primary":
                defs.append("`mira_row_id` BIGINT NOT NULL AUTO_INCREMENT INVISIBLE")
                keys.append("PRIMARY KEY (`mira_row_id`)")
                keys.append(f"UNIQUE KEY {_q(_name(table, 'pk'))} ({_q(gen)})")
            else:
                keys.append(f"UNIQUE KEY {_q(key_name)} ({_q(gen)})")
            lookup = _name(table, "pk" if kind == "primary" else f"uk_{unique_n}", "lookup")
            keys.append(f"KEY {_q(lookup)} ({_prefix_columns(cols, text)})")
        else:
            cols, parent, on_delete = spec
            fk_n += 1
            action = f" ON DELETE {on_delete.upper()}" if on_delete else ""
            if any(columns[c].text for c in cols):
                # Text keys are hashed on both sides; reference the parent's hash.
                gen = _name("mira_fk", str(fk_n))
                nullable = [c for c in cols if columns[c].nullable]
                defs.append(
                    f"{_q(gen)} BINARY(32) AS {_hash_expr(cols, nullable)} STORED INVISIBLE"
                )
                keys.append(
                    f"CONSTRAINT {_q(_name(table, 'fk', str(fk_n)))} FOREIGN KEY ({_q(gen)}) "
                    f"REFERENCES {_q(parent)} (`mira_pk`){action}"
                )
            else:
                keys.append(
                    f"CONSTRAINT {_q(_name(table, 'fk', str(fk_n)))} FOREIGN KEY "
                    f"({', '.join(_q(c) for c in cols)}) REFERENCES {_q(parent)} "
                    f"({_parent_columns(sql, cols)}){action}"
                )
    return [
        Step(
            sql=f"CREATE TABLE IF NOT EXISTS {_q(table)} (\n  "
            + ",\n  ".join(defs + keys)
            + f"\n) {TABLE_OPTIONS}"
        )
    ]


def _parent_columns(sql: str, cols: Sequence[str]) -> str:
    """Referenced columns of the foreign key on ``cols`` in the original statement."""
    for item in _split_top(re.match(r"[^(]*\((.*)\)\s*$", sql, flags=re.S).group(1)):  # type: ignore[union-attr]
        ref = _REFERENCES.search(item)
        if not ref:
            continue
        upper = item.upper()
        if upper.startswith("FOREIGN KEY"):
            own = _column_list(item)
        else:
            own = (re.match(r"\s*\"?(\w+)", item).group(1),)  # type: ignore[union-attr]
        if tuple(own) == tuple(cols):
            return ", ".join(_q(c.strip().strip('"')) for c in ref.group(2).split(","))
    raise UnsupportedSQL("foreign key target not found")


def _column_list(item: str) -> tuple[str, ...]:
    m = re.search(r"\(([^)]*)\)", item)
    if not m:
        raise UnsupportedSQL(f"no column list in: {item[:60]}")
    return tuple(c.strip().strip('"') for c in m.group(1).split(",") if c.strip())


def _create_index(sql: str) -> Step:
    m = re.match(
        r"\s*CREATE\s+(UNIQUE\s+)?INDEX\s+IF\s+NOT\s+EXISTS\s+\"?(\w+)\"?\s+ON\s+\"?(\w+)\"?\s*"
        r"\(([^)]*)\)\s*(?:WHERE\s+(.*))?$",
        sql,
        flags=re.S | re.I,
    )
    if not m:
        raise UnsupportedSQL(
            "only CREATE [UNIQUE] INDEX IF NOT EXISTS name ON table (cols) is translated"
        )
    unique, name, table, cols_sql, where = m.groups()
    columns = tuple(c.strip().strip('"') for c in cols_sql.split(","))
    for c in columns:
        if not re.fullmatch(r"\w+", c):
            raise UnsupportedSQL(f"index expression {c!r} is not a plain column")
    spec = IndexSpec(
        name=name,
        table=table,
        columns=columns,
        unique=bool(unique),
        where=(where or "").strip() or None,
    )
    return Step(action="index", index=spec, guard=_INDEX_EXISTS, guard_args=(table, name))


def _index_sql(cur: Any, spec: IndexSpec | None) -> str:
    """Build the index DDL once the column types are known."""
    assert spec is not None
    cur.execute(
        "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
        "WHERE table_schema = DATABASE() AND table_name = %s",
        (spec.table,),
    )
    info = {r[0].lower(): (r[1].lower(), r[2] == "YES") for r in cur.fetchall()}
    missing = [c for c in spec.columns if c.lower() not in info]
    if missing:
        raise UnsupportedSQL(f"index {spec.name} names unknown column {missing[0]}")
    text = [c for c in spec.columns if info[c.lower()][0] in ("text", "longtext", "mediumtext")]
    table = _q(spec.table)
    if not spec.unique:
        # A partial index only narrows which rows Postgres indexes; a full
        # index answers the same queries.
        return f"CREATE INDEX {_q(spec.name)} ON {table} ({_prefix_columns(spec.columns, text)})"
    if not text and not spec.where:
        return f"CREATE UNIQUE INDEX {_q(spec.name)} ON {table} ({', '.join(_q(c) for c in spec.columns)})"
    gen = _name("mira_ux", spec.name)
    nullable = [c for c in spec.columns if info[c.lower()][1]]
    where = _translate_where(spec.where) if spec.where else None
    return (
        f"ALTER TABLE {table} "
        f"ADD COLUMN {_q(gen)} BINARY(32) AS {_hash_expr(spec.columns, nullable, where)} STORED INVISIBLE, "
        f"ADD UNIQUE INDEX {_q(spec.name)} ({_q(gen)}), "
        f"ADD INDEX {_q(_name(spec.name, 'lookup'))} ({_prefix_columns(spec.columns, text)})"
    )


def _translate_where(where: str) -> str:
    masked, literals = _mask(where)
    if _LEFTOVER.search(masked) or ";" in masked:
        raise UnsupportedSQL(f"unsupported index predicate {where!r}")
    masked = re.sub(r"\b([a-z_]\w*)\b(?=\s*(<>|=|!=|<|>|IS\b))", r"`\1`", masked)
    return _unmask(masked, literals)


def _add_column(sql: str) -> Step:
    m = re.match(
        r"\s*ALTER\s+TABLE\s+\"?(\w+)\"?\s+ADD\s+COLUMN\s+(IF\s+NOT\s+EXISTS\s+)?(.*)$",
        sql,
        flags=re.S | re.I,
    )
    if not m:
        raise UnsupportedSQL("only ALTER TABLE .. ADD COLUMN is translated")
    table, _, definition = m.groups()
    col, constraints = _parse_column(definition)
    if constraints:
        raise UnsupportedSQL("ADD COLUMN with an inline key is not translated")
    return Step(
        sql=f"ALTER TABLE {_q(table)} ADD COLUMN {_column_sql(col)}",
        guard=_COLUMN_EXISTS,
        guard_args=(table, col.name),
    )


def errors() -> tuple[type[BaseException], ...]:
    """PyMySQL's base error, for ``except`` clauses that list database errors."""
    try:
        import pymysql
    except ImportError:  # pragma: no cover
        return ()
    return (pymysql.err.Error,)


__all__ = [
    "MIN_VERSION",
    "MySQLConnection",
    "MySQLCursor",
    "UnsupportedSQL",
    "check_server",
    "connect",
    "connect_kwargs",
    "errors",
    "is_duplicate_key",
    "is_lost_connection",
    "split_statements",
    "translate",
]

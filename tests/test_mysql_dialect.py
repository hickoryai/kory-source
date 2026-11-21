"""MySQL translation of the stores' Postgres-dialect SQL. No server needed.

End-to-end behavior on a real server is covered by the backend-parametrized
store tests (``REGISTRY_TEST_MYSQL_URL``).
"""

from __future__ import annotations

import pytest

from mira.db import mysql
from mira.db.mysql import UnsupportedSQL, split_statements, translate
from mira.db.url import backend, is_server


def one(sql: str) -> mysql.Step:
    steps = translate(sql)
    assert len(steps) == 1
    return steps[0]


# ── URLs ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("postgresql://u:p@h/db", "postgres"),
        ("postgres://u:p@h/db", "postgres"),
        ("mysql://u:p@h/db", "mysql"),
        ("mysql+pymysql://u:p@h/db", "mysql"),
        ("", "sqlite"),
        ("/data/app.db", "sqlite"),
        ("sqlite:///x.db", "sqlite"),
        (None, "sqlite"),
    ],
)
def test_backend_from_url(url, expected):
    assert backend(url) == expected
    assert is_server(url) is (expected != "sqlite")


def test_connect_kwargs_parse_credentials_and_database():
    pytest.importorskip("pymysql")
    kwargs = mysql.connect_kwargs(
        "mysql+pymysql://app%40svc:p%3Ass@db.example.test:3307/kory_source?connect_timeout=5"
    )
    assert kwargs["host"] == "db.example.test"
    assert kwargs["port"] == 3307
    assert kwargs["user"] == "app@svc"
    assert kwargs["password"] == "p:ss"
    assert kwargs["database"] == "kory_source"
    assert kwargs["connect_timeout"] == 5
    assert kwargs["autocommit"] is True
    assert kwargs["collation"] == "utf8mb4_0900_bin"
    assert "ssl" not in kwargs  # no TLS parameters: the driver default applies
    with pytest.raises(ValueError, match="name a database"):
        mysql.connect_kwargs("mysql://root@localhost/")


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ({}, None),
        ({"ssl_mode": "REQUIRED"}, mysql.TLS()),
        ({"ssl_mode": "required", "ssl_ca": "/ca"}, mysql.TLS(ca="/ca", verify_cert=True)),
        ({"ssl_ca": "/ca"}, mysql.TLS(ca="/ca", verify_cert=True)),
        ({"ssl_mode": "VERIFY_CA"}, mysql.TLS(verify_cert=True)),
        (
            {"ssl_mode": "VERIFY_IDENTITY", "ssl_ca": "/ca", "ssl_cert": "/c", "ssl_key": "/k"},
            mysql.TLS(ca="/ca", cert="/c", key="/k", verify_cert=True, verify_identity=True),
        ),
        ({"ssl_cert": "/c", "ssl_key": "/k"}, mysql.TLS(cert="/c", key="/k")),
        ({"ssl_mode": "DISABLED"}, mysql.DISABLED),
    ],
)
def test_tls_settings(query, expected):
    assert mysql.tls_settings(query) == expected


@pytest.mark.parametrize(
    ("query", "message"),
    [
        # MySQL clients fall back to plaintext under PREFERRED; refused here.
        ({"ssl_mode": "PREFERRED"}, "PREFERRED is not supported"),
        ({"ssl_mode": "DISABLED", "ssl_ca": "/ca"}, "conflicts"),
        ({"ssl_mode": "SOMETIMES"}, "unknown ssl_mode"),
        ({"ssl_key": "/k"}, "ssl_key needs ssl_cert"),
    ],
)
def test_tls_settings_errors(query, message):
    with pytest.raises(ValueError, match=message):
        mysql.tls_settings(query)


def _certificates(tmp_path):
    """A self-signed CA and a client certificate it signed, as PEM files."""
    import datetime as dt

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    def cert(subject, issuer, key, signer, *, ca):
        now = dt.datetime.now(dt.UTC)
        return (
            x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, subject)]))
            .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, issuer)]))
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(days=1))
            .not_valid_after(now + dt.timedelta(days=1))
            .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True)
            .sign(signer, hashes.SHA256())
        )

    ca_key = ec.generate_private_key(ec.SECP256R1())
    client_key = ec.generate_private_key(ec.SECP256R1())
    files = {
        "ca": cert("test-ca", "test-ca", ca_key, ca_key, ca=True),
        "cert": cert("test-client", "test-ca", client_key, ca_key, ca=False),
    }
    paths = {}
    for name, c in files.items():
        paths[name] = tmp_path / f"{name}.pem"
        paths[name].write_bytes(c.public_bytes(serialization.Encoding.PEM))
    paths["key"] = tmp_path / "key.pem"
    paths["key"].write_bytes(
        client_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return paths


def _driver(url):
    """The PyMySQL connection these settings produce, without connecting."""
    pymysql = pytest.importorskip("pymysql")
    return pymysql.connections.Connection(defer_connect=True, **mysql.connect_kwargs(url))


def test_tls_modes_reach_the_driver(tmp_path):
    import ssl

    pytest.importorskip("cryptography")
    files = _certificates(tmp_path)
    base = "mysql://u:p@db.example.test/kory"

    required = _driver(base + "?ssl_mode=REQUIRED")
    assert required.ssl is True and required._ssl_required is True
    assert required.ctx.verify_mode == ssl.CERT_NONE

    full = _driver(
        f"{base}?ssl_mode=VERIFY_IDENTITY&ssl_ca={files['ca']}"
        f"&ssl_cert={files['cert']}&ssl_key={files['key']}"
    )
    assert full.ssl is True and full._ssl_required is True
    assert full.ctx.verify_mode == ssl.CERT_REQUIRED
    assert full.ctx.check_hostname is True
    # Only the given CA is trusted: the system store was not loaded.
    assert [c["subject"] for c in full.ctx.get_ca_certs()] == [((("commonName", "test-ca"),),)]

    verify_ca = _driver(f"{base}?ssl_ca={files['ca']}")
    assert verify_ca._ssl_required is True
    assert verify_ca.ctx.verify_mode == ssl.CERT_REQUIRED
    assert verify_ca.ctx.check_hostname is False

    disabled = _driver(base + "?ssl_mode=DISABLED")
    assert disabled.ssl is False


def test_required_tls_encrypts_a_live_connection():
    """``ssl_mode=REQUIRED`` negotiates TLS with a real server."""
    from tests.server_db import server_url

    url = server_url("mysql")
    if not url:
        pytest.skip("REGISTRY_TEST_MYSQL_URL not set")
    sep = "&" if "?" in url else "?"
    conn = mysql.connect(url + sep + "ssl_mode=REQUIRED")
    try:
        with conn.raw.cursor() as cur:
            cur.execute("SHOW SESSION STATUS LIKE 'Ssl_cipher'")
            (_, cipher) = cur.fetchone()
        assert cipher
    finally:
        conn.close()


@pytest.mark.parametrize("version", ["8.0.23", "8.0.44", "8.4.10", "9.1.0"])
def test_supported_servers(version):
    mysql.check_server(version)


@pytest.mark.parametrize(
    ("version", "message"),
    [
        ("8.0.22", "8.0.23 or newer"),
        ("5.7.44-log", "8.0.23 or newer"),
        ("10.11.6-MariaDB", "MariaDB"),
    ],
)
def test_unsupported_servers(version, message):
    with pytest.raises(RuntimeError, match=message):
        mysql.check_server(version)


# ── DML ───────────────────────────────────────────────────────────────────


def test_upsert_uses_row_alias():
    step = one(
        "INSERT INTO t (a, b) VALUES (%s, %s) ON CONFLICT (a) DO UPDATE SET "
        "b = EXCLUDED.b, n = t.n + 1"
    )
    assert step.sql == (
        "INSERT INTO t (a, b) VALUES (%s, %s) AS excluded ON DUPLICATE KEY UPDATE "
        "b = excluded.b, n = t.n + 1"
    )
    assert not step.ignore_duplicate


def test_do_nothing_is_a_plain_insert_that_ignores_duplicates():
    step = one("INSERT INTO t (a) VALUES (%s) ON CONFLICT DO NOTHING")
    assert step.sql == "INSERT INTO t (a) VALUES (%s)"
    assert step.ignore_duplicate


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        (
            "INSERT INTO t (x, id) VALUES (%s, %s) ON CONFLICT DO NOTHING RETURNING id",
            mysql.Returning("id", param_index=1),
        ),
        (
            "INSERT INTO t (id, x) VALUES ('fixed', %s) ON CONFLICT DO NOTHING RETURNING id",
            mysql.Returning("id", has_literal=True, literal="fixed"),
        ),
        ("INSERT INTO t (x) VALUES (%s) RETURNING id", mysql.Returning("id")),
    ],
)
def test_insert_returning(sql, expected):
    step = one(sql)
    assert "RETURNING" not in step.sql
    assert step.returning == expected


@pytest.mark.parametrize(
    "sql",
    [
        # Conditional upsert: callers branch to a locked read-then-write.
        "INSERT INTO t (a, b) VALUES (?, ?) ON CONFLICT (a) DO UPDATE SET b = excluded.b "
        "WHERE t.b <= ? RETURNING b",
        "UPDATE t SET a = ? WHERE id = ? RETURNING id",
        "DELETE FROM t WHERE id = ? RETURNING id",
        "SELECT x FROM t ORDER BY x NULLS LAST",
        "SELECT a IS DISTINCT FROM b FROM t",
        "SELECT x::text FROM t",
        "SELECT to_jsonb(t) FROM t",
        "DO $$ BEGIN PERFORM 1; END $$",
        # MySQL would apply ``b`` after ``a`` changed; Postgres reads the old row.
        "INSERT INTO t (k, a) VALUES (?, ?) ON CONFLICT (k) DO UPDATE SET a = excluded.a, b = a + 1",
    ],
)
def test_postgres_only_sql_fails_closed(sql):
    with pytest.raises(UnsupportedSQL):
        translate(sql)


def test_rewrites():
    assert one("SELECT name FROM p WHERE name ILIKE %s").sql == (
        "SELECT name FROM p WHERE LOWER(name) LIKE LOWER(%s)"
    )
    assert one("SELECT EXTRACT(EPOCH FROM updated_at) FROM r").sql == (
        "SELECT UNIX_TIMESTAMP(updated_at) FROM r"
    )
    assert one("INSERT INTO s (t, e) VALUES (%s, to_timestamp(%s))").sql == (
        "INSERT INTO s (t, e) VALUES (%s, FROM_UNIXTIME(%s))"
    )
    assert one("UPDATE r SET updated_at = NOW() WHERE id = %s").sql == (
        "UPDATE r SET updated_at = NOW(6) WHERE id = %s"
    )


def test_reserved_columns_are_quoted_but_keywords_and_literals_are_not():
    step = one(
        "INSERT INTO settings (key, value) VALUES (%s, 'key signal') "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value"
    )
    assert step.sql == (
        "INSERT INTO settings (`key`, value) VALUES (%s, 'key signal') "
        "AS excluded ON DUPLICATE KEY UPDATE value = excluded.value"
    )
    assert one("SELECT signal, COUNT(*) FROM f GROUP BY signal").sql == (
        "SELECT `signal`, COUNT(*) FROM f GROUP BY `signal`"
    )


def test_literals_are_not_translated():
    step = one("SELECT 'ON CONFLICT DO NOTHING; RETURNING x::text ILIKE' FROM t")
    assert step.sql == "SELECT 'ON CONFLICT DO NOTHING; RETURNING x::text ILIKE' FROM t"


def test_advisory_locks():
    xact = one("SELECT pg_advisory_xact_lock(%s)")
    assert xact.action == "xact_lock"
    assert "GET_LOCK" not in xact.sql  # the name is computed first
    assert "SHA2(CAST(%s AS CHAR), 256)" in xact.sql
    hashed = one("SELECT pg_advisory_xact_lock(hashtext(%s))")
    assert "'mira:h:'" in hashed.sql
    assert one("SELECT pg_try_advisory_lock(%s)").action == "try_lock"


def test_read_only_session():
    assert one("SET default_transaction_read_only = on").sql == "SET SESSION TRANSACTION READ ONLY"


# ── DDL ───────────────────────────────────────────────────────────────────


def test_split_statements_respects_literals_comments_and_dollar_quotes():
    script = """
    -- a comment; with a semicolon
    CREATE TABLE IF NOT EXISTS a (x TEXT NOT NULL DEFAULT 'a;b');
    DO $$ BEGIN PERFORM 1; END $$;
    CREATE INDEX IF NOT EXISTS i ON a (x)
    """
    parts = split_statements(script)
    assert len(parts) == 3
    assert parts[0].endswith("DEFAULT 'a;b')")
    assert parts[1] == "DO $$ BEGIN PERFORM 1; END $$"


def test_text_keys_are_hash_backed_with_a_lookup_index():
    (step,) = translate(
        "CREATE TABLE IF NOT EXISTS anchors (object_id TEXT NOT NULL, path TEXT NOT NULL, "
        "line BIGINT NOT NULL DEFAULT 0, note TEXT, created_at DOUBLE PRECISION NOT NULL DEFAULT 0, "
        "PRIMARY KEY (object_id, path), UNIQUE (path, line, note))"
    )
    sql = step.sql
    assert "`object_id` LONGTEXT NOT NULL" in sql
    assert "`created_at` DOUBLE PRECISION NOT NULL DEFAULT 0" in sql
    assert (
        "`mira_pk` BINARY(32) AS (UNHEX(SHA2(JSON_ARRAY(`object_id`, `path`), 256))) "
        "STORED NOT NULL INVISIBLE"
    ) in sql
    assert "PRIMARY KEY (`mira_pk`)" in sql
    assert "KEY `anchors_pk_lookup` (`object_id`(255), `path`(255))" in sql
    # NULLs never collide in a SQL unique key; the hash is NULL when a column is.
    assert "CASE WHEN `note` IS NOT NULL THEN UNHEX(SHA2(JSON_ARRAY(`path`, `line`, `note`)" in sql
    assert sql.endswith("ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin")


def test_column_types_and_defaults():
    (step,) = translate(
        "CREATE TABLE IF NOT EXISTS users (id SERIAL PRIMARY KEY, name TEXT NOT NULL DEFAULT '', "
        "is_admin BOOLEAN NOT NULL DEFAULT FALSE, seen REAL NOT NULL DEFAULT 0, "
        "created_at TIMESTAMPTZ DEFAULT NOW(), checkpoint TEXT NOT NULL DEFAULT 'x')"
    )
    sql = step.sql
    assert "`id` INTEGER NOT NULL AUTO_INCREMENT" in sql
    assert "PRIMARY KEY (`id`)" in sql
    assert "`name` LONGTEXT NOT NULL DEFAULT ('')" in sql
    assert "`is_admin` TINYINT(1) NOT NULL DEFAULT FALSE" in sql
    assert "`seen` DOUBLE NOT NULL DEFAULT 0" in sql
    assert "`created_at` DATETIME(6) DEFAULT CURRENT_TIMESTAMP(6)" in sql
    # A column whose name starts with CHECK is a column, not a constraint.
    assert "`checkpoint` LONGTEXT NOT NULL DEFAULT ('x')" in sql


def test_foreign_keys():
    (integer,) = translate(
        "CREATE TABLE IF NOT EXISTS sessions (token TEXT PRIMARY KEY, "
        "user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE)"
    )
    assert "FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE" in integer.sql
    (text,) = translate(
        "CREATE TABLE IF NOT EXISTS work (id TEXT PRIMARY KEY, "
        "run_id TEXT NOT NULL REFERENCES runs(run_id))"
    )
    assert "`mira_fk_1` BINARY(32) AS (UNHEX(SHA2(JSON_ARRAY(`run_id`), 256))) STORED" in text.sql
    assert "FOREIGN KEY (`mira_fk_1`) REFERENCES `runs` (`mira_pk`)" in text.sql


def test_cascading_foreign_key_inside_a_text_key_uses_a_virtual_hash():
    (step,) = translate(
        "CREATE TABLE IF NOT EXISTS days (contributor_id INTEGER NOT NULL "
        "REFERENCES contributors(id) ON DELETE CASCADE, day TEXT NOT NULL, "
        "PRIMARY KEY (contributor_id, day))"
    )
    assert "VIRTUAL NOT NULL INVISIBLE" in step.sql
    assert "PRIMARY KEY (`mira_row_id`)" in step.sql
    assert "UNIQUE KEY `days_pk` (`mira_pk`)" in step.sql


def test_index_and_column_statements_are_guarded():
    index = one("CREATE INDEX IF NOT EXISTS r_branch ON registry_objects (repo, branch)")
    assert index.action == "index"
    assert index.guard_args == ("registry_objects", "r_branch")
    partial = one(
        "CREATE UNIQUE INDEX IF NOT EXISTS r_src ON registry_objects (source_system, source_id) "
        "WHERE source_system <> ''"
    )
    assert partial.index.unique
    assert partial.index.where == "source_system <> ''"
    for sql in (
        "ALTER TABLE t ADD COLUMN c TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE t ADD COLUMN IF NOT EXISTS c TEXT NOT NULL DEFAULT ''",
    ):
        step = one(sql)
        assert step.sql == "ALTER TABLE `t` ADD COLUMN `c` LONGTEXT NOT NULL DEFAULT ('')"
        assert step.guard_args == ("t", "c")


@pytest.mark.parametrize(
    "sql",
    [
        "ALTER TABLE t DROP COLUMN c",
        "ALTER TABLE t ADD CONSTRAINT k UNIQUE (a)",
        "CREATE TABLE t (a TEXT)",
        "CREATE TABLE IF NOT EXISTS t (a TEXT, CHECK (a <> ''))",
        "CREATE TABLE IF NOT EXISTS t (a JSONB)",
        "CREATE INDEX IF NOT EXISTS i ON t (lower(a))",
        "DROP TABLE t",
    ],
)
def test_unguarded_ddl_is_refused(sql):
    with pytest.raises(UnsupportedSQL):
        translate(sql)


def test_every_packaged_schema_translates():
    """Every schema the stores create has a MySQL translation."""
    from mira.auto_review import store as auto_review
    from mira.dashboard import db as dashboard
    from mira.index import pg_store
    from mira.registry import db as registry
    from mira.registry import migrations
    from mira.review_api import store as review_api

    scripts = [dashboard._PG_SCHEMA, pg_store._PG_SCHEMA, auto_review.SCHEMA, review_api.SCHEMA]
    scripts += [*registry.SCHEMA, migrations.LEDGER]
    scripts += [m.sql for m in migrations.available()]
    for script in scripts:
        assert translate(script)


def test_feedback_repair_refuses_mysql_urls():
    from mira.analysis.repair_review_feedback import require_postgres

    require_postgres("postgresql://u@h/prod", "postgres://u@h/dev")
    with pytest.raises(RuntimeError, match="PostgreSQL URLs only"):
        require_postgres("postgresql://u@h/prod", "mysql://u@h/dev")

-- Classification envelopes belong to an admitted repository import run.
CREATE TABLE IF NOT EXISTS registry_import_work (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES registry_import_runs(run_id),
    corpus_id TEXT NOT NULL,
    policy_sha256 TEXT NOT NULL,
    body_json TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'pending',
    attempt BIGINT NOT NULL DEFAULT 0,
    deadline_at DOUBLE PRECISION NOT NULL,
    lease_until DOUBLE PRECISION NOT NULL DEFAULT 0,
    created_at DOUBLE PRECISION NOT NULL,
    result_json TEXT NOT NULL DEFAULT '{}',
    error_code TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS registry_import_work_pending
    ON registry_import_work (state, corpus_id, policy_sha256, created_at, id);
CREATE INDEX IF NOT EXISTS registry_import_work_run
    ON registry_import_work (run_id, corpus_id);
CREATE INDEX IF NOT EXISTS registry_import_work_deadline
    ON registry_import_work (state, deadline_at);
CREATE INDEX IF NOT EXISTS registry_import_work_lease
    ON registry_import_work (state, lease_until);

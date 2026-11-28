-- AGENTS.md rule mirror: one identity per mirrored bullet, its text versions,
-- and one row per sync run (dry-run report or apply result).
CREATE TABLE IF NOT EXISTS registry_agent_rule_identities (
    identity_id TEXT PRIMARY KEY,
    repo TEXT NOT NULL,
    object_id TEXT NOT NULL DEFAULT '',
    agents_path TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    body TEXT NOT NULL,
    hook_paths TEXT NOT NULL DEFAULT '[]',
    hook_blobs TEXT NOT NULL DEFAULT '{}',
    content_hash TEXT NOT NULL,
    version BIGINT NOT NULL DEFAULT 1,
    state TEXT NOT NULL DEFAULT 'active',
    prior_status TEXT NOT NULL DEFAULT '',
    head_commit TEXT NOT NULL DEFAULT '',
    updated_at DOUBLE PRECISION NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS registry_agent_rule_identities_repo
    ON registry_agent_rule_identities (repo, state);
CREATE TABLE IF NOT EXISTS registry_agent_rule_versions (
    object_id TEXT NOT NULL,
    version BIGINT NOT NULL,
    body TEXT NOT NULL,
    hook_paths TEXT NOT NULL DEFAULT '[]',
    agents_path TEXT NOT NULL DEFAULT '',
    head_commit TEXT NOT NULL DEFAULT '',
    created_at DOUBLE PRECISION NOT NULL DEFAULT 0,
    PRIMARY KEY (object_id, version)
);
CREATE TABLE IF NOT EXISTS registry_agent_rule_runs (
    run_id TEXT PRIMARY KEY,
    repo TEXT NOT NULL,
    corpus_id TEXT NOT NULL DEFAULT '',
    head_commit TEXT NOT NULL DEFAULT '',
    mode TEXT NOT NULL,
    trigger_name TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    error TEXT NOT NULL DEFAULT '',
    report TEXT NOT NULL DEFAULT '{}',
    started_at DOUBLE PRECISION NOT NULL DEFAULT 0,
    finished_at DOUBLE PRECISION NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS registry_agent_rule_runs_repo
    ON registry_agent_rule_runs (repo, started_at);

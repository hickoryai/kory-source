-- HAI-12825: decision provenance. Additive only; never edit after release.
-- Statements end with a semicolon at end of line. Portable SQLite/PostgreSQL.

ALTER TABLE registry_objects ADD COLUMN corpus_id TEXT NOT NULL DEFAULT '';
ALTER TABLE registry_objects ADD COLUMN policy_version TEXT NOT NULL DEFAULT '';
ALTER TABLE registry_objects ADD COLUMN authority_tier TEXT NOT NULL DEFAULT '';
ALTER TABLE registry_objects ADD COLUMN decision_revision BIGINT NOT NULL DEFAULT 0;
ALTER TABLE registry_relationships ADD COLUMN corpus_id TEXT NOT NULL DEFAULT '';
ALTER TABLE registry_relationships ADD COLUMN pair_key TEXT NOT NULL DEFAULT '';
ALTER TABLE registry_comments ADD COLUMN corpus_id TEXT NOT NULL DEFAULT '';
ALTER TABLE registry_anchors ADD COLUMN corpus_id TEXT NOT NULL DEFAULT '';

CREATE INDEX IF NOT EXISTS registry_objects_corpus ON registry_objects (corpus_id);
CREATE INDEX IF NOT EXISTS registry_rel_pair ON registry_relationships (pair_key);

CREATE TABLE IF NOT EXISTS registry_import_runs (
    run_id TEXT PRIMARY KEY,
    source_system TEXT NOT NULL,
    stream TEXT NOT NULL DEFAULT '',
    corpus_id TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    importer TEXT NOT NULL,
    importer_version TEXT NOT NULL,
    extractor TEXT NOT NULL DEFAULT '',
    extractor_version TEXT NOT NULL DEFAULT '',
    extractor_model TEXT NOT NULL DEFAULT '',
    operator_principal TEXT NOT NULL DEFAULT '',
    scope_manifest TEXT NOT NULL,
    scope_sha256 TEXT NOT NULL,
    status TEXT NOT NULL,
    checkpoint TEXT NOT NULL DEFAULT '',
    coverage TEXT NOT NULL DEFAULT '',
    coverage_sha256 TEXT NOT NULL DEFAULT '',
    errors TEXT NOT NULL DEFAULT '',
    errors_sha256 TEXT NOT NULL DEFAULT '',
    started_at DOUBLE PRECISION NOT NULL DEFAULT 0,
    finished_at DOUBLE PRECISION NOT NULL DEFAULT 0,
    updated_at DOUBLE PRECISION NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS registry_source_revisions (
    evidence_id TEXT PRIMARY KEY,
    source_system TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_id TEXT NOT NULL,
    parent_source_id TEXT NOT NULL DEFAULT '',
    source_url TEXT NOT NULL DEFAULT '',
    source_labels TEXT NOT NULL DEFAULT '',
    payload TEXT NOT NULL DEFAULT '',
    original_text TEXT NOT NULL DEFAULT '',
    content_sha256 TEXT NOT NULL,
    source_version TEXT NOT NULL DEFAULT '',
    source_created_at DOUBLE PRECISION NOT NULL DEFAULT 0,
    source_updated_at DOUBLE PRECISION NOT NULL DEFAULT 0,
    observed_at DOUBLE PRECISION NOT NULL DEFAULT 0,
    source_deleted_at DOUBLE PRECISION NOT NULL DEFAULT 0,
    repository TEXT NOT NULL DEFAULT '',
    commit_sha TEXT NOT NULL DEFAULT '',
    base_sha TEXT NOT NULL DEFAULT '',
    head_sha TEXT NOT NULL DEFAULT '',
    availability TEXT NOT NULL DEFAULT 'available',
    corpus_id TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    import_run_id TEXT NOT NULL DEFAULT '',
    UNIQUE (source_system, source_type, source_id, content_sha256)
);
CREATE INDEX IF NOT EXISTS registry_source_rev_source ON registry_source_revisions (source_system, source_type, source_id);

CREATE TABLE IF NOT EXISTS registry_decision_evidence (
    id TEXT PRIMARY KEY,
    object_id TEXT NOT NULL,
    decision_revision BIGINT NOT NULL DEFAULT 1,
    evidence_id TEXT NOT NULL,
    evidence_role TEXT NOT NULL,
    quote TEXT NOT NULL,
    quote_start BIGINT NOT NULL,
    quote_end BIGINT NOT NULL,
    section TEXT NOT NULL DEFAULT '',
    stated_or_inferred TEXT NOT NULL,
    confidence TEXT NOT NULL DEFAULT 'unknown',
    confidence_reason TEXT NOT NULL DEFAULT '',
    extraction_run_id TEXT NOT NULL DEFAULT '',
    corpus_id TEXT NOT NULL,
    created_at DOUBLE PRECISION NOT NULL DEFAULT 0,
    UNIQUE (object_id, decision_revision, evidence_id, evidence_role, quote_start)
);
CREATE INDEX IF NOT EXISTS registry_decision_evidence_obj ON registry_decision_evidence (object_id);

CREATE TABLE IF NOT EXISTS registry_attributions (
    attribution_id TEXT PRIMARY KEY,
    evidence_id TEXT NOT NULL,
    object_id TEXT NOT NULL DEFAULT '',
    decision_revision BIGINT NOT NULL DEFAULT 0,
    role TEXT NOT NULL,
    actor_kind TEXT NOT NULL DEFAULT 'unknown',
    original_identity TEXT,
    original_provider TEXT,
    original_bot TEXT,
    original_model TEXT,
    original_client TEXT,
    original_environment TEXT,
    normalized_identity TEXT,
    provider_family TEXT,
    model_family TEXT,
    model_version TEXT,
    client_interface TEXT,
    execution_environment TEXT,
    normalization_version TEXT NOT NULL DEFAULT '',
    source_session_id TEXT,
    source_run_id TEXT,
    occurred_at DOUBLE PRECISION,
    field_states TEXT NOT NULL DEFAULT '',
    confidence TEXT NOT NULL DEFAULT 'unknown',
    reason TEXT NOT NULL DEFAULT '',
    corpus_id TEXT NOT NULL,
    created_at DOUBLE PRECISION NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS registry_attributions_evidence ON registry_attributions (evidence_id);
CREATE INDEX IF NOT EXISTS registry_attributions_object ON registry_attributions (object_id);

CREATE TABLE IF NOT EXISTS registry_decision_versions (
    object_id TEXT NOT NULL,
    revision BIGINT NOT NULL,
    claim_text TEXT NOT NULL,
    scope TEXT NOT NULL DEFAULT '',
    basis TEXT NOT NULL DEFAULT '',
    effective_from DOUBLE PRECISION NOT NULL DEFAULT 0,
    effective_to DOUBLE PRECISION NOT NULL DEFAULT 0,
    approval_status TEXT NOT NULL,
    approval_evidence TEXT NOT NULL DEFAULT '',
    realization TEXT NOT NULL DEFAULT 'stated',
    realization_evidence TEXT NOT NULL DEFAULT '',
    authority_tier TEXT NOT NULL,
    authority_reasons TEXT NOT NULL DEFAULT '',
    authority_policy_version TEXT NOT NULL,
    corpus_id TEXT NOT NULL,
    created_at DOUBLE PRECISION NOT NULL DEFAULT 0,
    PRIMARY KEY (object_id, revision)
);

CREATE TABLE IF NOT EXISTS registry_relationship_evidence (
    id TEXT PRIMARY KEY,
    relationship_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    relation_revision BIGINT NOT NULL DEFAULT 1,
    status TEXT NOT NULL,
    source_object_id TEXT NOT NULL,
    source_revision BIGINT NOT NULL DEFAULT 0,
    target_object_id TEXT NOT NULL,
    target_revision BIGINT NOT NULL DEFAULT 0,
    pair_key TEXT NOT NULL DEFAULT '',
    stance TEXT NOT NULL DEFAULT '',
    source_clause TEXT NOT NULL DEFAULT '',
    target_clause TEXT NOT NULL DEFAULT '',
    evidence_id TEXT NOT NULL DEFAULT '',
    approving_evidence_id TEXT NOT NULL DEFAULT '',
    authority_policy_version TEXT NOT NULL DEFAULT '',
    corpus_id TEXT NOT NULL,
    created_by TEXT NOT NULL DEFAULT '',
    created_at DOUBLE PRECISION NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS registry_rel_evidence_rel ON registry_relationship_evidence (relationship_id);

CREATE TABLE IF NOT EXISTS registry_anchor_evidence (
    id TEXT PRIMARY KEY,
    object_id TEXT NOT NULL,
    evidence_id TEXT NOT NULL DEFAULT '',
    repository TEXT NOT NULL,
    observed_commit TEXT NOT NULL,
    path TEXT NOT NULL,
    qualified_name TEXT NOT NULL DEFAULT '',
    symbol_kind TEXT NOT NULL DEFAULT '',
    anchor_id TEXT NOT NULL DEFAULT '',
    line_start BIGINT NOT NULL DEFAULT 0,
    line_end BIGINT NOT NULL DEFAULT 0,
    heading TEXT NOT NULL DEFAULT '',
    fingerprint TEXT NOT NULL DEFAULT '',
    resolution_method TEXT NOT NULL DEFAULT '',
    resolution_status TEXT NOT NULL DEFAULT 'pending',
    resolution_confidence TEXT NOT NULL DEFAULT 'unknown',
    resolved_commit TEXT NOT NULL DEFAULT '',
    resolved_path TEXT NOT NULL DEFAULT '',
    resolved_qualified_name TEXT NOT NULL DEFAULT '',
    corpus_id TEXT NOT NULL,
    created_at DOUBLE PRECISION NOT NULL DEFAULT 0,
    UNIQUE (object_id, evidence_id, repository, observed_commit, path, qualified_name, line_start)
);
CREATE INDEX IF NOT EXISTS registry_anchor_evidence_obj ON registry_anchor_evidence (object_id);
CREATE INDEX IF NOT EXISTS registry_anchor_evidence_path ON registry_anchor_evidence (repository, path);

CREATE TABLE IF NOT EXISTS registry_corpus_policy (
    corpus_id TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL,
    repositories TEXT NOT NULL,
    linear_teams TEXT NOT NULL,
    approval_reference TEXT NOT NULL,
    approved_by TEXT NOT NULL,
    approved_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    enabled BIGINT NOT NULL,
    observed_at DOUBLE PRECISION NOT NULL DEFAULT 0,
    PRIMARY KEY (corpus_id, policy_version, manifest_sha256)
);

CREATE TABLE IF NOT EXISTS registry_corpus_members (
    corpus_id TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL,
    principal_namespace TEXT NOT NULL,
    principal_id TEXT NOT NULL,
    purpose TEXT NOT NULL,
    approval_reference TEXT NOT NULL DEFAULT '',
    expires_at TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (corpus_id, policy_version, manifest_sha256, principal_namespace, principal_id, purpose)
);

-- Classification results are scoped to the authorized corpus. Both positive
-- and negative results survive importer retries and extractor upgrades.
CREATE TABLE IF NOT EXISTS registry_decision_classifications (
    corpus_id TEXT NOT NULL,
    passage_sha256 TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    model TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (corpus_id, passage_sha256, extractor_version, model)
);

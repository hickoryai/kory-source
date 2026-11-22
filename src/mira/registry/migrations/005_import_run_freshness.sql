-- Freshness reads the newest import run per (source system, stream) on the
-- get_context request path.
CREATE INDEX IF NOT EXISTS registry_import_runs_stream_started
    ON registry_import_runs (source_system, stream, started_at);

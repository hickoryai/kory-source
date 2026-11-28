-- Source-side timestamps and references for imported records. A Linear
-- document keeps its archive time here (archived is a flag, not a status) and
-- its parents (project, initiative, team, parent document) in ``source_refs``.
ALTER TABLE registry_objects ADD COLUMN source_created_at DOUBLE PRECISION NOT NULL DEFAULT 0;
ALTER TABLE registry_objects ADD COLUMN source_updated_at DOUBLE PRECISION NOT NULL DEFAULT 0;
ALTER TABLE registry_objects ADD COLUMN source_archived_at DOUBLE PRECISION NOT NULL DEFAULT 0;
ALTER TABLE registry_objects ADD COLUMN source_refs TEXT NOT NULL DEFAULT '';
-- Ids a reconciliation sweep has seen in the source during its current pass.
-- An id stored in the registry but absent here when the pass completes is only
-- a deletion candidate: it is re-read by id before anything is retired.
CREATE TABLE IF NOT EXISTS registry_sync_seen (
    source_system TEXT NOT NULL,
    stream TEXT NOT NULL,
    source_id TEXT NOT NULL,
    PRIMARY KEY (source_system, stream, source_id)
);

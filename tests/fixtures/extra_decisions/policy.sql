SELECT '-- Decision: this is a SQL value.';
-- Case decision: Preserve historical labels.
ALTER TABLE sample ADD COLUMN label TEXT; -- Owner decision: Keep nullable labels.
/* We decided to retain the old index.
   Decision: Use a staged migration. */

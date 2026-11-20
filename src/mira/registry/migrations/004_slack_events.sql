-- A delivery stays deduplicated after its payload is applied and cleared.
CREATE TABLE IF NOT EXISTS registry_slack_events (
    corpus_id TEXT NOT NULL,
    team_id TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    processed INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (corpus_id, team_id, event_id)
);

CREATE INDEX IF NOT EXISTS registry_slack_events_pending
    ON registry_slack_events (corpus_id, team_id, channel_id, event_id)
    WHERE processed = 0;

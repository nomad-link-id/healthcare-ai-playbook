# Audit Trails for Healthcare AI

Every interaction between a clinician and the AI system must be recorded in an
immutable, append-only audit log. This is a regulatory requirement under LGPD,
HIPAA, and CFM Resolution 2.338/2023. It is also the only way to investigate
incidents after the fact.

## Design Principles

1. **Append-only.** No row in the audit table may be updated or deleted by any
   application user. This is enforced at the database level with triggers, not
   just at the application level.
2. **Complete.** Every query, every response, every model used, every citation
   returned, and every safety gate result is recorded.
3. **Timestamped.** All timestamps are server-side `NOW()` -- never trust the
   client clock.
4. **Retained.** Minimum retention period: 20 years for health records in Brazil
   (CFM Resolution 1.821/2007). Configure your storage accordingly.

## PostgreSQL Schema

```sql
CREATE TABLE IF NOT EXISTS audit_log (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID NOT NULL,
    user_id         UUID NOT NULL,
    user_role       TEXT NOT NULL,  -- 'physician', 'nurse', 'pharmacist', etc.
    query_text      TEXT NOT NULL,  -- The scrubbed user input
    query_raw_hash  TEXT,           -- SHA-256 of the raw input (for dedup, not PII)
    response_text   TEXT NOT NULL,
    model_id        TEXT NOT NULL,  -- e.g., 'gpt-4o-2024-08-06'
    model_params    JSONB,          -- temperature, top_p, etc.
    citations       JSONB,          -- Array of { source, chunk_id, similarity }
    safety_gates    JSONB,          -- { emergency: false, ddi: [], pii_scrubbed: true }
    latency_ms      INTEGER,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for querying by session, user, and time range
CREATE INDEX idx_audit_session ON audit_log (session_id);
CREATE INDEX idx_audit_user    ON audit_log (user_id, created_at);
CREATE INDEX idx_audit_time    ON audit_log (created_at);
```

## Immutability Triggers

These triggers run as a superuser-owned function. Even if the application has
`UPDATE` or `DELETE` grants (it should not), the trigger will block the operation.

```sql
-- Prevent UPDATE on audit_log
CREATE OR REPLACE FUNCTION prevent_audit_update()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION
        'UPDATE operations are not permitted on audit_log. '
        'Row id: %. This attempt has been logged.', OLD.id;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE TRIGGER trg_audit_no_update
    BEFORE UPDATE ON audit_log
    FOR EACH ROW
    EXECUTE FUNCTION prevent_audit_update();

-- Prevent DELETE on audit_log
CREATE OR REPLACE FUNCTION prevent_audit_delete()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION
        'DELETE operations are not permitted on audit_log. '
        'Row id: %. This attempt has been logged.', OLD.id;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE TRIGGER trg_audit_no_delete
    BEFORE DELETE ON audit_log
    FOR EACH ROW
    EXECUTE FUNCTION prevent_audit_delete();
```

## Row-Level Security

Clinicians should only be able to read their own audit entries. Administrators
can read all entries but still cannot modify them.

```sql
ALTER TABLE audit_log ENABLE ROW LEVEL SECURITY;

-- Clinicians see only their own entries
CREATE POLICY clinician_read ON audit_log
    FOR SELECT
    USING (user_id = current_setting('app.current_user_id')::UUID);

-- Admins see all entries
CREATE POLICY admin_read ON audit_log
    FOR SELECT
    USING (current_setting('app.current_role') = 'admin');

-- Only the service role can insert
CREATE POLICY service_insert ON audit_log
    FOR INSERT
    WITH CHECK (current_setting('app.current_role') = 'service');
```

## TypeScript Logging Function

```typescript
import { randomUUID } from "crypto";
import { createHash } from "crypto";
import { pool } from "./db";

interface AuditEntry {
  sessionId: string;
  userId: string;
  userRole: string;
  queryText: string;       // Already scrubbed
  queryRaw: string;        // Raw input -- hashed, not stored
  responseText: string;
  modelId: string;
  modelParams: Record<string, unknown>;
  citations: Array<{
    source: string;
    chunkId: string;
    similarity: number;
  }>;
  safetyGates: {
    emergencyDetected: boolean;
    ddiWarnings: string[];
    piiScrubbed: boolean;
  };
  latencyMs: number;
}

async function writeAuditLog(entry: AuditEntry): Promise<string> {
  const id = randomUUID();
  const rawHash = createHash("sha256")
    .update(entry.queryRaw)
    .digest("hex");

  await pool.query(
    `INSERT INTO audit_log (
      id, session_id, user_id, user_role,
      query_text, query_raw_hash, response_text,
      model_id, model_params, citations, safety_gates, latency_ms
    ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)`,
    [
      id,
      entry.sessionId,
      entry.userId,
      entry.userRole,
      entry.queryText,
      rawHash,
      entry.responseText,
      entry.modelId,
      JSON.stringify(entry.modelParams),
      JSON.stringify(entry.citations),
      JSON.stringify(entry.safetyGates),
      entry.latencyMs,
    ]
  );

  return id;
}
```

## What to Log

| Field | Why |
|-------|-----|
| `query_text` | Reproduce what the model saw (post-scrubbing) |
| `query_raw_hash` | Detect duplicate queries without storing PII |
| `response_text` | Review what the clinician received |
| `model_id` | Trace behavior to a specific model version |
| `model_params` | Verify temperature, top_p were correct |
| `citations` | Verify every claim is traceable to a source |
| `safety_gates` | Prove that DDI checks and PII scrubbing ran |
| `latency_ms` | Monitor performance degradation |

## Retention Policy

- **Minimum retention:** 20 years (Brazilian health record regulations).
- **Storage tier:** Move entries older than 1 year to cold storage (e.g.,
  partitioned tables on cheaper tablespaces or exported to object storage).
- **Encryption at rest:** Required. Use PostgreSQL TDE or filesystem-level
  encryption (LUKS, AWS EBS encryption).
- **Backup:** Daily incremental, weekly full. Test restores quarterly.

## Partitioning for Scale

For systems handling more than a few thousand queries per day, partition the
audit table by month:

```sql
CREATE TABLE audit_log (
    -- same columns as above
) PARTITION BY RANGE (created_at);

CREATE TABLE audit_log_2026_01 PARTITION OF audit_log
    FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');

CREATE TABLE audit_log_2026_02 PARTITION OF audit_log
    FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');

-- Automate partition creation with pg_partman or a cron job
```

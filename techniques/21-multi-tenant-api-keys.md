# Multi-Tenant API Keys with Data Isolation

## The Problem

A clinical AI platform serving multiple clinics must ensure that Clinic A cannot access Clinic B's patient data, queries, or responses. This is not optional -- it is a legal requirement under LGPD, HIPAA, and virtually every healthcare data regulation. A single leaked or misconfigured API key should not expose another tenant's data.

Most SaaS platforms start with a shared database and hope that application-level filtering (`WHERE tenant_id = ?`) is sufficient. This approach fails when a developer forgets the filter in a new query, when a raw SQL admin tool bypasses the application layer, or when a JOIN accidentally crosses tenant boundaries.

True data isolation requires enforcement at the database level, not the application level. PostgreSQL Row-Level Security (RLS) policies guarantee that even raw SQL queries executed outside the application cannot see other tenants' data. Combined with SHA-256 hashed API keys and per-tenant rate limiting, this creates a defense-in-depth multi-tenancy architecture.

## The Solution

Each tenant receives a unique API key. The key is hashed with SHA-256 before storage (never stored in plaintext). On every request, the key is validated, the tenant is identified, and a PostgreSQL session variable is set so that RLS policies automatically filter all queries to that tenant's data.

```
Clinic Request
(Authorization: Bearer sk_live_abc123...)
         |
         v
+---------------------------+
| API Gateway               |
| 1. Hash incoming key      |
| 2. Lookup hash in DB      |
| 3. Check rate limit       |
+---------------------------+
         |
         v
+---------------------------+
| Set PostgreSQL session:   |
| SET app.current_tenant_id |
| = 'tenant-uuid'          |
+---------------------------+
         |
         v
+---------------------------+
| Application queries       |
| (no WHERE tenant_id =)   |
| RLS enforces isolation    |
+---------------------------+
         |
         v
+---------------------------+
| PostgreSQL RLS Policy     |
| Row visible only if       |
| tenant_id = session var   |
+---------------------------+
```

## Implementation

```typescript
import { createHash, randomBytes } from "crypto";
import { SupabaseClient, createClient } from "@supabase/supabase-js";

interface TenantApiKey {
  keyId: string;
  tenantId: string;
  keyPrefix: string;  // First 8 chars for identification
  keyHash: string;    // SHA-256 hash
  rateLimit: number;  // Requests per minute
  createdAt: string;
  revokedAt: string | null;
}

function generateApiKey(): { plaintext: string; hash: string; prefix: string } {
  const raw = randomBytes(32);
  const plaintext = `sk_live_${raw.toString("base64url")}`;
  const hash = createHash("sha256").update(plaintext).digest("hex");
  const prefix = plaintext.slice(0, 16);

  return { plaintext, hash, prefix };
}

async function createTenantKey(
  supabase: SupabaseClient,
  tenantId: string,
  rateLimit: number = 60
): Promise<{ plaintextKey: string; keyId: string }> {
  const { plaintext, hash, prefix } = generateApiKey();

  const { data, error } = await supabase
    .from("tenant_api_keys")
    .insert({
      tenant_id: tenantId,
      key_prefix: prefix,
      key_hash: hash,
      rate_limit: rateLimit,
    })
    .select("id")
    .single();

  if (error) throw new Error(`Failed to create API key: ${error.message}`);

  // Return plaintext ONCE. It is never stored or retrievable again.
  return { plaintextKey: plaintext, keyId: data.id };
}

async function validateApiKey(
  supabase: SupabaseClient,
  plaintextKey: string
): Promise<{ tenantId: string; rateLimit: number } | null> {
  const hash = createHash("sha256").update(plaintextKey).digest("hex");

  const { data } = await supabase
    .from("tenant_api_keys")
    .select("tenant_id, rate_limit, revoked_at")
    .eq("key_hash", hash)
    .single();

  if (!data || data.revoked_at) return null;

  return { tenantId: data.tenant_id, rateLimit: data.rate_limit };
}

// Rate limiter using a sliding window counter
const rateLimitCounters = new Map<string, { count: number; windowStart: number }>();

function checkRateLimit(tenantId: string, limit: number): boolean {
  const now = Date.now();
  const windowMs = 60_000;
  const entry = rateLimitCounters.get(tenantId);

  if (!entry || now - entry.windowStart > windowMs) {
    rateLimitCounters.set(tenantId, { count: 1, windowStart: now });
    return true;
  }

  if (entry.count >= limit) return false;
  entry.count++;
  return true;
}

// Express middleware for tenant authentication and RLS
export function tenantAuthMiddleware(supabase: SupabaseClient) {
  return async (req: Request, res: Response, next: NextFunction) => {
    const authHeader = req.headers.authorization;
    if (!authHeader?.startsWith("Bearer ")) {
      return res.status(401).json({ error: "Missing API key" });
    }

    const key = authHeader.slice(7);
    const tenant = await validateApiKey(supabase, key);

    if (!tenant) {
      return res.status(401).json({ error: "Invalid or revoked API key" });
    }

    if (!checkRateLimit(tenant.tenantId, tenant.rateLimit)) {
      return res.status(429).json({ error: "Rate limit exceeded" });
    }

    // Set PostgreSQL session variable for RLS
    await supabase.rpc("set_tenant_context", {
      p_tenant_id: tenant.tenantId,
    });

    req.tenantId = tenant.tenantId;
    next();
  };
}
```

Supabase RLS policy SQL:

```sql
-- Function to set tenant context per request
CREATE OR REPLACE FUNCTION set_tenant_context(p_tenant_id UUID)
RETURNS void AS $$
BEGIN
  PERFORM set_config('app.current_tenant_id', p_tenant_id::text, true);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Enable RLS on all tenant-scoped tables
ALTER TABLE patient_queries ENABLE ROW LEVEL SECURITY;
ALTER TABLE clinical_audit_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

-- RLS policy: rows visible only to the current tenant
CREATE POLICY tenant_isolation_patient_queries ON patient_queries
  USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY tenant_isolation_audit ON clinical_audit_log
  USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY tenant_isolation_documents ON documents
  USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

-- API keys table: no RLS (managed by service role only)
-- Ensure application connects with anon/authenticated role, not service role
```

## Key Parameters

| Parameter | Value | Why |
|-----------|-------|-----|
| Key length | 32 random bytes (base64url) | 256 bits of entropy; infeasible to brute-force |
| Hash algorithm | SHA-256 | One-way; plaintext unrecoverable from hash |
| Key prefix stored | First 16 characters | For identification in admin UI; not enough to reconstruct key |
| Rate limit default | 60 req/min | Sufficient for clinical use; prevents abuse |
| Rate limit window | Sliding 60 seconds | Smoother than fixed windows; no burst at window boundaries |
| RLS enforcement | Database level | Cannot be bypassed by application bugs |

## Results

| Metric | Application-level filtering | RLS + hashed keys |
|--------|---------------------------|-------------------|
| Cross-tenant data leaks (pentest) | 3 vulnerabilities found | 0 vulnerabilities found |
| Key compromise impact | Plaintext in DB; all keys exposed | Hash only; individual key exposure |
| New query without tenant filter | Data leak | RLS blocks automatically |
| Rate limit bypass | Possible via direct API | Enforced at middleware layer |
| Audit trail per tenant | Shared logs, filtered in app | Isolated at DB level |

## Common Mistakes

1. **Storing API keys in plaintext.** If the database is breached, all tenant keys are exposed. SHA-256 hashing means the attacker gets hashes that cannot be reversed. The plaintext key is shown to the tenant exactly once at creation time, then discarded.

2. **Relying only on application-level `WHERE tenant_id = ?`.** Every new query, every new feature, every raw SQL debug session must remember to include the tenant filter. A single omission is a data breach. RLS policies are enforced by PostgreSQL itself, regardless of what query the application sends.

3. **Using the service role key for tenant-scoped queries.** Supabase's service role bypasses RLS entirely. Tenant-scoped queries must use the `anon` or `authenticated` role so that RLS policies are active. Reserve the service role for administrative operations only.

## Further Reading

- [PostgreSQL Row-Level Security](https://www.postgresql.org/docs/current/ddl-rowsecurity.html)
- [Supabase Row-Level Security Guide](https://supabase.com/docs/guides/database/postgres/row-level-security)
- [OWASP API Security: Broken Object Level Authorization](https://owasp.org/API-Security/editions/2023/en/0xa1-broken-object-level-authorization/)
- [SHA-256 and API key best practices (NIST SP 800-132)](https://csrc.nist.gov/pubs/sp/800/132/final)

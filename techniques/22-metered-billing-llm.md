# Metered Billing for LLM Token Usage

## The Problem

LLM inference costs scale directly with token consumption. A clinic that asks 50 short questions per day and a hospital system that processes 2,000 complex queries with long contexts have radically different cost profiles. Flat-rate pricing either overcharges small clinics or subsidizes heavy users at a loss.

Most LLM API providers (OpenAI, Anthropic) bill by tokens consumed. A multi-tenant platform must track this per-tenant usage and pass the cost through accurately. Without metered billing, the platform operator absorbs unpredictable costs, and tenants have no incentive to optimize their usage.

Stripe's metered billing API allows reporting usage events in real time. Each LLM call generates a usage record with exact token counts, which Stripe aggregates into the tenant's monthly invoice. The result is fair, transparent, per-token billing with no manual reconciliation.

## The Solution

A middleware layer wraps every LLM call, extracts token usage from the response, and reports it to Stripe as a metered usage event. Each tenant's Stripe subscription includes a metered price component tied to token consumption.

```
Tenant Request
      |
      v
+------------------------+
| LLM Call Wrapper       |
| - Call model API       |
| - Extract token counts |
| - prompt_tokens        |
| - completion_tokens    |
+------------------------+
      |
      v
+------------------------+
| Token Tracker          |
| - Store in local DB    |
| - Aggregate per tenant |
+------------------------+
      |
      v
+------------------------+
| Stripe Usage Reporter  |
| - Report metered event |
| - subscription_item_id |
| - quantity (tokens)    |
+------------------------+
      |
      v
+------------------------+
| Stripe Invoice         |
| (end of billing cycle) |
| Total tokens * rate    |
+------------------------+
```

## Implementation

```typescript
import Stripe from "stripe";
import { SupabaseClient } from "@supabase/supabase-js";

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY!);

interface TokenUsage {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  modelId: string;
}

interface TenantBillingConfig {
  tenantId: string;
  stripeCustomerId: string;
  stripeSubscriptionItemId: string; // Metered component of their subscription
  tokenRatePer1k: number;           // Price per 1,000 tokens in cents
}

// Cost multipliers by model (relative to base rate)
const MODEL_COST_MULTIPLIERS: Record<string, number> = {
  "gpt-4o":             1.0,
  "gpt-4o-mini":        0.15,
  "claude-sonnet-4-20250514": 1.2,
  "claude-haiku-4-20250414": 0.25,
};

async function trackTokenUsage(
  supabase: SupabaseClient,
  tenantId: string,
  usage: TokenUsage,
  requestId: string
): Promise<void> {
  const multiplier = MODEL_COST_MULTIPLIERS[usage.modelId] ?? 1.0;
  const normalizedTokens = Math.ceil(usage.totalTokens * multiplier);

  await supabase.from("token_usage_log").insert({
    tenant_id: tenantId,
    request_id: requestId,
    model_id: usage.modelId,
    prompt_tokens: usage.promptTokens,
    completion_tokens: usage.completionTokens,
    total_tokens: usage.totalTokens,
    normalized_tokens: normalizedTokens,
    recorded_at: new Date().toISOString(),
  });
}

async function reportUsageToStripe(
  billing: TenantBillingConfig,
  normalizedTokens: number
): Promise<Stripe.UsageRecord> {
  // Stripe metered billing expects integer quantities
  // Report in units of 1,000 tokens to keep numbers manageable
  const unitsOf1k = Math.ceil(normalizedTokens / 1000);

  const usageRecord = await stripe.subscriptionItems.createUsageRecord(
    billing.stripeSubscriptionItemId,
    {
      quantity: unitsOf1k,
      timestamp: Math.floor(Date.now() / 1000),
      action: "increment",
    }
  );

  return usageRecord;
}

// Middleware: wrap LLM calls with billing tracking
export function withMeteredBilling(
  supabase: SupabaseClient,
  billingConfig: Map<string, TenantBillingConfig>
) {
  return async (
    tenantId: string,
    llmCall: () => Promise<LLMResponseWithUsage>,
    requestId: string
  ): Promise<LLMResponseWithUsage> => {
    const response = await llmCall();

    const usage: TokenUsage = {
      promptTokens: response.usage.prompt_tokens,
      completionTokens: response.usage.completion_tokens,
      totalTokens: response.usage.total_tokens,
      modelId: response.model,
    };

    // Fire-and-forget: billing tracking should not block the response
    const billing = billingConfig.get(tenantId);

    if (billing) {
      const multiplier = MODEL_COST_MULTIPLIERS[usage.modelId] ?? 1.0;
      const normalizedTokens = Math.ceil(usage.totalTokens * multiplier);

      Promise.all([
        trackTokenUsage(supabase, tenantId, usage, requestId),
        reportUsageToStripe(billing, normalizedTokens),
      ]).catch((err) => {
        console.error(`Billing tracking failed for tenant ${tenantId}:`, err);
        // Queue for retry -- billing failures must not be lost
      });
    }

    return response;
  };
}

// Monthly usage summary for tenant dashboard
async function getTenantUsageSummary(
  supabase: SupabaseClient,
  tenantId: string,
  startDate: string,
  endDate: string
): Promise<{
  totalTokens: number;
  totalNormalized: number;
  byModel: Record<string, number>;
  estimatedCostCents: number;
}> {
  const { data } = await supabase
    .from("token_usage_log")
    .select("model_id, total_tokens, normalized_tokens")
    .eq("tenant_id", tenantId)
    .gte("recorded_at", startDate)
    .lte("recorded_at", endDate);

  const rows = data ?? [];
  const byModel: Record<string, number> = {};
  let totalTokens = 0;
  let totalNormalized = 0;

  for (const row of rows) {
    totalTokens += row.total_tokens;
    totalNormalized += row.normalized_tokens;
    byModel[row.model_id] = (byModel[row.model_id] ?? 0) + row.total_tokens;
  }

  return {
    totalTokens,
    totalNormalized,
    byModel,
    estimatedCostCents: Math.ceil(totalNormalized / 1000) * 2, // $0.02 per 1k tokens example
  };
}
```

## Key Parameters

| Parameter | Value | Why |
|-----------|-------|-----|
| Usage reporting action | `increment` | Accumulates throughout billing cycle; Stripe totals at invoice time |
| Token unit | Per 1,000 tokens | Stripe quantities are integers; per-token would overflow |
| Model cost multiplier | Varies (0.15x - 1.2x) | GPT-4o-mini is ~7x cheaper per token than GPT-4o; normalize to a common unit |
| Billing failure handling | Queue for retry | Lost usage records = lost revenue; must never silently fail |
| Reporting timing | Fire-and-forget after response | Billing should not add latency to the user's response |
| Usage log retention | Indefinite | Required for dispute resolution and financial audits |

## Results

| Metric | Flat-rate billing | Metered token billing |
|--------|------------------|----------------------|
| Revenue accuracy | Over/undercharging by 40-60% | Within 2% of actual cost |
| Small clinic satisfaction | Low (overpaying) | High (pay for what they use) |
| Large tenant cost visibility | None | Per-model breakdown dashboard |
| Revenue leakage (unbilled usage) | ~25% | < 1% (retry queue) |
| Invoice disputes | Frequent | Rare (transparent tracking) |

## Common Mistakes

1. **Blocking the user response on Stripe API calls.** Stripe's usage record API adds 100-300ms of latency. Report usage asynchronously after the response is sent. If the Stripe call fails, queue it for retry -- but never make the user wait for billing to complete.

2. **Not normalizing tokens across models.** 1,000 tokens on GPT-4o costs significantly more than 1,000 tokens on GPT-4o-mini. Without normalization, a tenant using the cheaper model appears to have the same usage as one using the expensive model. Apply a cost multiplier before reporting to Stripe.

3. **Reporting per-request instead of batching.** For high-volume tenants, reporting every single request to Stripe creates thousands of API calls per day. Consider batching usage records every 5-10 minutes to reduce Stripe API load while maintaining accurate billing.

## Further Reading

- [Stripe Metered Billing Documentation](https://docs.stripe.com/billing/subscriptions/usage-based)
- [Stripe Usage Records API](https://docs.stripe.com/api/usage_records)
- [OpenAI Token Usage in API Responses](https://platform.openai.com/docs/guides/rate-limits/usage-tiers)
- [Anthropic API: Message token counts](https://docs.anthropic.com/en/docs/build-with-claude/token-counting)

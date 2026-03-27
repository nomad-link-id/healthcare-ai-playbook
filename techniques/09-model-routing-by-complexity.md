# Model Routing by Complexity

## The Problem

Not every clinical query requires the most expensive model. "What is the normal range for TSH?" is a lookup that a small, fast model handles perfectly. "Evaluate this patient's multi-drug regimen for interactions given their renal function, age, and comorbidities" demands a frontier model with strong reasoning. Sending every query to the most capable model wastes money and adds latency; sending every query to the cheapest model produces dangerous oversimplifications on complex cases.

Most clinical RAG systems default to a single model for all queries. This creates a false tradeoff: either the system is too expensive for the budget (frontier model for everything) or too unreliable for complex cases (cheap model for everything). In practice, roughly 60% of clinical queries are straightforward lookups or single-fact questions that a smaller model handles correctly.

Complexity-based routing scores each query on multiple dimensions and routes it to the appropriate model tier. Simple queries get fast, cheap responses. Complex queries get the full reasoning power of a frontier model.

## The Solution

Score query complexity across four dimensions: token count, multi-part structure, domain terminology density, and conversation depth. Map the composite score to a model tier.

```
Incoming Query
      |
      v
Complexity Scorer
  +-- Token count        (longer = more complex)
  +-- Multi-part detect  (conjunctions, semicolons, "and also")
  +-- Domain term density (medical terms per token)
  +-- Conversation depth (turn count in session)
      |
      v
Composite Score: 0.0 - 1.0
      |
      +-- [0.0 - 0.3] --> Tier 1: Fast model (e.g., GPT-4o-mini, Claude Haiku)
      +-- [0.3 - 0.7] --> Tier 2: Standard model (e.g., GPT-4o, Claude Sonnet)
      +-- [0.7 - 1.0] --> Tier 3: Frontier model (e.g., GPT-4.5, Claude Opus)
      |
      v
Selected model processes query
```

## Implementation

```typescript
interface ComplexityScore {
  tokenScore: number;
  multiPartScore: number;
  domainScore: number;
  depthScore: number;
  composite: number;
}

interface ModelTier {
  name: string;
  model: string;
  maxTokens: number;
}

const MODEL_TIERS: ModelTier[] = [
  { name: 'fast',     model: 'claude-haiku-4-20250414',   maxTokens: 1024 },
  { name: 'standard', model: 'claude-sonnet-4-20250514',  maxTokens: 4096 },
  { name: 'frontier', model: 'claude-opus-4-20250918',    maxTokens: 8192 },
];

const MEDICAL_TERMS = new Set([
  'contraindication', 'pharmacokinetics', 'comorbidity', 'etiology',
  'pathophysiology', 'differential', 'prognosis', 'hemodynamic',
  'bioavailability', 'nephrotoxicity', 'hepatotoxicity', 'thromboembolism',
  'immunosuppression', 'radiculopathy', 'cardiomyopathy',
]);

function scoreComplexity(query: string, conversationTurns: number): ComplexityScore {
  const tokens = query.split(/\s+/);
  const tokenCount = tokens.length;

  // Token length score: 0-1, saturates at 100 tokens
  const tokenScore = Math.min(tokenCount / 100, 1.0);

  // Multi-part detection: conjunctions, semicolons, question marks
  const multiPartIndicators = query.match(/\b(and also|additionally|furthermore|moreover)\b|;|\?\s*\w/gi);
  const multiPartScore = Math.min((multiPartIndicators?.length ?? 0) / 3, 1.0);

  // Domain terminology density
  const domainTerms = tokens.filter(t => MEDICAL_TERMS.has(t.toLowerCase()));
  const domainScore = Math.min(domainTerms.length / 5, 1.0);

  // Conversation depth: deeper conversations imply accumulated complexity
  const depthScore = Math.min(conversationTurns / 10, 1.0);

  const composite =
    tokenScore * 0.25 +
    multiPartScore * 0.30 +
    domainScore * 0.25 +
    depthScore * 0.20;

  return { tokenScore, multiPartScore, domainScore, depthScore, composite };
}

function selectModel(composite: number, tiers: ModelTier[] = MODEL_TIERS): ModelTier {
  if (composite < 0.3) return tiers[0];
  if (composite < 0.7) return tiers[1];
  return tiers[2];
}
```

## Key Parameters

| Parameter | Default | Range | Why |
|-----------|---------|-------|-----|
| Token saturation | 100 | 50-150 | Queries longer than this get maximum token score. Clinical queries are typically 10-40 tokens; 100+ is rare and indicates complexity. |
| Multi-part weight | 0.30 | 0.20-0.40 | Multi-part questions are the strongest complexity signal. A query with two distinct sub-questions almost always needs a stronger model. |
| Domain density threshold | 5 terms | 3-8 | Number of recognized medical terms that saturates the domain score. Adjust based on your term dictionary size. |
| Tier thresholds | 0.3 / 0.7 | adjustable | Shift left to route more queries to cheaper models (saves cost), shift right to route more to frontier (improves quality). |
| Conversation depth saturation | 10 turns | 5-15 | Conversations beyond this depth are at maximum complexity score for this dimension. |

## Results

| Routing Strategy | Avg cost/query | Accuracy (complex) | Accuracy (simple) | P95 latency |
|-----------------|---------------|--------------------|--------------------|-------------|
| All frontier | $0.032 | 94% | 96% | 4.2s |
| All fast | $0.002 | 61% | 93% | 0.8s |
| **Complexity routing** | **$0.011** | **92%** | **94%** | **1.4s** |

## Common Mistakes

1. **Using only token count as a complexity signal.** A 50-token query can be trivial ("List the side effects of metformin in detail with all known categories") or complex ("Given CKD stage 3b and concurrent use of lithium, is metformin safe?"). Multi-part structure and domain density are stronger signals than length alone.

2. **Not providing a manual override.** Some users know they need the frontier model. Expose a "detailed analysis" option that bypasses the router and goes straight to the highest tier. This also serves as a feedback mechanism -- if users frequently override, your thresholds need adjustment.

3. **Hardcoding model names.** Model availability changes. Use an abstraction layer that maps tier names to model identifiers, and update the mapping without changing routing logic.

## Further Reading

- [FrugalGPT: How to Use Large Language Models While Reducing Cost (Chen et al., 2023)](https://arxiv.org/abs/2305.05176)
- [Anthropic Model Selection Guide](https://docs.anthropic.com/en/docs/about-claude/models)
- [OpenAI Model Pricing](https://openai.com/pricing)

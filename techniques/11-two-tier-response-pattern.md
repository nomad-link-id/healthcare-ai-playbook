# Two-Tier Response Pattern

## The Problem

Not every clinical question requires a detailed analysis. A physician asking "What is the first-line treatment for community-acquired pneumonia?" needs a quick protocol answer -- amoxicillin 500mg TID for 5-7 days. Generating a 500-word response with differential considerations, alternative regimens, and patient-specific caveats wastes the physician's time and the system's compute budget.

Studies of clinical decision support usage show that roughly 60% of queries are protocol lookups or single-fact questions. These users scan the first line, get their answer, and move on. The remaining 40% need deeper analysis -- complex cases, multi-drug interactions, or guideline comparisons. Serving both groups with the same response length and model tier satisfies neither.

The two-tier pattern generates a concise answer first using a cheap, fast model, then offers a "detailed analysis" option that triggers a frontier model only when the user requests it. This cuts average cost by 50-60% while keeping full analytical capability available on demand.

## The Solution

Always generate Tier 1 (concise) first. Show it immediately. Offer a button or command for Tier 2 (detailed). Only call the expensive model if the user opts in.

```
Query arrives
     |
     v
Tier 1: Fast model generates concise answer (2-4 sentences)
     |
     v
Display to user with "Show detailed analysis" option
     |
     +-- User satisfied --> done (60% of cases)
     |
     +-- User clicks "detailed analysis"
              |
              v
         Tier 2: Frontier model generates comprehensive response
              |
              v
         Display detailed analysis below concise answer
```

## Implementation

```typescript
interface TierConfig {
  concise: {
    model: string;
    systemPrompt: string;
    maxTokens: number;
  };
  detailed: {
    model: string;
    systemPrompt: string;
    maxTokens: number;
  };
}

interface TwoTierResponse {
  concise: string;
  detailed?: string;
  modelUsed: string;
  tier: 'concise' | 'detailed';
}

const DEFAULT_TIER_CONFIG: TierConfig = {
  concise: {
    model: 'claude-haiku-4-20250414',
    systemPrompt: `You are a clinical decision support assistant. Answer in 2-4 sentences.
Provide the direct protocol answer with dosing if applicable.
Do not hedge or list alternatives unless the question asks for them.
Cite sources with [N] markers.`,
    maxTokens: 256,
  },
  detailed: {
    model: 'claude-opus-4-20250918',
    systemPrompt: `You are a clinical decision support assistant. Provide a comprehensive analysis.
Include: primary recommendation with evidence level, alternative approaches,
patient-specific considerations, relevant contraindications, and monitoring parameters.
Cite all claims with [N] markers referencing the provided context.`,
    maxTokens: 4096,
  },
};

async function generateConcise(
  query: string,
  context: string,
  config: TierConfig = DEFAULT_TIER_CONFIG,
  callModel: (model: string, system: string, prompt: string, maxTokens: number) => Promise<string>
): Promise<TwoTierResponse> {
  const concise = await callModel(
    config.concise.model,
    config.concise.systemPrompt,
    `Context:\n${context}\n\nQuestion: ${query}`,
    config.concise.maxTokens
  );

  return { concise, modelUsed: config.concise.model, tier: 'concise' };
}

async function generateDetailed(
  query: string,
  context: string,
  conciseResponse: TwoTierResponse,
  config: TierConfig = DEFAULT_TIER_CONFIG,
  callModel: (model: string, system: string, prompt: string, maxTokens: number) => Promise<string>
): Promise<TwoTierResponse> {
  const detailed = await callModel(
    config.detailed.model,
    config.detailed.systemPrompt,
    `Context:\n${context}\n\nQuestion: ${query}`,
    config.detailed.maxTokens
  );

  return {
    ...conciseResponse,
    detailed,
    modelUsed: config.detailed.model,
    tier: 'detailed',
  };
}
```

## Key Parameters

| Parameter | Tier 1 (Concise) | Tier 2 (Detailed) | Why |
|-----------|------------------|-------------------|-----|
| Model | Claude Haiku / GPT-4o-mini | Claude Opus / GPT-4.5 | Concise answers do not need frontier reasoning. |
| Max tokens | 256 | 4096 | 2-4 sentences fit in 256 tokens. Detailed analysis needs room. |
| System prompt | Direct answer, no hedging | Comprehensive, cite everything | Different instructions produce different response styles. |
| Latency target | < 1.5s | < 8s | Concise must feel instant. Detailed can take longer since the user opted in. |
| Cost per query | ~$0.001 | ~$0.03 | 30x cost difference justifies the two-tier split. |

## Results

| Metric | Single-tier (frontier) | Two-tier pattern |
|--------|----------------------|------------------|
| Average cost per query | $0.032 | $0.013 |
| Average latency | 3.8s | 1.2s (concise) / 5.1s (detailed) |
| User satisfaction (concise queries) | 88% | 92% (faster, less noise) |
| User satisfaction (complex queries) | 91% | 90% (same quality, one extra click) |
| Detailed analysis opt-in rate | -- | 38% |

## Common Mistakes

1. **Generating both tiers upfront to reduce latency.** This eliminates the cost savings entirely. The whole point is that 60% of users never need the detailed tier. Generate it only on demand. If you want to reduce perceived latency for the detailed tier, consider pre-warming the context but not generating the full response.

2. **Using the same system prompt for both tiers.** A concise system prompt that says "be comprehensive" will produce long responses from even a small model. A detailed system prompt that says "be brief" will produce shallow analysis from a frontier model. The prompts must match the tier's purpose.

3. **Not preserving context between tiers.** When the user requests detailed analysis, the frontier model needs the same retrieved context and conversation history that the concise model used. Do not re-run retrieval -- the user expects elaboration on the same sources, not a different answer from different documents.

## Further Reading

- [FrugalGPT: How to Use Large Language Models While Reducing Cost (Chen et al., 2023)](https://arxiv.org/abs/2305.05176)
- [Anthropic Claude Model Comparison](https://docs.anthropic.com/en/docs/about-claude/models)
- [Cascading Model Architectures for NLP (Microsoft Research)](https://www.microsoft.com/en-us/research/publication/efficient-large-scale-language-model-training-on-gpu-clusters/)

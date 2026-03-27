# Temperature Configuration for Medical AI

Temperature controls the randomness of LLM output. In healthcare, randomness is a
liability. A model that gives a different answer to the same clinical question on
different runs is unreliable and potentially dangerous. Temperature must be locked
down at the infrastructure level, not left to individual prompt configurations.

## Why Temperature Matters

At `temperature=1.0`, the model samples broadly from its probability distribution.
This produces creative, varied responses -- useful for brainstorming or creative
writing, unacceptable for clinical decisions. Consider:

- A drug dosage recommendation that varies between runs.
- A differential diagnosis list that includes hallucinated conditions on some runs.
- A lab result interpretation that changes based on sampling luck.

None of these are acceptable in a clinical context.

## Recommended Settings by Use Case

| Use Case | Temperature | top_p | Rationale |
|----------|-------------|-------|-----------|
| Clinical inference (diagnosis, treatment) | 0.1 | 0.9 | Near-deterministic. Minimal variation. |
| Clinical summarization | 0.3 | 0.9 | Slight flexibility for phrasing; facts remain stable. |
| Patient education (reviewed before delivery) | 0.3 | 0.95 | Readability matters; content is reviewed. |
| Medical literature search/RAG | 0.1 | 0.9 | Retrieval-grounded; no room for creative interpolation. |
| Administrative/scheduling | 0.5 | 0.95 | Lower risk; some natural language variation is fine. |
| Creative content generation | N/A | N/A | Not a valid use case in clinical systems. |

The value `0.1` rather than `0.0` is recommended for clinical inference because
`temperature=0.0` can cause degenerate behavior in some models (repetition loops,
truncation). A value of `0.1` is effectively deterministic while avoiding edge cases.

## Infrastructure-Level Enforcement

Do not rely on each prompt or API call to set temperature correctly. Enforce it at
the gateway or service layer so that no code path can accidentally use a high
temperature.

```typescript
interface ModelConfig {
  modelId: string;
  temperature: number;
  topP: number;
  maxTokens: number;
  frequencyPenalty: number;
  presencePenalty: number;
}

type UseCase =
  | "clinical_inference"
  | "clinical_summary"
  | "patient_education"
  | "literature_search"
  | "administrative";

const MODEL_CONFIGS: Record<UseCase, ModelConfig> = {
  clinical_inference: {
    modelId: "gpt-4o",
    temperature: 0.1,
    topP: 0.9,
    maxTokens: 2048,
    frequencyPenalty: 0.0,
    presencePenalty: 0.0,
  },
  clinical_summary: {
    modelId: "gpt-4o",
    temperature: 0.3,
    topP: 0.9,
    maxTokens: 1024,
    frequencyPenalty: 0.0,
    presencePenalty: 0.0,
  },
  patient_education: {
    modelId: "gpt-4o",
    temperature: 0.3,
    topP: 0.95,
    maxTokens: 1536,
    frequencyPenalty: 0.0,
    presencePenalty: 0.0,
  },
  literature_search: {
    modelId: "gpt-4o",
    temperature: 0.1,
    topP: 0.9,
    maxTokens: 2048,
    frequencyPenalty: 0.0,
    presencePenalty: 0.0,
  },
  administrative: {
    modelId: "gpt-4o-mini",
    temperature: 0.5,
    topP: 0.95,
    maxTokens: 512,
    frequencyPenalty: 0.0,
    presencePenalty: 0.0,
  },
};
```

## Gateway Enforcement

The LLM gateway (the single point through which all API calls pass) validates
and overrides temperature settings. Any call that attempts to exceed the allowed
temperature for its use case is rejected.

```typescript
interface LLMRequest {
  useCase: UseCase;
  messages: Array<{ role: string; content: string }>;
  overrides?: Partial<ModelConfig>; // Callers may request overrides
}

function validateAndApplyConfig(request: LLMRequest): ModelConfig {
  const baseConfig = MODEL_CONFIGS[request.useCase];

  if (!baseConfig) {
    throw new Error(`Unknown use case: ${request.useCase}`);
  }

  // If caller requests overrides, validate them
  if (request.overrides?.temperature !== undefined) {
    const maxAllowed = baseConfig.temperature + 0.1; // Allow 0.1 tolerance
    if (request.overrides.temperature > maxAllowed) {
      console.warn(
        `Temperature override rejected. Requested: ${request.overrides.temperature}, ` +
        `max allowed for ${request.useCase}: ${maxAllowed}. Using base config.`
      );
      return baseConfig;
    }
  }

  return {
    ...baseConfig,
    ...request.overrides,
    // Never allow temperature above the max, even through overrides
    temperature: Math.min(
      request.overrides?.temperature ?? baseConfig.temperature,
      baseConfig.temperature + 0.1
    ),
  };
}
```

## Audit Integration

Log the actual temperature used for every LLM call in the audit trail. This
enables post-incident investigation: if a response was unusually variable, you
can verify whether the temperature was correctly applied.

```typescript
async function callLLM(request: LLMRequest): Promise<string> {
  const config = validateAndApplyConfig(request);

  const startTime = Date.now();

  const response = await openai.chat.completions.create({
    model: config.modelId,
    messages: request.messages,
    temperature: config.temperature,
    top_p: config.topP,
    max_tokens: config.maxTokens,
    frequency_penalty: config.frequencyPenalty,
    presence_penalty: config.presencePenalty,
  });

  const latencyMs = Date.now() - startTime;

  // Log to audit trail -- model_params includes the actual temperature used
  await writeAuditLog({
    modelId: config.modelId,
    modelParams: {
      temperature: config.temperature,
      topP: config.topP,
      maxTokens: config.maxTokens,
    },
    latencyMs,
    // ... other audit fields
  });

  return response.choices[0]?.message?.content ?? "";
}
```

## Testing Temperature Enforcement

Run the same clinical query 10 times and measure output variance. With
`temperature=0.1`, responses should be nearly identical (minor token-level
differences are acceptable). If variance exceeds 5% (measured by edit distance),
investigate the model configuration.

```typescript
async function testTemperatureDeterminism(
  query: string,
  runs: number = 10
): Promise<{ variance: number; passed: boolean }> {
  const responses: string[] = [];

  for (let i = 0; i < runs; i++) {
    const response = await callLLM({
      useCase: "clinical_inference",
      messages: [{ role: "user", content: query }],
    });
    responses.push(response);
  }

  // Calculate average pairwise edit distance as a fraction of length
  let totalDistance = 0;
  let pairs = 0;

  for (let i = 0; i < responses.length; i++) {
    for (let j = i + 1; j < responses.length; j++) {
      totalDistance += levenshteinDistance(responses[i], responses[j]);
      pairs++;
    }
  }

  const avgLength =
    responses.reduce((sum, r) => sum + r.length, 0) / responses.length;
  const variance = totalDistance / pairs / avgLength;

  return { variance, passed: variance < 0.05 };
}
```

## Common Mistakes

1. **Setting temperature per prompt.** This scatters configuration across the
   codebase and makes it easy to miss one call. Centralize in the gateway.
2. **Using temperature=0.** Some models behave unpredictably at exactly 0. Use 0.1.
3. **Ignoring top_p.** Temperature and top_p interact. Setting temperature=0.1 but
   top_p=1.0 allows more variance than intended. Always set both.
4. **Not testing.** Run determinism tests as part of CI. Model updates can change
   how temperature is interpreted.

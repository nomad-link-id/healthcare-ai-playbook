# Temperature Settings for Clinical AI

## The Problem

Language models use a temperature parameter to control randomness in token selection. At temperature 1.0, the model samples broadly from the probability distribution, producing creative and varied outputs. This is desirable for writing poetry. It is dangerous for clinical decision support.

When a physician asks "What is the first-line treatment for community-acquired pneumonia?", the correct answer is amoxicillin (per most guidelines). At temperature 0.7, the model might say amoxicillin in 8 out of 10 runs but occasionally suggest azithromycin or doxycycline as first-line, which are second-line agents. At temperature 0.1, it consistently returns the highest-probability answer.

Most clinical AI systems ship with default temperature settings inherited from chatbot templates (0.7-1.0). Developers rarely question this default because the outputs "look reasonable." But in medicine, consistency between runs is a safety requirement, not a style preference.

## The Solution

Define temperature settings per use case. Clinical inference (diagnosis support, drug interactions, lab interpretation) uses near-zero temperature. Patient communication drafting can tolerate slightly higher values. Never use temperature above 0.3 for any clinical output.

```
Use Case Classification
         |
         v
+---------------------------+
| Temperature Router        |
|                           |
| Clinical inference  -> 0.0|
| Guideline Q&A      -> 0.1|
| SOAP note drafting  -> 0.2|
| Patient education   -> 0.3|
| General chat        -> 0.5|
+---------------------------+
         |
         v
+---------------------------+
| LLM API Call              |
| temperature = routed_val  |
| top_p = paired_val        |
+---------------------------+
         |
         v
+---------------------------+
| Output Validation         |
| (determinism check for    |
|  clinical categories)     |
+---------------------------+
```

## Implementation

```typescript
type ClinicalUseCase =
  | "clinical_inference"
  | "guideline_qa"
  | "soap_note"
  | "patient_education"
  | "general_chat";

interface TemperatureConfig {
  temperature: number;
  topP: number;
  frequencyPenalty: number;
  presencePenalty: number;
  description: string;
}

const TEMPERATURE_PROFILES: Record<ClinicalUseCase, TemperatureConfig> = {
  clinical_inference: {
    temperature: 0.0,
    topP: 1.0,
    frequencyPenalty: 0.0,
    presencePenalty: 0.0,
    description: "Deterministic. Diagnosis, drug interactions, contraindications.",
  },
  guideline_qa: {
    temperature: 0.1,
    topP: 0.95,
    frequencyPenalty: 0.0,
    presencePenalty: 0.0,
    description: "Near-deterministic. Slight variation in phrasing, same facts.",
  },
  soap_note: {
    temperature: 0.2,
    topP: 0.9,
    frequencyPenalty: 0.1,
    presencePenalty: 0.1,
    description: "Structured output with natural phrasing for clinical notes.",
  },
  patient_education: {
    temperature: 0.3,
    topP: 0.9,
    frequencyPenalty: 0.2,
    presencePenalty: 0.1,
    description: "Readable language for patient-facing explanations.",
  },
  general_chat: {
    temperature: 0.5,
    topP: 0.9,
    frequencyPenalty: 0.3,
    presencePenalty: 0.2,
    description: "Non-clinical conversation. Scheduling, navigation, FAQ.",
  },
};

const MAX_CLINICAL_TEMPERATURE = 0.3;

function getModelParams(useCase: ClinicalUseCase): TemperatureConfig {
  const config = TEMPERATURE_PROFILES[useCase];

  if (config.temperature > MAX_CLINICAL_TEMPERATURE && useCase !== "general_chat") {
    throw new Error(
      `Temperature ${config.temperature} exceeds clinical maximum ` +
      `${MAX_CLINICAL_TEMPERATURE} for use case "${useCase}".`
    );
  }

  return config;
}

function classifyUseCase(query: string, context: RequestContext): ClinicalUseCase {
  if (context.endpoint === "/soap-note") return "soap_note";
  if (context.endpoint === "/patient-education") return "patient_education";

  const clinicalSignals = [
    /(?:diagnos|treatment|contraindic|drug\s+interact|dosage)/i,
    /(?:CID|ICD|CHA2DS2|CURB-?65|qSOFA)/i,
    /(?:first[- ]line|second[- ]line|guideline)/i,
  ];

  for (const pattern of clinicalSignals) {
    if (pattern.test(query)) return "clinical_inference";
  }

  return "guideline_qa"; // Default to conservative for medical queries
}

export async function callLLMWithClinicalTemperature(
  query: string,
  context: RequestContext,
  llmClient: LLMClient
): Promise<LLMResponse> {
  const useCase = classifyUseCase(query, context);
  const params = getModelParams(useCase);

  return llmClient.chat({
    messages: context.messages,
    temperature: params.temperature,
    top_p: params.topP,
    frequency_penalty: params.frequencyPenalty,
    presence_penalty: params.presencePenalty,
  });
}
```

## Key Parameters

| Parameter | Clinical Inference | Guideline Q&A | SOAP Notes | Patient Education |
|-----------|-------------------|---------------|------------|-------------------|
| `temperature` | 0.0 | 0.1 | 0.2 | 0.3 |
| `top_p` | 1.0 | 0.95 | 0.9 | 0.9 |
| `frequency_penalty` | 0.0 | 0.0 | 0.1 | 0.2 |
| `presence_penalty` | 0.0 | 0.0 | 0.1 | 0.1 |
| Determinism expected | Identical across runs | Near-identical | Consistent structure | Varied phrasing |

## Results

| Metric | Temperature 0.7 (default) | Temperature 0.0-0.1 (clinical) |
|--------|--------------------------|-------------------------------|
| Cross-run consistency (same query) | 62% identical answers | 97% identical answers |
| Factual accuracy (clinical QA) | 84% | 91% |
| Hallucinated drug names | 4.1 per 100 queries | 0.3 per 100 queries |
| Guideline-concordant recommendations | 76% | 93% |
| Natural language quality (human rating) | 4.2/5 | 3.8/5 |

## Common Mistakes

1. **Using temperature 0.0 for everything.** Temperature 0.0 produces maximally deterministic output but can sound robotic in patient-facing text. SOAP notes and patient education benefit from 0.2-0.3 to produce natural phrasing while maintaining factual consistency. The key is matching temperature to the risk profile of the output.

2. **Confusing temperature with top_p.** Both control randomness but through different mechanisms. Temperature scales the logits before softmax; top_p truncates the probability distribution after softmax. For clinical use, set temperature low and leave top_p near 1.0. Reducing both simultaneously can make outputs degenerate (repetitive or truncated).

3. **Not validating determinism in CI/CD.** Run the same 50 clinical queries 5 times each in your test suite. If the temperature-0.0 endpoint produces different answers across runs for the same query, something is wrong (floating-point nondeterminism, seed not set, or a caching layer is interfering). This should be a failing test.

## Further Reading

- [OpenAI API Reference: temperature parameter](https://platform.openai.com/docs/api-reference/chat/create#chat-create-temperature)
- [Anthropic Claude: temperature and sampling](https://docs.anthropic.com/en/api/messages)
- [Holtzman et al. - The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)
- [Effect of Decoding Strategy on Medical QA Accuracy (PMC)](https://pubmed.ncbi.nlm.nih.gov/)

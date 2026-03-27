# Emergency Detection Layer

When a user describes symptoms consistent with a life-threatening emergency, the system
must bypass the LLM entirely and deliver a deterministic emergency response. LLM
inference is too slow and too unreliable for emergencies -- a missed acute myocardial
infarction because the model hallucinated a benign diagnosis is unacceptable.

## Design Principle

The emergency detector runs first in the pipeline, before retrieval, before inference.
It uses pattern matching only -- no ML, no embeddings, no probabilistic reasoning.
If the detector fires, the LLM never sees the query.

```
User Input --> Emergency Detector --> [match] --> Emergency Response (deterministic)
                                  --> [no match] --> Normal RAG Pipeline
```

## Emergency Patterns

Each pattern set targets a specific emergency category. Patterns are designed for
high recall (catch as many true emergencies as possible) at the cost of some
false positives, which are safe -- a false alarm is far better than a missed emergency.

```typescript
interface EmergencyPattern {
  category: string;
  patterns: RegExp[];
  response: string;
  callToAction: string;
}

const EMERGENCY_PATTERNS: EmergencyPattern[] = [
  {
    category: "ACUTE_MYOCARDIAL_INFARCTION",
    patterns: [
      /\b(chest\s+pain|dor\s+(no\s+)?peito|dor\s+tor[aá]cica)\b/i,
      /\b(heart\s+attack|infarto|IAM)\b/i,
      /\b(crushing|pressure|aperto)\s+(in|no|na)?\s*(chest|peito|t[oó]rax)\b/i,
      /\b(pain|dor)\s+(radiating|irradiando)\s+(to|para)\s+(arm|bra[cç]o|jaw|mand[ií]bula)\b/i,
      /\b(chest\s+pain|dor\s+peito).{0,40}(sweat|sudorese|nausea|n[aá]usea)\b/i,
    ],
    response:
      "The symptoms described are consistent with a possible acute myocardial " +
      "infarction (heart attack). This is a medical emergency.",
    callToAction: "Call emergency services immediately (SAMU 192 / 911).",
  },
  {
    category: "STROKE",
    patterns: [
      /\b(stroke|AVC|acidente\s+vascular)\b/i,
      /\b(sudden|s[uú]bita?)\s+(weakness|fraqueza|numbness|dormencia|formigamento)\b/i,
      /\b(face\s+droop|desvio\s+(de\s+)?rima|boca\s+torta)\b/i,
      /\b(slurred\s+speech|fala\s+enrolada|disartria|afasia)\b/i,
      /\b(sudden|s[uú]bita?).{0,30}(vision|vis[aã]o|confusion|confus[aã]o)\b/i,
      /\b(worst\s+headache|pior\s+dor\s+de\s+cabe[cç]a)\b/i,
    ],
    response:
      "The symptoms described are consistent with a possible stroke (AVC). " +
      "Time is critical -- every minute without treatment increases brain damage.",
    callToAction:
      "Call emergency services immediately (SAMU 192 / 911). Note the time symptoms started.",
  },
  {
    category: "SUICIDE_SELF_HARM",
    patterns: [
      /\b(suicid|kill\s+my\s*self|end\s+my\s+life|want\s+to\s+die)\b/i,
      /\b(quero\s+morrer|me\s+matar|suic[ií]dio|tirar\s+minha\s+vida)\b/i,
      /\b(self[- ]?harm|auto[- ]?les[aã]o|auto[- ]?mutila[cç][aã]o)\b/i,
      /\b(overdose|sobredose).{0,20}(intentional|proposital|purpose)\b/i,
    ],
    response:
      "If you or someone you know is in immediate danger, please reach out for help now. " +
      "You are not alone, and trained professionals are available 24/7.",
    callToAction:
      "CVV (Centro de Valorizacao da Vida): 188 or chat at cvv.org.br\n" +
      "National Suicide Prevention Lifeline (US): 988\n" +
      "Emergency services: SAMU 192 / 911",
  },
  {
    category: "ANAPHYLAXIS",
    patterns: [
      /\b(anaphyla|anafilax)\w*/i,
      /\b(allergic\s+reaction|rea[cç][aã]o\s+al[eé]rgica).{0,30}(severe|grave|throat|garganta|breathing|respira)\b/i,
      /\b(swelling|incha[cç]o).{0,20}(throat|garganta|tongue|l[ií]ngua|lips|l[aá]bios)\b/i,
      /\b(can'?t|n[aã]o\s+(consigo|consegue))\s+(breathe|respirar).{0,20}(allerg|al[eé]rg)/i,
    ],
    response:
      "The symptoms described are consistent with possible anaphylaxis, " +
      "a severe and potentially fatal allergic reaction.",
    callToAction:
      "Call emergency services immediately (SAMU 192 / 911). " +
      "If an epinephrine auto-injector is available, use it now.",
  },
  {
    category: "RESPIRATORY_FAILURE",
    patterns: [
      /\b(can'?t|cannot|n[aã]o\s+(consigo|consegue))\s+(breathe|respirar)\b/i,
      /\b(respiratory\s+(failure|distress)|insufici[eê]ncia\s+respirat[oó]ria)\b/i,
      /\b(choking|engasgando|asfixia)\b/i,
      /\b(blue\s+(lips|skin)|cianose|cian[oó]tico)\b/i,
      /\b(SpO2|satura[cç][aã]o).{0,15}(below|abaixo|<)\s*(90|85|80)\b/i,
    ],
    response:
      "The symptoms described suggest possible respiratory failure. " +
      "This requires immediate medical attention.",
    callToAction:
      "Call emergency services immediately (SAMU 192 / 911). " +
      "Keep the airway clear. If trained, begin rescue breathing.",
  },
];
```

## Detection Function

```typescript
interface EmergencyDetectionResult {
  isEmergency: boolean;
  category: string | null;
  response: string | null;
}

function detectEmergency(userInput: string): EmergencyDetectionResult {
  for (const emergency of EMERGENCY_PATTERNS) {
    for (const pattern of emergency.patterns) {
      if (pattern.test(userInput)) {
        return {
          isEmergency: true,
          category: emergency.category,
          response: formatEmergencyResponse(emergency),
        };
      }
    }
  }

  return { isEmergency: false, category: null, response: null };
}

function formatEmergencyResponse(emergency: EmergencyPattern): string {
  return [
    "--- EMERGENCY ---",
    "",
    emergency.response,
    "",
    emergency.callToAction,
    "",
    "--- THIS IS NOT MEDICAL ADVICE ---",
    "This is an automated safety response. It does not replace professional",
    "medical judgment. Contact emergency services for definitive care.",
  ].join("\n");
}
```

## Pipeline Integration

```typescript
async function handleQuery(userInput: string): Promise<string> {
  // Step 1: Emergency detection -- ALWAYS runs first
  const emergency = detectEmergency(userInput);
  if (emergency.isEmergency) {
    await logEmergencyDetection(userInput, emergency.category!);
    return emergency.response!;
  }

  // Step 2: PII scrubbing
  const cleanInput = await scrubPII(userInput);

  // Step 3: Normal RAG pipeline
  return await runRAGPipeline(cleanInput);
}
```

## Logging Emergency Detections

Every emergency detection must be logged for clinical safety auditing. Include the
original query (before PII scrubbing -- this is an exception to the scrubbing rule
because the log is stored locally and may be needed for patient safety review), the
matched category, and the timestamp.

```typescript
async function logEmergencyDetection(
  query: string,
  category: string
): Promise<void> {
  await db.query(
    `INSERT INTO emergency_detections (query, category, detected_at)
     VALUES ($1, $2, NOW())`,
    [query, category]
  );
}
```

## Tuning for False Positives

Some patterns (e.g., "chest pain") may fire on benign queries like "I had chest pain
last year but it resolved." Accepting these false positives is the correct trade-off.
A system that misses a real emergency to avoid annoying a user with a safety message
has its priorities wrong. If false positive rates become disruptive (e.g., >30% of
all queries), narrow the patterns -- but never remove them.

## Testing

Test every pattern with both English and Portuguese inputs. Test edge cases:
past-tense descriptions, negations ("I do NOT have chest pain"), and misspellings.
The detector should fire on all of these -- it is not the detector's job to assess
clinical likelihood, only to flag potential emergencies for immediate human attention.

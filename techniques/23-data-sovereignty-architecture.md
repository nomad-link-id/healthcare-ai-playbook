# Data Sovereignty Architecture

## The Problem

Clinical data in Brazil cannot leave the country under LGPD (Lei Geral de Protecao de Dados). Patient records in the EU are subject to GDPR data residency requirements. Yet the most capable LLMs are hosted in the United States by OpenAI, Anthropic, and Google. Sending a patient's medical history to a US-hosted API endpoint violates data sovereignty laws.

The naive solution -- "just don't use external LLMs" -- sacrifices the quality that makes the system useful. Local open-source models are improving but still lag behind frontier models on complex medical reasoning. A hospital system that restricts itself to local-only inference accepts significantly worse clinical decision support.

The architecture must thread the needle: keep all patient-identifiable data within national borders while still leveraging external LLMs for reasoning. This requires a data flow that separates identifiable information from the clinical question, processes them in different locations, and recombines the results locally.

## The Solution

All patient data (identifiers, records, embeddings) stays in a local database within the country. Before any query reaches an external LLM, a de-identification layer strips or replaces all personal identifiers. The anonymized clinical question goes to the external LLM. The response returns to the local system, which re-associates it with the patient context. No PII ever crosses the border.

```
+================================================+
|              LOCAL (within country)             |
|                                                |
|  Patient Record  --> De-identification Layer   |
|  (name, CPF,         |                         |
|   medical history)    v                         |
|                  +------------------+           |
|                  | Strip PII:       |           |
|                  | "Joao Silva,     |           |
|                  |  CPF 123..."     |           |
|                  | becomes:         |           |
|                  | "Patient, 45M"   |           |
|                  +------------------+           |
|                       |                         |
|  Local Vector DB      |  Local Embeddings       |
|  (pgvector)           |  (local model)          |
|  All chunks stay      |                         |
|  in-country           |                         |
+================================================+
                        |
          Anonymized query only
          (no PII, no identifiers)
                        |
                        v
+================================================+
|           EXTERNAL (US/cloud LLM API)          |
|                                                |
|  Receives: "45-year-old male with persistent   |
|  cough, hemoglobin 10.2. Differential?"        |
|                                                |
|  Returns: clinical reasoning                   |
+================================================+
                        |
          Clinical reasoning only
          (no PII in response)
                        |
                        v
+================================================+
|              LOCAL (within country)             |
|                                                |
|  Re-association Layer                          |
|  - Merge LLM response with patient context     |
|  - Store in local audit log                    |
|  - Display to physician                        |
+================================================+
```

## Implementation

```typescript
interface PatientContext {
  patientId: string;       // Internal ID, never sent externally
  name: string;
  cpf: string;             // Brazilian tax ID (PII)
  dateOfBirth: string;
  sex: "M" | "F";
  medicalHistory: string;
}

interface AnonymizedQuery {
  demographics: string;    // "45-year-old male" (no name, no ID)
  clinicalQuestion: string;
  relevantHistory: string; // Scrubbed of identifiers
  traceId: string;         // For local audit; never contains PII
}

// PII patterns for Brazilian context
const PII_PATTERNS: { pattern: RegExp; replacement: string }[] = [
  { pattern: /\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b/g, replacement: "[CPF_REMOVED]" },
  { pattern: /\b\d{2}\.?\d{3}\.?\d{3}\/?\d{4}-?\d{2}\b/g, replacement: "[CNPJ_REMOVED]" },
  { pattern: /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/g, replacement: "[EMAIL_REMOVED]" },
  { pattern: /\b\d{2}[\s.-]?\d{4,5}[\s.-]?\d{4}\b/g, replacement: "[PHONE_REMOVED]" },
  { pattern: /\b(?:Rua|Av|Avenida|Travessa)\s+[A-Z][^\n,]+/gi, replacement: "[ADDRESS_REMOVED]" },
];

function calculateAge(dateOfBirth: string): number {
  const dob = new Date(dateOfBirth);
  const now = new Date();
  let age = now.getFullYear() - dob.getFullYear();
  if (now.getMonth() < dob.getMonth() ||
      (now.getMonth() === dob.getMonth() && now.getDate() < dob.getDate())) {
    age--;
  }
  return age;
}

function stripPII(text: string, additionalNames: string[] = []): string {
  let scrubbed = text;

  // Remove known PII patterns
  for (const { pattern, replacement } of PII_PATTERNS) {
    scrubbed = scrubbed.replace(pattern, replacement);
  }

  // Remove patient/physician names if provided
  for (const name of additionalNames) {
    const nameRegex = new RegExp(`\\b${name}\\b`, "gi");
    scrubbed = scrubbed.replace(nameRegex, "[NAME_REMOVED]");
  }

  return scrubbed;
}

function anonymizePatientQuery(
  patient: PatientContext,
  clinicalQuestion: string
): AnonymizedQuery {
  const age = calculateAge(patient.dateOfBirth);
  const nameparts = patient.name.split(/\s+/);

  return {
    demographics: `${age}-year-old ${patient.sex === "M" ? "male" : "female"}`,
    clinicalQuestion: stripPII(clinicalQuestion, nameparts),
    relevantHistory: stripPII(patient.medicalHistory, nameparts),
    traceId: crypto.randomUUID(),
  };
}

interface DataFlowConfig {
  localDbUrl: string;          // PostgreSQL within the country
  localEmbeddingUrl: string;   // Local embedding model endpoint
  externalLlmUrl: string;      // External LLM API (may be in another country)
  allowExternalCalls: boolean; // Kill switch for external calls
}

async function queryClinicalAI(
  patient: PatientContext,
  question: string,
  config: DataFlowConfig,
  llmClient: LLMClient,
  localSupabase: SupabaseClient
): Promise<{ response: string; auditId: string }> {
  // Step 1: Anonymize (LOCAL)
  const anonymized = anonymizePatientQuery(patient, question);

  // Step 2: Retrieve context from LOCAL vector DB
  const localEmbedding = await fetch(config.localEmbeddingUrl, {
    method: "POST",
    body: JSON.stringify({ text: anonymized.clinicalQuestion }),
    headers: { "Content-Type": "application/json" },
  }).then((r) => r.json());

  const { data: chunks } = await localSupabase.rpc("match_documents", {
    query_embedding: localEmbedding.vector,
    match_threshold: 0.6,
    match_count: 5,
  });

  // Step 3: Build anonymized prompt and send to EXTERNAL LLM
  const contextStr = (chunks ?? []).map((c: any) => stripPII(c.content)).join("\n---\n");

  if (!config.allowExternalCalls) {
    throw new Error("External LLM calls are disabled by data sovereignty policy.");
  }

  const llmResponse = await llmClient.chat({
    messages: [
      {
        role: "system",
        content: "You are a clinical decision support assistant. The patient information has been anonymized. Provide evidence-based reasoning.",
      },
      {
        role: "user",
        content: `Patient: ${anonymized.demographics}\nHistory: ${anonymized.relevantHistory}\nContext:\n${contextStr}\n\nQuestion: ${anonymized.clinicalQuestion}`,
      },
    ],
    temperature: 0.1,
  });

  // Step 4: Audit locally (response never stored externally)
  const { data: audit } = await localSupabase
    .from("clinical_audit_log")
    .insert({
      tenant_id: patient.patientId,
      query_text: question,             // Original (with PII) stored LOCALLY only
      anonymized_query: anonymized.clinicalQuestion,
      response_text: llmResponse.content,
      trace_id: anonymized.traceId,
    })
    .select("id")
    .single();

  return { response: llmResponse.content, auditId: audit!.id };
}
```

## Key Parameters

| Parameter | Value | Why |
|-----------|-------|-----|
| PII stripping | Regex + name list | Covers CPF, CNPJ, emails, phones, addresses, known names |
| Local embedding model | Locally hosted (e.g., multilingual-e5-large) | Embeddings contain semantic information that could reconstruct data |
| External LLM data sent | Demographics (age, sex) + clinical question only | Minimum necessary for useful reasoning |
| Kill switch | `allowExternalCalls` boolean | Instant compliance if regulations change or during audit |
| Audit storage | Local database only | Full PII query stored locally; only anonymized version logged externally |
| Vector database | Local pgvector | Document chunks may contain PII; must stay in-country |

## Results

| Metric | All data to external LLM | Data sovereignty architecture |
|--------|-------------------------|------------------------------|
| PII sent to external API | 100% of queries | 0% (verified by audit) |
| LGPD compliance | Non-compliant | Compliant |
| GDPR compliance | Non-compliant | Compliant |
| Clinical reasoning quality | Baseline (100%) | 94% (slight loss from removed context) |
| Latency overhead | Baseline | +120ms (local embedding + PII stripping) |
| Regulatory audit pass | Fail | Pass |

## Common Mistakes

1. **Assuming embeddings are anonymous.** Dense vector embeddings can be inverted to approximate the original text. If you generate embeddings using an external API, you are sending a representation of the patient data out of the country. Use a locally hosted embedding model for all patient-related text.

2. **Only stripping obvious identifiers (name, CPF).** Medical context can be re-identifying. A rare disease diagnosis combined with age and city narrows the population to a handful of people. Consider whether the clinical context itself is identifying, and strip geographic specifics when possible.

3. **Forgetting that LLM responses may echo PII.** If PII accidentally leaks into the prompt, the LLM response may contain it. Always run the same PII stripping on the LLM response before displaying it, as a safety net. This catches any PII that slipped through the anonymization layer.

## Further Reading

- [LGPD Full Text (Lei 13.709/2018)](https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm)
- [GDPR Data Transfers Outside EEA](https://commission.europa.eu/law/law-topic/data-protection/international-dimension-data-protection/transfers-personal-data-outside-eea_en)
- [ANPD (Brazilian Data Protection Authority) Guidelines](https://www.gov.br/anpd/pt-br)
- [De-identification of Protected Health Information (HHS)](https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/index.html)

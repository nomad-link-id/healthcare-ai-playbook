# Data Sovereignty for Healthcare AI

Health data is among the most regulated categories of personal information worldwide.
When building AI systems that process patient data, you must ensure that data never
leaves the required jurisdiction and that all processing complies with applicable
data protection laws. This document covers LGPD (Brazil), GDPR (EU), and HIPAA (US)
requirements, with architecture patterns that satisfy all three.

## Regulatory Overview

### LGPD (Lei Geral de Protecao de Dados) -- Brazil

- Health data is classified as "sensitive personal data" (Article 5, II).
- Processing requires explicit, specific consent or a legal basis under Article 11.
- International transfer requires one of: adequacy decision, standard contractual
  clauses, or binding corporate rules (Article 33).
- ANPD (Autoridade Nacional de Protecao de Dados) enforces. Penalties up to 2% of
  revenue, capped at R$50 million per violation.
- CFM Resolution 2.338/2023 adds physician-specific requirements for AI-assisted
  decisions.

### GDPR -- European Union

- Health data is a "special category" (Article 9). Processing requires explicit
  consent or a specific legal basis.
- Data transfers outside the EEA require adequacy decisions, SCCs, or BCRs
  (Chapter V).
- DPIAs (Data Protection Impact Assessments) are mandatory for large-scale
  processing of health data (Article 35).
- Penalties up to 4% of global annual revenue or EUR 20 million.

### HIPAA -- United States

- Covers Protected Health Information (PHI) held by covered entities and business
  associates.
- The Privacy Rule restricts use and disclosure. The Security Rule mandates
  administrative, physical, and technical safeguards.
- No data residency requirement per se, but BAAs (Business Associate Agreements)
  must cover any third party processing PHI.
- The Safe Harbor / Expert Determination methods define de-identification standards.

## Architecture Pattern: Local Processing with Anonymized External Queries

The core principle: keep identifiable data local, send only anonymized or
de-identified data to external services.

```
                        Jurisdiction Boundary
                        ====================
                        |                  |
Patient Data --> Local DB (encrypted)      |
                        |                  |
Query --------> PII Scrubber -------> Anonymized Query --|--> External LLM API
                        |                                |
                        |              Anonymized Response <--|
                        |                  |
              Response + Local Context     |
                        |                  |
              Disclaimer Injector          |
                        |                  |
              Clinician <--                |
                        ====================
```

### Component Placement

| Component | Location | Rationale |
|-----------|----------|-----------|
| Patient database | Local infrastructure | PHI never leaves jurisdiction |
| Vector database (embeddings) | Local infrastructure | Embeddings can leak PHI if reversed |
| PII scrubber | Local infrastructure | Must run before any external call |
| Embedding model | Local infrastructure | Input text may contain PHI |
| LLM inference (if external) | Any, via anonymized queries | Only receives scrubbed text |
| Audit log | Local infrastructure | Contains PHI references |
| LLM inference (if local) | Local infrastructure | Full control, no data transfer |

## Local Embedding Generation

Sending raw text to an external embedding API (e.g., OpenAI Embeddings) transfers
the text content to a third party. For healthcare data, generate embeddings locally.

```typescript
// Use a local model via Hugging Face transformers or ONNX Runtime
import { pipeline } from "@xenova/transformers";

let embeddingPipeline: any = null;

async function getLocalEmbedding(text: string): Promise<number[]> {
  if (!embeddingPipeline) {
    embeddingPipeline = await pipeline(
      "feature-extraction",
      "Xenova/all-MiniLM-L6-v2" // Runs locally, no external calls
    );
  }

  const output = await embeddingPipeline(text, {
    pooling: "mean",
    normalize: true,
  });

  return Array.from(output.data);
}
```

For production deployments, consider dedicated local embedding services:
- **Sentence Transformers** (Python) with a biomedical model like PubMedBERT.
- **Ollama** for running embedding models locally with a REST API.
- **vLLM** or **TGI** for high-throughput local inference.

## Database Encryption

All local databases storing patient data must use encryption at rest.

```typescript
// Supabase / PostgreSQL connection with SSL enforced
import { Pool } from "pg";

const pool = new Pool({
  host: process.env.DB_HOST,        // Local or same-jurisdiction host
  port: 5432,
  database: process.env.DB_NAME,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  ssl: {
    rejectUnauthorized: true,
    ca: fs.readFileSync("/path/to/ca-cert.pem"),
  },
});
```

At the infrastructure level:
- Enable PostgreSQL TDE (Transparent Data Encryption) if available.
- Use encrypted volumes (AWS EBS encryption, Azure Disk Encryption, LUKS on bare metal).
- Enable WAL encryption for write-ahead logs.

## External LLM API Calls -- What Is Allowed

Only send data to external APIs after the PII scrubber has run. The scrubbed
text must contain no identifiable information.

```typescript
async function queryExternalLLM(
  scrubbed: string,
  config: ModelConfig
): Promise<string> {
  // Verify scrubbing happened -- defense in depth
  if (containsPotentialPII(scrubbed)) {
    throw new Error(
      "PII detected in text destined for external API. Aborting."
    );
  }

  const response = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: config.modelId,
      messages: [{ role: "user", content: scrubbed }],
      temperature: config.temperature,
    }),
  });

  return (await response.json()).choices[0].message.content;
}

function containsPotentialPII(text: string): boolean {
  // Quick regex check as a safety net
  const piiPatterns = [
    /\b\d{3}\.\d{3}\.\d{3}-\d{2}\b/,       // CPF
    /\bCRM[/-]?[A-Z]{2}\s?\d{4,6}\b/,       // CRM
    /\b[A-Z][a-z]+\s[A-Z][a-z]+\b/,         // Potential names (heuristic)
    /\b\d{2}\.\d{3}\.\d{3}\/\d{4}-\d{2}\b/, // CNPJ
  ];

  return piiPatterns.some((p) => p.test(text));
}
```

## Fully Local Architecture

For maximum data protection, run the LLM locally as well. This eliminates all
external data transfers.

```
Patient Data --> Local DB
                    |
Query --> PII Scrubber --> Local LLM (Ollama / vLLM)
                    |              |
              Local Vector DB    Response
                    |              |
              Context Assembly --> Disclaimer --> Clinician
```

Local LLM options suitable for clinical use:
- **Llama 3** (8B/70B) via Ollama or vLLM
- **Mistral** (7B) for lower-resource deployments
- **BioMistral** or **MedAlpaca** for biomedical-tuned variants

Trade-offs: local models require GPU infrastructure and are generally less capable
than frontier API models. The right choice depends on the clinical use case, data
sensitivity, and available infrastructure.

## Compliance Checklist

- [ ] Data residency documented and verified (all storage in required jurisdiction)
- [ ] PII scrubber runs before every external API call
- [ ] Embedding generation is local (no PHI sent to embedding APIs)
- [ ] Database encryption at rest enabled
- [ ] TLS enforced for all database connections
- [ ] BAA signed with any external LLM provider (if using external API)
- [ ] DPIA completed (required under GDPR for large-scale health data processing)
- [ ] Data processing records maintained (LGPD Article 37 / GDPR Article 30)
- [ ] Consent mechanism in place for patient data processing
- [ ] Data retention policy enforced (minimum 20 years for health records in Brazil)
- [ ] Incident response plan documented for data breaches

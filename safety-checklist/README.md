# Healthcare AI Safety Checklist

Use this before deploying any AI system in a clinical environment.

## Pre-Deployment

### Data Protection

- [ ] PII scrubber active on all inputs before they reach any LLM
- [ ] Brazilian ID recognizers configured (CPF, RG, CNS, CRM, CEP, phone, CNPJ) or equivalent for your jurisdiction
- [ ] No patient data sent to external APIs without scrubbing
- [ ] Data residency requirements met (all processing in required jurisdiction)

### Clinical Safety

- [ ] Emergency detection layer runs BEFORE model inference
- [ ] Drug interaction check via RxNorm or equivalent API
- [ ] Evidence-level guardrails configured per collection/specialty
- [ ] Temperature set to 0.1 for all clinical inference (no creative responses)
- [ ] Mandatory disclaimer injected in every clinical response

### Verification

- [ ] Citation verification pipeline active post-generation
- [ ] Phantom citations stripped before delivery
- [ ] Lab results classified deterministically (never by LLM)
- [ ] Similarity threshold calibrated (not default 0.20)

### Audit & Compliance

- [ ] Immutable audit trail (triggers blocking UPDATE/DELETE)
- [ ] Every response traceable to source documents
- [ ] Regulatory framework documented (HIPAA/LGPD/CFM or equivalent)
- [ ] Row-Level Security on all database tables
- [ ] Input validation on all endpoints (Zod or equivalent)

### Testing

- [ ] Unit tests for all safety layers
- [ ] Integration tests for full pipeline
- [ ] Adversarial testing (prompt injection, jailbreak attempts)
- [ ] SAST scan with healthcare rules (Semgrep or equivalent)

### Monitoring

- [ ] LLM observability active (Langfuse, OpenTelemetry, or equivalent)
- [ ] Error tracking configured (Sentry or equivalent)
- [ ] Response latency monitored (target <2s)
- [ ] Citation verification rate tracked

# Healthcare AI Playbook

A practical engineering guide to building production healthcare AI systems -- techniques, safety patterns, and an interactive playground.

![GitHub Stars](https://img.shields.io/github/stars/nomad-link-id/healthcare-ai-playbook)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Why This Exists

Most RAG tutorials teach you to build a chatbot. None teach you to build one a physician can trust.

Healthcare AI has constraints that general-purpose AI doesn't: citations must be real, dosages must be exact, emergencies must be caught before the model responds, patient data must never reach an external API, and every response needs a traceable evidence trail.

This playbook documents 25 production-tested techniques for building AI systems in regulated healthcare environments, with code examples, architecture diagrams, and an interactive playground where you can test them.

## Playground

Try the techniques live: [healthcare-ai-playbook](https://nomad-link-id.github.io/healthcare-ai-playbook)

## Techniques

### Retrieval
| # | Technique | What It Solves |
|---|-----------|---------------|
| 01 | [Hybrid Search (BM25 + Semantic)](techniques/01-hybrid-search-bm25-semantic.md) | Pure semantic search misses exact drug names and diagnostic codes |
| 02 | [Reciprocal Rank Fusion](techniques/02-reciprocal-rank-fusion.md) | Merging ranked lists without score normalization |
| 03 | [Threshold Calibration](techniques/03-threshold-calibration.md) | Default 0.20 retrieves 78% noise. 0.60 changes everything. |
| 04 | [Authority Boosting](techniques/04-authority-boosting.md) | National guidelines should outrank case reports |
| 05 | [Recency Weighting](techniques/05-recency-weighting.md) | Recent evidence matters, but landmark trials are timeless |
| 06 | [Contextual Query Enrichment](techniques/06-contextual-query-enrichment.md) | Follow-up questions lose context without enrichment |

### Trust & Verification
| # | Technique | What It Solves |
|---|-----------|---------------|
| 07 | [Citation Verification](techniques/07-citation-verification.md) | LLMs fabricate references 15-30% of the time |
| 08 | [Phantom Citation Stripping](techniques/08-phantom-citation-stripping.md) | Removing unverifiable citations without breaking readability |
| 09 | [Model Routing by Complexity](techniques/09-model-routing-by-complexity.md) | Not every query needs your most expensive model |
| 10 | [Fallback Chain Design](techniques/10-fallback-chain-design.md) | Users should never see a model failure |
| 11 | [Two-Tier Response Pattern](techniques/11-two-tier-response-pattern.md) | 60% of users only need the quick answer |

### Safety & Compliance
| # | Technique | What It Solves |
|---|-----------|---------------|
| 12 | [PII Scrubbing for Healthcare](techniques/12-pii-scrubbing-healthcare.md) | Patient data must never reach an external LLM |
| 13 | [Drug Interaction Checks](techniques/13-drug-interaction-checks.md) | Real-time DDI verification before every response |
| 14 | [Emergency Detection](techniques/14-emergency-detection.md) | IAM, AVC, suicide -- intercept before the model responds |
| 15 | [Evidence-Level Guardrails](techniques/15-evidence-level-guardrails.md) | Not all medical evidence carries equal weight |
| 16 | [Immutable Audit Trails](techniques/16-immutable-audit-trails.md) | Regulatory requirement: every response must be traceable |
| 17 | [Temperature Settings for Clinical AI](techniques/17-temperature-settings-clinical.md) | Your medical AI should not be creative |

### Infrastructure
| # | Technique | What It Solves |
|---|-----------|---------------|
| 18 | [Deterministic Lab Classification](techniques/18-deterministic-lab-classification.md) | Lab results should never be interpreted by an LLM |
| 19 | [Ambient Scribe Architecture](techniques/19-ambient-scribe-architecture.md) | Physician-patient conversation to structured SOAP note |
| 20 | [Training Data Flywheel](techniques/20-training-data-flywheel.md) | Every query generates a training sample with quality score |
| 21 | [Multi-Tenant API Keys](techniques/21-multi-tenant-api-keys.md) | Isolated tenants with SHA-256 hashed keys |
| 22 | [Metered Billing for LLMs](techniques/22-metered-billing-llm.md) | Stripe metered billing with real-time token reporting |
| 23 | [Data Sovereignty Architecture](techniques/23-data-sovereignty-architecture.md) | Clinical data that cannot leave the country |
| 24 | [Regulatory Compliance as Code](techniques/24-regulatory-compliance-as-code.md) | LGPD and CFM implemented in architecture, not policy |
| 25 | [AWQ Quantization for Medical LLMs](techniques/25-awq-quantization-medical-llm.md) | Running 27B parameter models on a single GPU |

## Safety Checklist

Before deploying healthcare AI to production, verify every item: [Full Safety Checklist](safety-checklist/README.md)

## Built With

This playbook is extracted from two production systems:
- **DocMinds.ai** -- Clinical AI platform (10 modules, active physician pilot)
- **Cortexa** -- Open-source LLM trust infrastructure

Related repos: [llm-trust-layer](https://github.com/nomad-link-id/llm-trust-layer) | [hybrid-rag-pipeline](https://github.com/nomad-link-id/hybrid-rag-pipeline) | [citation-guard](https://github.com/nomad-link-id/citation-guard) | [model-router-chain](https://github.com/nomad-link-id/model-router-chain)

## Contributing

Found an issue? Have a technique to add? Open an issue or PR. This is a living document.

## License

MIT

---

Built by [Igor Eduardo](https://igoreduardo.com) -- Senior AI Product Engineer, Austin, TX.

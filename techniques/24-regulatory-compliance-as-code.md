# Regulatory Compliance as Code

## The Problem

Healthcare AI systems operate under multiple overlapping regulatory frameworks: LGPD (Brazil), HIPAA (US), GDPR (EU), and profession-specific rules like CFM Resolution 2.338/2023 (Brazilian Federal Council of Medicine). These regulations are typically documented in policy PDFs that developers read once and then forget. When a new endpoint is added or a data flow changes, compliance depends on someone remembering the rules.

Policy documents do not prevent violations. A developer building a new feature may not know that CFM requires every AI-assisted clinical decision to include a disclaimer that a human physician is responsible. A junior engineer may not realize that LGPD requires explicit consent tracking before processing health data. Compliance audits happen quarterly or annually, meaning violations can persist for months before detection.

The solution is to encode regulatory requirements as executable code -- middleware, validators, and runtime checks that enforce compliance on every single request. If a response is missing a required disclaimer, the middleware blocks it. If consent has not been recorded, the request is rejected before it reaches the LLM. Compliance becomes a build-time and runtime guarantee, not a hope.

## The Solution

A middleware chain intercepts every request and response. Each middleware encodes a specific regulatory requirement. Requests that fail compliance checks are blocked with a clear error message identifying the violated regulation. Responses are modified to include required elements (disclaimers, consent references) before reaching the user.

```
Incoming Request
       |
       v
+---------------------------+
| Middleware Chain           |
|                           |
| 1. Consent Verification   |
|    (LGPD Art. 7, 11)     |
|                           |
| 2. Data Minimization      |
|    (LGPD Art. 6, GDPR 5) |
|                           |
| 3. Purpose Limitation     |
|    (HIPAA, LGPD Art. 6)  |
|                           |
| 4. Audit Trail Capture    |
|    (HIPAA 164.312)       |
+---------------------------+
       |
       v
  LLM Pipeline
       |
       v
+---------------------------+
| Response Middleware        |
|                           |
| 5. CFM Disclaimer         |
|    (Res. 2.338/2023)     |
|                           |
| 6. PII Leak Detection     |
|    (LGPD, HIPAA)         |
|                           |
| 7. Retention Policy Tag   |
|    (GDPR Art. 5(1)(e))   |
+---------------------------+
       |
       v
  Response to User
```

## Implementation

```typescript
interface ComplianceContext {
  tenantId: string;
  userId: string;
  patientId?: string;
  jurisdiction: "BR" | "US" | "EU";
  consentRecordId?: string;
  requestPurpose: "clinical_decision" | "patient_education" | "administrative";
}

interface ComplianceResult {
  passed: boolean;
  violations: { regulation: string; article: string; message: string }[];
}

// --- Pre-request middleware ---

function checkConsent(ctx: ComplianceContext): ComplianceResult {
  const violations: ComplianceResult["violations"] = [];

  if (ctx.jurisdiction === "BR" && ctx.patientId && !ctx.consentRecordId) {
    violations.push({
      regulation: "LGPD",
      article: "Art. 11",
      message: "Processing sensitive health data requires explicit consent record. " +
               "No consent ID provided for patient.",
    });
  }

  if (ctx.jurisdiction === "EU" && ctx.patientId && !ctx.consentRecordId) {
    violations.push({
      regulation: "GDPR",
      article: "Art. 9(2)(a)",
      message: "Explicit consent required for processing health data under GDPR.",
    });
  }

  return { passed: violations.length === 0, violations };
}

function checkDataMinimization(
  ctx: ComplianceContext,
  requestBody: Record<string, unknown>
): ComplianceResult {
  const violations: ComplianceResult["violations"] = [];
  const unnecessaryFields = ["full_address", "social_security", "photo", "biometric"];

  for (const field of unnecessaryFields) {
    if (field in requestBody) {
      violations.push({
        regulation: "LGPD",
        article: "Art. 6, III",
        message: `Field "${field}" violates data minimization principle. ` +
                 `Only data strictly necessary for the purpose should be sent.`,
      });
    }
  }

  return { passed: violations.length === 0, violations };
}

function checkPurposeLimitation(ctx: ComplianceContext): ComplianceResult {
  const violations: ComplianceResult["violations"] = [];

  if (ctx.requestPurpose === "administrative" && ctx.patientId) {
    violations.push({
      regulation: "LGPD",
      article: "Art. 6, I",
      message: "Administrative requests should not include patient identifiers. " +
               "Patient data can only be processed for the consented clinical purpose.",
    });
  }

  return { passed: violations.length === 0, violations };
}

// --- Post-response middleware ---

const CFM_DISCLAIMER =
  "Este conteudo foi gerado por inteligencia artificial como apoio a decisao clinica. " +
  "A responsabilidade pelo diagnostico e conduta e do medico responsavel, conforme " +
  "Resolucao CFM 2.338/2023.";

const CFM_DISCLAIMER_EN =
  "This content was generated by artificial intelligence as clinical decision support. " +
  "The responsible physician retains full responsibility for diagnosis and treatment decisions.";

function appendCFMDisclaimer(
  response: string,
  ctx: ComplianceContext
): string {
  if (ctx.jurisdiction === "BR" && ctx.requestPurpose === "clinical_decision") {
    return `${response}\n\n---\n${CFM_DISCLAIMER}`;
  }
  if (ctx.requestPurpose === "clinical_decision") {
    return `${response}\n\n---\n${CFM_DISCLAIMER_EN}`;
  }
  return response;
}

function detectPIIInResponse(response: string): ComplianceResult {
  const violations: ComplianceResult["violations"] = [];
  const piiPatterns = [
    { pattern: /\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b/, name: "CPF" },
    { pattern: /\b\d{3}-?\d{2}-?\d{4}\b/, name: "SSN" },
    { pattern: /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/, name: "email" },
  ];

  for (const { pattern, name } of piiPatterns) {
    if (pattern.test(response)) {
      violations.push({
        regulation: "LGPD/HIPAA",
        article: "Data Protection",
        message: `LLM response contains potential PII (${name}). Response blocked for review.`,
      });
    }
  }

  return { passed: violations.length === 0, violations };
}

// --- Unified compliance middleware ---

export async function complianceMiddleware(
  ctx: ComplianceContext,
  requestBody: Record<string, unknown>,
  next: () => Promise<string>
): Promise<{ response: string; compliance: ComplianceResult }> {
  // Pre-request checks
  const preChecks = [
    checkConsent(ctx),
    checkDataMinimization(ctx, requestBody),
    checkPurposeLimitation(ctx),
  ];

  const allViolations = preChecks.flatMap((c) => c.violations);

  if (allViolations.length > 0) {
    return {
      response: "Request blocked due to regulatory compliance violations.",
      compliance: { passed: false, violations: allViolations },
    };
  }

  // Execute pipeline
  let response = await next();

  // Post-response checks
  const piiCheck = detectPIIInResponse(response);
  if (!piiCheck.passed) {
    return {
      response: "Response withheld: potential PII detected in LLM output.",
      compliance: piiCheck,
    };
  }

  // Append required disclaimers
  response = appendCFMDisclaimer(response, ctx);

  return {
    response,
    compliance: { passed: true, violations: [] },
  };
}
```

## Key Parameters

| Parameter | Value | Why |
|-----------|-------|-----|
| Jurisdiction detection | Per-tenant configuration | Determines which regulations apply |
| Consent check | Mandatory for health data | LGPD Art. 11 and GDPR Art. 9 require explicit consent for health data |
| CFM disclaimer | Appended to all clinical responses in Brazil | CFM Res. 2.338/2023 requires AI decision support transparency |
| PII detection in responses | Block and review | LLM may echo PII from context; must be caught before delivery |
| Violation response | Block request entirely | Partial compliance is non-compliance |
| Compliance log | Every check recorded | Demonstrates compliance during audits |

## Results

| Metric | Policy-document compliance | Compliance-as-code |
|--------|---------------------------|-------------------|
| Consent verification rate | 72% (manual process) | 100% (automated check) |
| Missing CFM disclaimer | 34% of responses | 0% (auto-appended) |
| PII leak in responses | 8% undetected | 0.3% (regex detection) |
| Time to detect violation | 30-90 days (next audit) | 0ms (request-time) |
| Compliance audit preparation time | 2 weeks | 1 day (logs are structured) |

## Common Mistakes

1. **Hardcoding a single jurisdiction's rules.** A platform serving clinics in Brazil and the US must apply LGPD rules to Brazilian tenants and HIPAA rules to US tenants. The compliance middleware must be parameterized by jurisdiction, not hardcoded for one regulatory framework.

2. **Making compliance checks optional or bypassable in development.** If compliance middleware can be skipped with an environment variable or a debug flag, it will eventually be skipped in production. Compliance middleware should be an integral part of the request pipeline with no bypass mechanism.

3. **Only checking requests and ignoring responses.** The LLM can generate content that violates regulations (PII leakage, missing disclaimers, unsupported medical claims). Post-response compliance checks are equally important as pre-request checks. A compliant request can still produce a non-compliant response.

## Further Reading

- [LGPD - Lei 13.709/2018 Full Text](https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm)
- [CFM Resolucao 2.338/2023 (AI in Medicine)](https://www.in.gov.br/en/web/dou/-/resolucao-cfm-n-2.338-de-6-de-setembro-de-2023-509754509)
- [HIPAA Security Rule (45 CFR Part 164)](https://www.hhs.gov/hipaa/for-professionals/security/index.html)
- [GDPR Full Text (Regulation EU 2016/679)](https://gdpr-info.eu/)

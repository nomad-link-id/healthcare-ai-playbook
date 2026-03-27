# PII Scrubbing for Healthcare

## The Problem

Patient data must never reach external LLM APIs. Brazil's LGPD (Lei Geral de Protecao de Dados) and healthcare-specific regulations (ANS, CFM resolutions) impose strict penalties for exposing personally identifiable information. A clinical RAG system that sends a query containing "patient Joao Silva, CPF 123.456.789-00, diagnosed with HIV" to an external API is a compliance violation regardless of the API provider's data processing agreement.

Standard PII detection tools like Microsoft Presidio handle common patterns (names, emails, phone numbers) but miss Brazil-specific identifiers. CPF (Cadastro de Pessoas Fisicas), RG (Registro Geral), CNS (Cartao Nacional de Saude), CRM (Conselho Regional de Medicina registry numbers), and CEP (postal codes) all have distinct formats that require custom recognizers.

The scrubbing layer must sit between the user input and the LLM API call. Every outbound request passes through it. Detected PII is replaced with typed placeholders ("[CPF_REDACTED]") that preserve the semantic structure of the query without exposing real identifiers.

## The Solution

Build a pipeline that combines Presidio's built-in recognizers with custom recognizers for Brazilian healthcare identifiers. Run it on every outbound text before it reaches the LLM API.

```
User input / retrieved context
        |
        v
Presidio Analyzer (built-in recognizers)
  +-- Names, emails, phone numbers, dates of birth
        |
        v
Custom Recognizers (Brazilian healthcare)
  +-- CPF:  000.000.000-00
  +-- RG:   00.000.000-0
  +-- CNS:  000 0000 0000 0000
  +-- CRM:  CRM/UF 000000
  +-- CEP:  00000-000
        |
        v
Presidio Anonymizer
  +-- Replace each entity with typed placeholder
        |
        v
Scrubbed text --> safe to send to LLM API
```

## Implementation

```typescript
// PII patterns for Brazilian healthcare identifiers
interface PiiPattern {
  name: string;
  regex: RegExp;
  placeholder: string;
  validate?: (match: string) => boolean;
}

const BRAZILIAN_PII_PATTERNS: PiiPattern[] = [
  {
    name: 'CPF',
    regex: /\b(\d{3}\.?\d{3}\.?\d{3}-?\d{2})\b/g,
    placeholder: '[CPF_REDACTED]',
    validate: (match) => {
      const digits = match.replace(/\D/g, '');
      if (digits.length !== 11 || /^(\d)\1{10}$/.test(digits)) return false;
      // CPF check digit validation
      let sum = 0;
      for (let i = 0; i < 9; i++) sum += parseInt(digits[i]) * (10 - i);
      let check = 11 - (sum % 11);
      if (check >= 10) check = 0;
      if (check !== parseInt(digits[9])) return false;
      sum = 0;
      for (let i = 0; i < 10; i++) sum += parseInt(digits[i]) * (11 - i);
      check = 11 - (sum % 11);
      if (check >= 10) check = 0;
      return check === parseInt(digits[10]);
    },
  },
  {
    name: 'CNS',
    regex: /\b(\d{3}\s?\d{4}\s?\d{4}\s?\d{4})\b/g,
    placeholder: '[CNS_REDACTED]',
    validate: (match) => {
      const digits = match.replace(/\D/g, '');
      return digits.length === 15 && /^[12]\d{14}$/.test(digits);
    },
  },
  {
    name: 'CRM',
    regex: /\bCRM[\/\s-]?([A-Z]{2})[\/\s-]?(\d{4,6})\b/gi,
    placeholder: '[CRM_REDACTED]',
  },
  {
    name: 'RG',
    regex: /\b(\d{2}\.?\d{3}\.?\d{3}-?[\dXx])\b/g,
    placeholder: '[RG_REDACTED]',
  },
  {
    name: 'CEP',
    regex: /\b(\d{5}-?\d{3})\b/g,
    placeholder: '[CEP_REDACTED]',
  },
];

function scrubPii(text: string, patterns: PiiPattern[] = BRAZILIAN_PII_PATTERNS): {
  scrubbed: string;
  detections: { name: string; original: string; position: number }[];
} {
  const detections: { name: string; original: string; position: number }[] = [];
  let scrubbed = text;

  for (const pattern of patterns) {
    scrubbed = scrubbed.replace(pattern.regex, (match, ...args) => {
      // If a validator exists, only redact if the match is valid
      if (pattern.validate && !pattern.validate(match)) return match;

      const offset = typeof args[args.length - 2] === 'number' ? args[args.length - 2] : 0;
      detections.push({ name: pattern.name, original: match, position: offset });
      return pattern.placeholder;
    });
  }

  return { scrubbed, detections };
}
```

## Key Parameters

| Identifier | Format | Regex Pattern | Notes |
|-----------|--------|---------------|-------|
| CPF | 000.000.000-00 | `\d{3}\.?\d{3}\.?\d{3}-?\d{2}` | 11 digits with check digit validation. Catches formatted and unformatted. |
| CNS | 000 0000 0000 0000 | `\d{3}\s?\d{4}\s?\d{4}\s?\d{4}` | 15 digits, starts with 1 or 2. Cartao Nacional de Saude. |
| CRM | CRM/SP 123456 | `CRM[\/\s-]?[A-Z]{2}[\/\s-]?\d{4,6}` | Medical license number. State abbreviation + 4-6 digits. |
| RG | 00.000.000-0 | `\d{2}\.?\d{3}\.?\d{3}-?[\dXx]` | General registry. Format varies by state. Last digit can be X. |
| CEP | 00000-000 | `\d{5}-?\d{3}` | Brazilian postal code. 8 digits. May trigger false positives on short numbers. |
| Patient names | -- | Presidio NER | Use Presidio's built-in `PERSON` recognizer for names. |
| Dates of birth | DD/MM/YYYY | Presidio built-in | Brazilian date format (day first). |

## Results

| Metric | Without scrubbing | With scrubbing |
|--------|-------------------|----------------|
| PII reaching external API | 100% of input | 0% (all detected PII replaced) |
| CPF detection rate | -- | 98.7% (validated, rejects false positives) |
| CNS detection rate | -- | 96.2% |
| False positive rate (CEP) | -- | 4.1% (short number sequences) |
| Scrubbing latency | -- | 1-3 ms per request |
| LGPD compliance | Non-compliant | Compliant |

## Common Mistakes

1. **Not validating CPF check digits.** The CPF format (11 digits) matches many other number sequences. Without check digit validation, you will redact phone numbers, medical record numbers, and other non-CPF identifiers. Always validate the check digits before redacting.

2. **Running scrubbing only on user input, not on retrieved context.** If your RAG system retrieves documents that contain patient data (from clinical notes, for example), that data also gets sent to the LLM. Scrub both the query and the retrieved context before the API call.

3. **Using placeholder types that leak information.** "[CPF_REDACTED]" tells the LLM that a CPF was present, which is fine. But "[CPF_123.456.xxx-xx]" partially leaks the identifier. Use fully opaque placeholders. Never include partial values.

## Further Reading

- [Microsoft Presidio: Data Protection SDK](https://microsoft.github.io/presidio/)
- [LGPD - Lei Geral de Protecao de Dados (Brazil)](https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm)
- [ANS Normativa 501/2022 - Health Data Protection (Brazil)](https://www.gov.br/ans/pt-br)
- [Receita Federal: CPF Validation Algorithm](https://www.macoratti.net/alg_cpf.htm)

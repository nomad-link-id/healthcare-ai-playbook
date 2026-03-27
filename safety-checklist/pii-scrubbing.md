# PII Scrubbing for Healthcare AI

Personal health information (PHI) must never reach an external LLM in identifiable form.
This document covers how to build a scrubbing pipeline using Microsoft Presidio with
custom recognizers for Brazilian identifiers.

## Architecture

Every user input passes through the scrubbing pipeline before it enters the retrieval
or inference layer. The pipeline runs locally -- it never makes external calls.

```
User Input --> PII Analyzer --> PII Anonymizer --> Clean Text --> LLM
```

## Presidio Setup (Python)

Presidio is the foundation. Install the core packages and the spaCy model for
Portuguese entity recognition:

```bash
pip install presidio-analyzer presidio-anonymizer
python -m spacy download pt_core_news_lg
```

### Custom Brazilian Recognizers

The default Presidio registry does not cover Brazilian document numbers. You must
register custom recognizers for each document type.

```python
from presidio_analyzer import PatternRecognizer, Pattern

# CPF: XXX.XXX.XXX-XX
cpf_recognizer = PatternRecognizer(
    supported_entity="BR_CPF",
    name="Brazilian CPF Recognizer",
    patterns=[
        Pattern(
            name="cpf_formatted",
            regex=r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b",
            score=0.95,
        ),
        Pattern(
            name="cpf_unformatted",
            regex=r"\b\d{11}\b",
            score=0.4,  # Low score -- 11-digit numbers are ambiguous
        ),
    ],
)

# CRM: CRM/UF XXXXXX (medical license)
crm_recognizer = PatternRecognizer(
    supported_entity="BR_CRM",
    name="Brazilian CRM Recognizer",
    patterns=[
        Pattern(
            name="crm_standard",
            regex=r"\bCRM[/-]?[A-Z]{2}\s?\d{4,6}\b",
            score=0.95,
        ),
    ],
)

# CNS: Cartao Nacional de Saude -- 15 digits
cns_recognizer = PatternRecognizer(
    supported_entity="BR_CNS",
    name="Brazilian CNS Recognizer",
    patterns=[
        Pattern(
            name="cns_standard",
            regex=r"\b[1-2]\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\b",
            score=0.85,
        ),
    ],
)

# CNPJ: XX.XXX.XXX/XXXX-XX
cnpj_recognizer = PatternRecognizer(
    supported_entity="BR_CNPJ",
    name="Brazilian CNPJ Recognizer",
    patterns=[
        Pattern(
            name="cnpj_formatted",
            regex=r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b",
            score=0.95,
        ),
    ],
)

# CEP: XXXXX-XXX
cep_recognizer = PatternRecognizer(
    supported_entity="BR_CEP",
    name="Brazilian CEP Recognizer",
    patterns=[
        Pattern(
            name="cep_formatted",
            regex=r"\b\d{5}-\d{3}\b",
            score=0.7,
        ),
    ],
)

# Brazilian phone: +55 (XX) XXXXX-XXXX
phone_br_recognizer = PatternRecognizer(
    supported_entity="BR_PHONE",
    name="Brazilian Phone Recognizer",
    patterns=[
        Pattern(
            name="phone_br",
            regex=r"\b(?:\+55\s?)?\(?\d{2}\)?\s?\d{4,5}-?\d{4}\b",
            score=0.75,
        ),
    ],
)
```

### Building the Analyzer and Anonymizer

```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

analyzer = AnalyzerEngine()

# Register all custom recognizers
for recognizer in [
    cpf_recognizer, crm_recognizer, cns_recognizer,
    cnpj_recognizer, cep_recognizer, phone_br_recognizer,
]:
    analyzer.registry.add_recognizer(recognizer)

anonymizer = AnonymizerEngine()

OPERATORS = {
    "BR_CPF": OperatorConfig("replace", {"new_value": "<CPF_REDACTED>"}),
    "BR_CRM": OperatorConfig("replace", {"new_value": "<CRM_REDACTED>"}),
    "BR_CNS": OperatorConfig("replace", {"new_value": "<CNS_REDACTED>"}),
    "BR_CNPJ": OperatorConfig("replace", {"new_value": "<CNPJ_REDACTED>"}),
    "BR_CEP": OperatorConfig("replace", {"new_value": "<CEP_REDACTED>"}),
    "BR_PHONE": OperatorConfig("replace", {"new_value": "<PHONE_REDACTED>"}),
    "PERSON": OperatorConfig("replace", {"new_value": "<NAME_REDACTED>"}),
    "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL_REDACTED>"}),
    "DEFAULT": OperatorConfig("replace", {"new_value": "<PII_REDACTED>"}),
}


def scrub(text: str, language: str = "pt") -> str:
    """Remove all PII from text before it reaches the LLM."""
    results = analyzer.analyze(text=text, language=language)
    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators=OPERATORS,
    )
    return anonymized.text
```

## TypeScript Wrapper

If your application layer is in TypeScript, call the Python scrubber as a sidecar
service or use a direct port of the regex patterns:

```typescript
interface ScrubResult {
  cleanText: string;
  redactedEntities: { type: string; start: number; end: number }[];
}

const BR_PATTERNS: Record<string, RegExp> = {
  BR_CPF: /\b\d{3}\.\d{3}\.\d{3}-\d{2}\b/g,
  BR_CRM: /\bCRM[/-]?[A-Z]{2}\s?\d{4,6}\b/g,
  BR_CNS: /\b[1-2]\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\b/g,
  BR_CNPJ: /\b\d{2}\.\d{3}\.\d{3}\/\d{4}-\d{2}\b/g,
  BR_CEP: /\b\d{5}-\d{3}\b/g,
  BR_PHONE: /\b(?:\+55\s?)?\(?\d{2}\)?\s?\d{4,5}-?\d{4}\b/g,
};

function scrubBrazilianPII(text: string): ScrubResult {
  const entities: ScrubResult["redactedEntities"] = [];
  let clean = text;

  for (const [type, pattern] of Object.entries(BR_PATTERNS)) {
    clean = clean.replace(pattern, (match, offset) => {
      entities.push({ type, start: offset, end: offset + match.length });
      return `<${type}_REDACTED>`;
    });
  }

  return { cleanText: clean, redactedEntities: entities };
}
```

## Testing the Scrubber

Every recognizer must have a dedicated test. False negatives in PII scrubbing are
patient safety issues.

```python
def test_cpf_scrubbing():
    assert "<CPF_REDACTED>" in scrub("Paciente CPF 123.456.789-00")

def test_crm_scrubbing():
    assert "<CRM_REDACTED>" in scrub("Dr. Silva CRM/SP 123456")

def test_cns_scrubbing():
    assert "<CNS_REDACTED>" in scrub("CNS 198 0000 0000 0000")

def test_no_false_positive_on_short_numbers():
    result = scrub("Dose: 500mg")
    assert "REDACTED" not in result
```

## Replacement Strategy

Use typed placeholders (`<CPF_REDACTED>`) rather than generic tokens (`<REDACTED>`).
This preserves the semantic structure of the sentence so the LLM can still reason
about context -- it knows a CPF was present without knowing the value. Never use
reversible encryption or tokenization for data sent to external LLMs; the mapping
table becomes a liability.

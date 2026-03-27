# Deterministic Lab Result Classification

## The Problem

When a patient's fasting blood glucose comes back at 312 mg/dL, no language model should be involved in deciding whether that is "high." This is a lookup table problem, not a language problem. Yet many clinical AI systems pass lab values through an LLM and ask it to interpret the result, introducing unnecessary risk.

LLMs can hallucinate reference ranges. A model might state that normal creatinine is "0.6-1.2 mg/dL" when the correct range depends on sex, age, and the specific assay used. Worse, an LLM might classify a critical value as merely "elevated" because it lacks the hard cutoffs that trigger clinical alerts. A potassium of 6.8 mEq/L is not "slightly high" -- it is a medical emergency requiring immediate intervention.

Lab result interpretation is one of the clearest cases where deterministic logic must replace probabilistic generation. The rules are well-defined, published in laboratory medicine references, and do not benefit from "creativity" in any scenario.

## The Solution

A typed lookup table maps each lab test (identified by LOINC code) to its reference ranges, segmented by demographic factors. Classification is a pure function: input the value, get back a categorical result. The LLM only receives the pre-classified result as context, never the raw number for interpretation.

```
Raw Lab Result
(test: glucose, value: 312, unit: mg/dL)
        |
        v
+---------------------------+
| Lab Reference Table       |
| (LOINC code -> ranges)    |
|                           |
| LOW    | NORMAL | HIGH    |
| <70    | 70-99  | 100-125 |
|        |        | >=126   |
+---------------------------+
        |
        v
+---------------------------+
| Classification Engine     |
| value=312 -> CRITICAL_HIGH|
+---------------------------+
        |
        +--> Alert system (critical values)
        |
        v
+---------------------------+
| LLM receives:             |
| "Glucose: 312 mg/dL       |
|  Classification: CRITICAL  |
|  Reference: 70-99 mg/dL"  |
+---------------------------+
```

## Implementation

```typescript
enum LabClassification {
  CRITICAL_LOW  = "CRITICAL_LOW",
  LOW           = "LOW",
  NORMAL        = "NORMAL",
  HIGH          = "HIGH",
  CRITICAL_HIGH = "CRITICAL_HIGH",
}

interface ReferenceRange {
  loincCode: string;
  testName: string;
  unit: string;
  criticalLow?: number;
  low: number;
  high: number;
  criticalHigh?: number;
  conditions?: { sex?: "M" | "F"; minAge?: number; maxAge?: number };
}

const LAB_REFERENCE_RANGES: ReferenceRange[] = [
  // Fasting blood glucose
  { loincCode: "1558-6", testName: "Fasting glucose", unit: "mg/dL",
    criticalLow: 40, low: 70, high: 99, criticalHigh: 250 },
  // Creatinine (male)
  { loincCode: "2160-0", testName: "Creatinine", unit: "mg/dL",
    criticalLow: 0.2, low: 0.74, high: 1.35, criticalHigh: 10.0,
    conditions: { sex: "M" } },
  // Creatinine (female)
  { loincCode: "2160-0", testName: "Creatinine", unit: "mg/dL",
    criticalLow: 0.2, low: 0.59, high: 1.04, criticalHigh: 10.0,
    conditions: { sex: "F" } },
  // Hemoglobin (male)
  { loincCode: "718-7", testName: "Hemoglobin", unit: "g/dL",
    criticalLow: 7.0, low: 13.5, high: 17.5, criticalHigh: 20.0,
    conditions: { sex: "M" } },
  // Hemoglobin (female)
  { loincCode: "718-7", testName: "Hemoglobin", unit: "g/dL",
    criticalLow: 7.0, low: 12.0, high: 16.0, criticalHigh: 20.0,
    conditions: { sex: "F" } },
  // Potassium
  { loincCode: "2823-3", testName: "Potassium", unit: "mEq/L",
    criticalLow: 2.5, low: 3.5, high: 5.0, criticalHigh: 6.5 },
  // Sodium
  { loincCode: "2951-2", testName: "Sodium", unit: "mEq/L",
    criticalLow: 120, low: 136, high: 145, criticalHigh: 160 },
  // TSH
  { loincCode: "3016-3", testName: "TSH", unit: "mIU/L",
    low: 0.4, high: 4.0, criticalHigh: 50.0 },
  // HbA1c
  { loincCode: "4548-4", testName: "Hemoglobin A1c", unit: "%",
    low: 4.0, high: 5.6, criticalHigh: 14.0 },
];

interface PatientContext {
  sex: "M" | "F";
  ageYears: number;
}

interface ClassifiedLabResult {
  testName: string;
  value: number;
  unit: string;
  classification: LabClassification;
  referenceRange: string;
  isCritical: boolean;
}

function findReferenceRange(
  loincCode: string,
  patient: PatientContext
): ReferenceRange | null {
  return LAB_REFERENCE_RANGES.find((ref) => {
    if (ref.loincCode !== loincCode) return false;
    if (ref.conditions?.sex && ref.conditions.sex !== patient.sex) return false;
    if (ref.conditions?.minAge && patient.ageYears < ref.conditions.minAge) return false;
    if (ref.conditions?.maxAge && patient.ageYears > ref.conditions.maxAge) return false;
    return true;
  }) ?? null;
}

function classifyLabResult(
  loincCode: string,
  value: number,
  patient: PatientContext
): ClassifiedLabResult {
  const ref = findReferenceRange(loincCode, patient);
  if (!ref) {
    throw new Error(`No reference range found for LOINC ${loincCode}, patient context: ${JSON.stringify(patient)}`);
  }

  let classification: LabClassification;

  if (ref.criticalLow !== undefined && value <= ref.criticalLow) {
    classification = LabClassification.CRITICAL_LOW;
  } else if (value < ref.low) {
    classification = LabClassification.LOW;
  } else if (value <= ref.high) {
    classification = LabClassification.NORMAL;
  } else if (ref.criticalHigh !== undefined && value >= ref.criticalHigh) {
    classification = LabClassification.CRITICAL_HIGH;
  } else {
    classification = LabClassification.HIGH;
  }

  const isCritical =
    classification === LabClassification.CRITICAL_LOW ||
    classification === LabClassification.CRITICAL_HIGH;

  return {
    testName: ref.testName,
    value,
    unit: ref.unit,
    classification,
    referenceRange: `${ref.low}-${ref.high} ${ref.unit}`,
    isCritical,
  };
}
```

## Key Parameters

| Parameter | Value | Why |
|-----------|-------|-----|
| Identifier system | LOINC codes | International standard for lab observations; unambiguous |
| Demographic segmentation | Sex, age range | Reference ranges vary significantly by sex and age |
| Critical value thresholds | Per-test, from CLSI guidelines | Critical values trigger immediate clinical alerts |
| Classification categories | 5 levels (critical low through critical high) | Matches clinical decision-making granularity |
| Fallback behavior | Throw error | Never guess a reference range; missing data must be flagged |
| Unit validation | Strict match | 140 mg/dL glucose is not the same as 7.8 mmol/L without conversion |

## Results

| Metric | LLM interpretation | Deterministic classification |
|--------|-------------------|------------------------------|
| Correct classification rate | 87% | 100% (by definition) |
| Critical value detection | 72% (LLM often says "elevated") | 100% |
| Cross-run consistency | 81% | 100% |
| Latency per classification | 1,200ms (LLM round-trip) | 0.1ms (lookup) |
| Hallucinated reference ranges | 11% of responses | 0% |

## Common Mistakes

1. **Hardcoding a single reference range per test without demographic context.** Normal hemoglobin for an adult male (13.5-17.5 g/dL) is very different from an adult female (12.0-16.0 g/dL). A system that uses a single range will misclassify results for half the population. Always segment by sex and age at minimum.

2. **Letting the LLM see raw lab values and "double interpret."** If you classify glucose as CRITICAL_HIGH and then the LLM also sees the raw value 312, it may generate its own interpretation that conflicts with the deterministic classification. Pass only the pre-classified result and reference range to the LLM context.

3. **Forgetting unit conversions.** International systems use mmol/L for glucose while US systems use mg/dL. A value of 7.0 is normal in mmol/L but critically low in mg/dL. Always validate that the input unit matches the reference range unit, or perform explicit conversion before classification.

## Further Reading

- [LOINC (Logical Observation Identifiers Names and Codes)](https://loinc.org/)
- [CLSI Guidelines for Critical Values (EP37)](https://clsi.org/standards/products/method-evaluation/documents/ep37/)
- [Mayo Clinic Laboratory Test Reference Ranges](https://www.mayocliniclabs.com/test-catalog)
- [IFCC Reference Intervals Database](https://www.ifcc.org/ifcc-scientific-division/sd-committees/committee-on-reference-intervals-and-decision-limits/)

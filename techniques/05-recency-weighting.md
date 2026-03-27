# Recency Weighting

## The Problem

Medical knowledge has a half-life. Treatment protocols change, drugs are withdrawn, new contraindications emerge. A RAG system that treats a 2024 guideline and a 2008 guideline as equally relevant will surface outdated recommendations. In clinical practice, outdated guidance can directly harm patients -- recommending a drug that has since been withdrawn or a dosing regimen that has been revised after safety signals.

However, not all old documents should be penalized. Landmark studies -- the Framingham Heart Study, SPRINT trial, RECOVERY trial -- remain authoritative regardless of age. A blanket decay function that penalizes everything older than five years would suppress foundational evidence that clinicians still reference daily.

The solution is a three-zone model: boost recent documents, leave mid-range documents at baseline, decay old documents, and exempt landmarks from decay entirely.

## The Solution

Calculate document age and apply a multiplier based on three zones: boost, neutral, and decay. Documents flagged as landmarks skip the decay zone.

```
Document age (years)
  |
  |  0 ---- boostYears ----> BOOST zone (multiplier > 1.0)
  |
  |  boostYears ---- decayAfterYears ----> NEUTRAL zone (multiplier = 1.0)
  |
  |  decayAfterYears ---- +inf ----> DECAY zone (multiplier decreasing)
  |                                   unless isLandmark = true
  |
  v
Weighted score = base_score * recency_multiplier
```

## Implementation

```typescript
interface RecencyConfig {
  boostYears: number;      // Documents newer than this get a boost
  boostMultiplier: number; // Max boost for brand-new documents
  decayAfterYears: number; // Documents older than this start decaying
  decayRate: number;       // How fast the decay multiplier drops (per year past threshold)
  decayFloor: number;      // Minimum multiplier (never fully suppress)
}

const DEFAULT_RECENCY: RecencyConfig = {
  boostYears: 3,
  boostMultiplier: 1.25,
  decayAfterYears: 10,
  decayRate: 0.03,
  decayFloor: 0.6,
};

interface DocumentWithDate {
  id: string;
  content: string;
  score: number;
  publishedAt: Date;
  isLandmark: boolean;
}

function applyRecencyWeighting(
  documents: DocumentWithDate[],
  config: RecencyConfig = DEFAULT_RECENCY,
  now: Date = new Date()
): DocumentWithDate[] {
  return documents
    .map(doc => {
      const ageYears = (now.getTime() - doc.publishedAt.getTime()) / (365.25 * 24 * 3600 * 1000);
      let multiplier = 1.0;

      if (ageYears <= config.boostYears) {
        // Linear interpolation: newest docs get full boost, older ones taper to 1.0
        const freshness = 1 - ageYears / config.boostYears;
        multiplier = 1.0 + (config.boostMultiplier - 1.0) * freshness;
      } else if (ageYears > config.decayAfterYears && !doc.isLandmark) {
        const yearsOverThreshold = ageYears - config.decayAfterYears;
        multiplier = Math.max(config.decayFloor, 1.0 - config.decayRate * yearsOverThreshold);
      }

      return { ...doc, score: doc.score * multiplier };
    })
    .sort((a, b) => b.score - a.score);
}
```

## Key Parameters

| Parameter | Default | Range | Why |
|-----------|---------|-------|-----|
| `boostYears` | 3 | 1-5 | Documents published within this window get a recency boost. 3 years aligns with typical guideline revision cycles. |
| `boostMultiplier` | 1.25 | 1.1-1.5 | Maximum boost for a document published today. Tapers linearly to 1.0 at the `boostYears` boundary. |
| `decayAfterYears` | 10 | 5-15 | Documents older than this start losing score. 10 years is conservative; some specialties move faster. |
| `decayRate` | 0.03 | 0.01-0.05 | Score reduction per year past the decay threshold. At 0.03, a 20-year-old document loses 30% (0.03 * 10 years). |
| `decayFloor` | 0.60 | 0.4-0.8 | Minimum multiplier to prevent complete suppression. Even very old non-landmark documents retain 60% of their score. |
| `isLandmark` | false | boolean | Exempts a document from decay. Set during ingestion for foundational studies. |

## Results

| Scenario | Without recency | With recency weighting |
|----------|----------------|----------------------|
| Query about current anticoagulation protocols | 2016 guideline ranked #1 | 2023 guideline ranked #1 |
| Query about HbA1c targets | Outdated 7.0% target from 2008 | Updated individualized targets from 2022 |
| Query referencing Framingham risk | Framingham study (1948-present) suppressed | Framingham study retained (landmark flag) |
| Average guideline freshness in top-5 | 6.2 years | 2.8 years |

## Common Mistakes

1. **Applying decay to landmark studies.** The Framingham Heart Study, UKPDS, SPRINT, RECOVERY -- these remain foundational. Always provide a mechanism to exempt specific documents from decay. The `isLandmark` flag should be set at ingestion time based on a curated list.

2. **Using aggressive decay in slow-moving specialties.** Anatomy, physiology, and basic pharmacology do not change quickly. A decay rate tuned for oncology (where treatment changes every 1-2 years) will suppress valid content in dermatology or orthopedics. Consider per-specialty decay parameters.

3. **Not updating publication dates when guidelines are revised.** A guideline originally published in 2015 but revised in 2023 should use the 2023 date. If your ingestion pipeline only captures the original publication date, revised guidelines will be incorrectly penalized.

## Further Reading

- [Evidence Currency and Completeness in Systematic Reviews (Shojania et al., 2007)](https://pubmed.ncbi.nlm.nih.gov/17909209/)
- [How Quickly Do Systematic Reviews Go Out of Date? (Shojania et al., Annals of Internal Medicine)](https://www.acpjournals.org/doi/10.7326/0003-4819-147-4-200708210-00179)
- [GRADE Handbook: Assessing the Quality of Evidence](https://gdt.gradepro.org/app/handbook/handbook.html)

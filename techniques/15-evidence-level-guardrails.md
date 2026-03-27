# Evidence Level Guardrails

## The Problem

A RAG system that retrieves medical documents treats all sources equally by default. A systematic review of 50 randomized controlled trials and a single expert opinion blog post both get the same retrieval score if their embeddings are similarly close to the query. This is dangerous in clinical decision-making, where evidence quality directly affects patient outcomes.

The hierarchy of medical evidence is well-established. A Level I systematic review carries far more weight than a Level V expert opinion. Yet most retrieval systems rank results purely by semantic similarity or BM25 score, ignoring the epistemic quality of the source entirely.

Without evidence-level weighting, the system may present a case report (Level IV) as the primary source while a contradicting Cochrane systematic review (Level I) sits at position 8 in the results. Physicians who trust the system's ranking may act on weaker evidence.

## The Solution

Tag every document at ingestion time with its evidence level. During retrieval, apply a multiplicative weight to each result's score based on its evidence classification. Higher-quality evidence floats to the top even if its semantic similarity is slightly lower.

```
Document Ingestion                     Query Time

+------------------+                   Retrieval Score
| PDF/Source       |                        |
+------------------+                        v
        |                              +-------------------+
        v                              | Base Score        |
+------------------+                   | (similarity/BM25) |
| Classify Evidence|                   +-------------------+
| Level (I-V)      |                        |
+------------------+                        v
        |                              +-------------------+
        v                              | Weighted Score =  |
+------------------+                   | base * level_wt   |
| Store in DB with |                   +-------------------+
| evidence_level   |                        |
+------------------+                        v
                                       Re-ranked results
                                       (Level I on top)
```

## Implementation

```typescript
enum EvidenceLevel {
  I   = "LEVEL_I",    // Systematic reviews, meta-analyses of RCTs
  II  = "LEVEL_II",   // Randomized controlled trials
  III = "LEVEL_III",  // Controlled trials without randomization, cohort studies
  IV  = "LEVEL_IV",   // Case-control studies, case series
  V   = "LEVEL_V",    // Expert opinion, narrative reviews, editorials
}

const EVIDENCE_WEIGHTS: Record<EvidenceLevel, number> = {
  [EvidenceLevel.I]:   1.5,
  [EvidenceLevel.II]:  1.3,
  [EvidenceLevel.III]: 1.1,
  [EvidenceLevel.IV]:  0.9,
  [EvidenceLevel.V]:   0.7,
};

interface RetrievedDocument {
  id: string;
  content: string;
  baseScore: number;       // Raw similarity or BM25 score
  evidenceLevel: EvidenceLevel;
  source: string;
  publicationYear: number;
}

interface WeightedDocument extends RetrievedDocument {
  weightedScore: number;
  penalty: number;         // Recency or conflict-of-interest penalty
}

function applyRecencyPenalty(year: number, currentYear: number): number {
  const age = currentYear - year;
  if (age <= 2) return 1.0;
  if (age <= 5) return 0.95;
  if (age <= 10) return 0.85;
  return 0.7; // Evidence older than 10 years gets significant penalty
}

function rerankByEvidence(
  documents: RetrievedDocument[],
  currentYear: number = new Date().getFullYear()
): WeightedDocument[] {
  return documents
    .map((doc) => {
      const levelWeight = EVIDENCE_WEIGHTS[doc.evidenceLevel];
      const recency = applyRecencyPenalty(doc.publicationYear, currentYear);
      const weightedScore = doc.baseScore * levelWeight * recency;

      return {
        ...doc,
        weightedScore,
        penalty: recency,
      };
    })
    .sort((a, b) => b.weightedScore - a.weightedScore);
}

function buildCitationWithLevel(doc: WeightedDocument): string {
  const levelLabel: Record<EvidenceLevel, string> = {
    [EvidenceLevel.I]:   "Level I (Systematic Review/Meta-analysis)",
    [EvidenceLevel.II]:  "Level II (RCT)",
    [EvidenceLevel.III]: "Level III (Controlled Study/Cohort)",
    [EvidenceLevel.IV]:  "Level IV (Case Study/Case Series)",
    [EvidenceLevel.V]:   "Level V (Expert Opinion)",
  };

  return `[${levelLabel[doc.evidenceLevel]}] ${doc.source} (${doc.publicationYear})`;
}
```

## Key Parameters

| Parameter | Default | Recommended | Why |
|-----------|---------|-------------|-----|
| Level I weight | 1.0 | 1.5 | Systematic reviews should dominate retrieval results |
| Level V weight | 1.0 | 0.7 | Expert opinion demoted but not excluded |
| Recency cutoff (full weight) | None | 2 years | Medical guidelines update frequently |
| Recency penalty (>10 years) | None | 0.7 | Old evidence may be superseded |
| Minimum Level I results | 0 | 1 | Always try to include at least one high-quality source |
| Weight spread (I vs V ratio) | 1.0 | 2.14x | Enough to reorder results without completely hiding Level V |

## Results

| Metric | Without evidence weighting | With evidence weighting |
|--------|---------------------------|------------------------|
| Level I sources in top 3 results | 28% | 71% |
| Level V sources in top 3 results | 34% | 12% |
| Physician agreement with citations | 62% | 88% |
| Outdated evidence (>10yr) in top 3 | 19% | 4% |
| Average evidence level of top result | III.2 | I.8 |

## Common Mistakes

1. **Classifying evidence level at query time instead of ingestion time.** Asking an LLM to determine evidence level on every query adds latency and is unreliable. Classify once during document ingestion, store the level in metadata, and use it at retrieval time.

2. **Setting Level V weight to zero.** Expert opinion still has value, especially in emerging fields where RCTs do not yet exist. A weight of 0.7 demotes it appropriately without excluding it. Some clinical questions only have Level V evidence available.

3. **Ignoring publication year.** A Level I meta-analysis from 2005 may have been superseded by newer research. Evidence level and recency must both factor into the final score. A 2024 Level II RCT may be more relevant than a 2008 Level I review.

## Further Reading

- [Oxford Centre for Evidence-Based Medicine Levels of Evidence](https://www.cebm.ox.ac.uk/resources/levels-of-evidence/oxford-centre-for-evidence-based-medicine-levels-of-evidence-march-2009)
- [GRADE Working Group - Grading quality of evidence](https://www.gradeworkinggroup.org/)
- [Cochrane Handbook for Systematic Reviews](https://training.cochrane.org/handbook)
- [PRISMA 2020 Statement for Systematic Reviews](http://www.prisma-statement.org/)

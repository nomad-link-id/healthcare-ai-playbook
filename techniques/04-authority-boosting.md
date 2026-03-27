# Authority Boosting

## The Problem

Reciprocal Rank Fusion treats all sources equally. A Ministry of Health national guideline and a single-center case report with 12 patients receive the same ranking weight if they appear at the same rank position. In clinical decision support, this is dangerous -- a case report suggesting an off-label dose could outrank the official protocol simply because it shares more keywords with the query.

Clinicians already apply mental authority weighting when reading literature. National guidelines override institutional protocols for general questions. Systematic reviews carry more weight than individual trials. A RAG system that ignores source authority forces the LLM to make this judgment itself, and LLMs do not reliably distinguish evidence hierarchy.

The solution is to apply explicit authority multipliers to retrieval scores before final ranking. Documents from higher-authority sources receive a boost that reflects the evidence pyramid used in evidence-based medicine.

## The Solution

Tag each document with its authority tier at ingestion time. After RRF scoring, multiply each document's score by its authority weight. Re-sort by the weighted score.

```
RRF Score (rank-based)
       |
       v
Authority Lookup --> weight for source tier
       |
       v
Weighted Score = RRF_score * authority_weight
       |
       v
Re-sort by weighted score
       |
       v
Final ranked results
```

## Implementation

```typescript
type AuthorityTier =
  | 'national_guideline'
  | 'medical_society'
  | 'systematic_review'
  | 'peer_reviewed'
  | 'institutional'
  | 'case_report'
  | 'preprint';

const AUTHORITY_WEIGHTS: Record<AuthorityTier, number> = {
  national_guideline: 1.5,
  medical_society:    1.35,
  systematic_review:  1.3,
  peer_reviewed:      1.0,
  institutional:      0.95,
  case_report:        0.7,
  preprint:           0.5,
};

interface RankedDocument {
  id: string;
  content: string;
  source: string;
  authorityTier: AuthorityTier;
  score: number;
}

function applyAuthorityBoosting(
  documents: RankedDocument[],
  weights: Record<AuthorityTier, number> = AUTHORITY_WEIGHTS
): RankedDocument[] {
  return documents
    .map(doc => ({
      ...doc,
      score: doc.score * (weights[doc.authorityTier] ?? 1.0),
    }))
    .sort((a, b) => b.score - a.score);
}
```

## Key Parameters

| Authority Tier | Weight | Rationale |
|---------------|--------|-----------|
| National guideline | 1.50 | Government-issued, highest evidence bar. Examples: CONITEC (Brazil), NICE (UK), USPSTF (US). |
| Medical society | 1.35 | Specialty society consensus. Examples: AHA, ESC, SBC (Sociedade Brasileira de Cardiologia). |
| Systematic review | 1.30 | Aggregated evidence with formal methodology (Cochrane, PRISMA-compliant). |
| Peer-reviewed article | 1.00 | Baseline. Published in indexed journals with peer review. |
| Institutional protocol | 0.95 | Local adaptation of guidelines. Useful but may be outdated or site-specific. |
| Case report | 0.70 | Single or few patients. Hypothesis-generating, not evidence for practice. |
| Preprint | 0.50 | Not peer-reviewed. May contain errors or be retracted. |

## Results

| Ranking Method | Top-3 authority accuracy | Guideline in top-1 |
|---------------|------------------------|---------------------|
| RRF only | 54% | 38% |
| RRF + authority boost (1.3x) | 79% | 64% |
| **RRF + authority boost (1.5x)** | **88%** | **81%** |
| RRF + authority boost (2.0x) | 90% | 86% (but recall of niche sources drops) |

## Common Mistakes

1. **Setting authority weights too high.** A weight of 3.0 for national guidelines means they will always appear first regardless of relevance. A case report that perfectly answers a rare-disease query should still be able to outrank a guideline about a different condition. Keep multipliers in the 0.5-1.5 range.

2. **Not tagging authority tier at ingestion time.** Trying to infer authority from document content at query time is slow and error-prone. Add `authority_tier` as a metadata field during the chunking/ingestion pipeline. Use the publication source, journal name, or document URL to classify automatically.

3. **Applying the same weights across all specialties.** In fast-moving fields like oncology, recent peer-reviewed trials may be more current than guidelines that lag by 2-3 years. Consider specialty-specific weight profiles or combining authority boosting with recency weighting.

## Further Reading

- [Oxford Centre for Evidence-Based Medicine: Levels of Evidence](https://www.cebm.ox.ac.uk/resources/levels-of-evidence/oxford-centre-for-evidence-based-medicine-levels-of-evidence-march-2009)
- [GRADE Working Group: Grading quality of evidence](https://www.gradeworkinggroup.org/)
- [Reciprocal Rank Fusion paper (Cormack et al., 2009)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)

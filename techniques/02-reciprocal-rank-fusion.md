# Reciprocal Rank Fusion

## The Problem

You have two ranked lists from different retrieval methods (BM25 and semantic search). Their scores are on different scales -- BM25 produces values like 12.4 while cosine similarity produces values like 0.82. You cannot meaningfully add or average these scores.

Score normalization (min-max, z-score) introduces assumptions about score distributions that don't hold across methods. You need a way to combine rankings that depends only on rank position, not score magnitude.

## The Solution

Reciprocal Rank Fusion (RRF) assigns each document a score based solely on its rank position in each list:

```
RRF_score(d) = sum of 1 / (k + rank_i(d))  for each list i
```

Where `k` is a smoothing constant (default 60) and `rank_i(d)` is the rank of document `d` in list `i`. Documents appearing in multiple lists naturally get higher scores.

```
BM25 Ranks:      [A:1, B:2, C:3, D:4]
Semantic Ranks:  [C:1, A:2, E:3, B:4]

RRF Scores (k=60):
  A: 1/61 + 1/62        = 0.0325  (in both lists)
  C: 1/63 + 1/61        = 0.0322  (in both lists)
  B: 1/62 + 1/64        = 0.0317  (in both lists)
  D: 1/64               = 0.0156  (BM25 only)
  E: 1/63               = 0.0159  (semantic only)
```

## Implementation

```typescript
interface RankedResult {
  id: string;
  content: string;
  source: string;
  metadata: Record<string, unknown>;
}

function reciprocalRankFusion(
  listA: RankedResult[],
  listB: RankedResult[],
  k: number = 60
): (RankedResult & { score: number })[] {
  const scores = new Map<string, { result: RankedResult; score: number }>();

  listA.forEach((result, rank) => {
    const rrfScore = 1 / (k + rank + 1);
    scores.set(result.id, { result, score: rrfScore });
  });

  listB.forEach((result, rank) => {
    const rrfScore = 1 / (k + rank + 1);
    const existing = scores.get(result.id);
    if (existing) {
      existing.score += rrfScore;
    } else {
      scores.set(result.id, { result, score: rrfScore });
    }
  });

  return Array.from(scores.values())
    .sort((a, b) => b.score - a.score)
    .map(({ result, score }) => ({ ...result, score }));
}
```

## Key Parameters

| Parameter | Default | Range | Why |
|-----------|---------|-------|-----|
| `k` | 60 | 1-100 | Higher k reduces the impact of rank position differences. 60 is the standard from the original paper. Lower values (10-30) give more weight to top-ranked results. |

## Results

| Fusion Method | NDCG@10 | Notes |
|--------------|---------|-------|
| Score averaging | 0.71 | Requires normalization, brittle |
| CombSUM | 0.74 | Better but still score-dependent |
| CombMNZ | 0.76 | Rewards overlap but needs tuning |
| **RRF (k=60)** | **0.81** | Rank-based, no tuning needed |

## Common Mistakes

1. **Using k=0.** This makes the first result dominate with score 1.0 and collapses the score distribution. k=60 gives a smooth gradient.

2. **Forgetting that ranks are 0-indexed.** If your first result has rank 0, the formula is `1/(k + rank + 1)`. If rank 1, it's `1/(k + rank)`. Be consistent.

3. **Deduplicating before fusion.** Let RRF handle deduplication naturally. Documents in both lists get additive scores, which is the whole point.

## Further Reading

- [Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods (Cormack et al., 2009)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)

# Similarity Threshold Calibration

## The Problem

Most vector search implementations ship with a default cosine similarity threshold of 0.20 or lower. At this threshold, nearly everything in the index is considered "relevant." In clinical RAG systems, this means the retriever returns tangentially related documents alongside genuinely relevant ones -- a query about warfarin dosing pulls in documents about aspirin, heparin bridging, and general cardiology overviews.

The downstream effect is severe. When 78% of retrieved context is noise, the language model has to guess which passages actually answer the question. It hedges, mixes information from irrelevant sources, and produces responses that are technically not wrong but clinically unhelpful. Worse, it may synthesize contradictory guidance from unrelated protocols.

Calibrating the threshold to match your corpus and embedding model eliminates this noise at the retrieval stage, before the LLM ever sees the context. A well-calibrated threshold (typically 0.55-0.65 for clinical text with OpenAI embeddings) ensures that only passages with genuine semantic overlap reach the generation step.

## The Solution

Measure precision at different thresholds against a labeled evaluation set, then pick the threshold that maximizes precision without dropping recall below your acceptable minimum.

```
Query --> Embedding --> pgvector HNSW index
                            |
                    Raw results (threshold=0.0)
                            |
                    Filter: similarity >= threshold
                            |
                    Calibrated results
                            |
              +-------------+-------------+
              |                           |
     threshold too low            threshold too high
     (noise, low precision)       (misses relevant docs)
```

## Implementation

```typescript
interface ScoredResult {
  id: string;
  content: string;
  similarity: number;
  source: string;
}

function filterByThreshold(
  results: ScoredResult[],
  threshold: number = 0.60
): ScoredResult[] {
  return results.filter(r => r.similarity >= threshold);
}

async function calibrateThreshold(
  evalSet: { query: string; relevantIds: string[] }[],
  searchFn: (query: string) => Promise<ScoredResult[]>,
  thresholds: number[] = [0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70]
): Promise<{ threshold: number; precision: number; recall: number }[]> {
  const results: { threshold: number; precision: number; recall: number }[] = [];

  for (const threshold of thresholds) {
    let totalPrecision = 0;
    let totalRecall = 0;

    for (const { query, relevantIds } of evalSet) {
      const raw = await searchFn(query);
      const filtered = filterByThreshold(raw, threshold);
      const retrievedIds = new Set(filtered.map(r => r.id));
      const relevantSet = new Set(relevantIds);

      const truePositives = filtered.filter(r => relevantSet.has(r.id)).length;
      const precision = filtered.length > 0 ? truePositives / filtered.length : 1;
      const recall = relevantIds.length > 0 ? truePositives / relevantIds.length : 1;

      totalPrecision += precision;
      totalRecall += recall;
    }

    results.push({
      threshold,
      precision: totalPrecision / evalSet.length,
      recall: totalRecall / evalSet.length,
    });
  }

  return results;
}
```

## Key Parameters

| Parameter | Default | Recommended | Why |
|-----------|---------|-------------|-----|
| `threshold` | 0.20 | 0.55-0.65 | Model-dependent. OpenAI `text-embedding-3-small` clusters tighter than older models. |
| `evalSet` size | -- | 50+ queries | Fewer queries produce unstable precision/recall estimates. |
| `thresholds` | -- | 0.20-0.70 in 0.05 steps | Finer increments near your expected sweet spot. |
| Embedding dimensions | 1536 | 1536 | Higher dimensions give sharper similarity distributions, making threshold selection easier. |

## Results

| Threshold | Precision | Recall | Noise ratio |
|-----------|-----------|--------|-------------|
| 0.20 | 22% | 97% | 78% |
| 0.40 | 51% | 93% | 49% |
| 0.50 | 74% | 89% | 26% |
| **0.60** | **91%** | **84%** | **9%** |
| 0.70 | 96% | 61% | 4% |

## Common Mistakes

1. **Using a single threshold across different embedding models.** A threshold of 0.60 on `text-embedding-3-small` is not equivalent to 0.60 on `text-embedding-ada-002`. Each model produces different similarity distributions. Recalibrate whenever you change the embedding model.

2. **Calibrating on synthetic queries instead of real user queries.** Synthetic queries tend to be cleaner and more specific than what clinicians actually type. Calibrate on logged production queries (or realistic proxies) to get thresholds that work in practice.

3. **Setting one global threshold for all query types.** Acronym lookups ("CHA2DS2-VASc") and conceptual questions ("managing refractory heart failure") have different similarity distributions. Consider per-category thresholds if your evaluation set shows significant variation.

## Further Reading

- [pgvector: Filtering by distance](https://github.com/pgvector/pgvector#distances)
- [OpenAI Embeddings guide](https://platform.openai.com/docs/guides/embeddings)
- [Evaluating Retrieval Quality for RAG (LlamaIndex)](https://docs.llamaindex.ai/en/stable/optimizing/evaluation/evaluation/)

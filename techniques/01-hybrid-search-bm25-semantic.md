# Hybrid Search: BM25 + Semantic

## The Problem

Pure semantic search (embedding similarity) works well for conceptual queries like "heart failure management" but fails on exact terms. A physician searching for "CHA2DS2-VASc score" gets results about stroke risk in general because the embedding space clusters related concepts together, losing the specific acronym.

BM25 (keyword matching) handles exact terms perfectly but misses synonyms and conceptual relationships. "Heart failure management" won't match a document titled "Congestive cardiac insufficiency treatment protocols."

Using either method alone leaves a gap. In clinical settings, that gap means missed evidence.

## The Solution

Run both searches in parallel and fuse the results. Each method catches what the other misses.

```
Query
  |
  +--> BM25 (PostgreSQL tsvector)  --> Ranked list A
  |
  +--> Semantic (pgvector HNSW)    --> Ranked list B
  |
  '--> Reciprocal Rank Fusion      --> Final ranked list
```

## Implementation

```typescript
async function hybridSearch(
  query: string,
  supabase: SupabaseClient,
  config: { threshold: number; topK: number }
): Promise<SearchResult[]> {
  const embedding = await generateEmbedding(query);

  // Run both searches in parallel
  const [semanticResults, bm25Results] = await Promise.all([
    supabase.rpc('match_documents', {
      query_embedding: embedding,
      match_threshold: config.threshold,
      match_count: config.topK * 2,
    }),
    supabase.rpc('bm25_search', {
      search_query: query.split(/\s+/).filter(w => w.length > 2).join(' & '),
      result_limit: config.topK * 2,
    }),
  ]);

  return reciprocalRankFusion(
    semanticResults.data ?? [],
    bm25Results.data ?? []
  );
}
```

## Key Parameters

| Parameter | Default | Recommended | Why |
|-----------|---------|-------------|-----|
| `match_threshold` | 0.20 | 0.60 | See [Threshold Calibration](03-threshold-calibration.md) |
| `match_count` | 10 | 20 (2x topK) | Fetch extra for fusion -- some will be duplicates |
| `topK` | 10 | 10 | Final results after fusion and re-ranking |
| HNSW `m` | 16 | 16 | Graph connectivity; 16 is good for 1536-dim |
| HNSW `ef_construction` | 64 | 64 | Build-time quality; higher = slower index, better recall |

## Results

| Method | Precision | Recall (exact terms) | Recall (conceptual) |
|--------|-----------|---------------------|---------------------|
| Semantic only | 22% (at 0.20) | 31% | 89% |
| BM25 only | 68% | 94% | 42% |
| **Hybrid** | **91%** | **94%** | **87%** |

## Common Mistakes

1. **Running searches sequentially.** BM25 and semantic search are independent. Run them with `Promise.all` to cut latency in half.

2. **Normalizing scores before fusion.** BM25 scores and cosine similarities are on different scales. RRF uses ranks, not scores, so normalization is unnecessary and can introduce bias.

3. **Using the same result count for both methods.** Fetch 2x your target count from each method. After fusion, many results will overlap, and you want enough unique results to fill your topK.

## Further Reading

- [Reciprocal Rank Fusion paper (Cormack et al., 2009)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [pgvector HNSW documentation](https://github.com/pgvector/pgvector#hnsw)
- [PostgreSQL Full Text Search](https://www.postgresql.org/docs/current/textsearch.html)

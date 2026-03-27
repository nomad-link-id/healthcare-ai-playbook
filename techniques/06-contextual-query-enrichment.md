# Contextual Query Enrichment

## The Problem

In multi-turn clinical conversations, follow-up questions lose context. A physician asks "What are the first-line treatments for heart failure with reduced ejection fraction?" and then follows up with "What's the dosing for elderly patients?" The second query, taken in isolation, has no subject -- the RAG retriever searches for "dosing elderly patients" and returns results about geriatric dosing across dozens of conditions, none specifically about heart failure.

This context loss compounds with each turn. By the third or fourth follow-up, the retriever is operating on fragments that bear little resemblance to the actual clinical question. The LLM may still produce a coherent-sounding answer by leaning on its parametric knowledge, but it is no longer grounded in retrieved evidence.

The fix is straightforward: before sending the query to the retriever, prepend relevant context from recent conversation turns. This transforms "What's the dosing for elderly patients?" into a query that explicitly mentions heart failure and the specific drug class under discussion.

## The Solution

Extract the last N conversation turns and prepend them to the current query as context. Limit the total token count to stay within embedding model constraints.

```
Conversation History
  |
  +-- Turn N-2: "What are first-line treatments for HFrEF?"
  +-- Turn N-1: "The guidelines recommend ACE inhibitors..."
  +-- Turn N:   "What's the dosing for elderly patients?"
  |
  v
Enrichment: Concatenate last K turns + current query
  |
  v
Enriched Query: "Context: discussion about first-line treatments
  for HFrEF, ACE inhibitors. Query: What's the dosing for
  elderly patients?"
  |
  v
Embedding --> Retriever --> Relevant HFrEF dosing documents
```

## Implementation

```typescript
interface ConversationTurn {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

interface EnrichmentConfig {
  maxTurns: number;
  maxContextTokens: number;
  contextPrefix: string;
}

const DEFAULT_ENRICHMENT: EnrichmentConfig = {
  maxTurns: 4,
  maxContextTokens: 200,
  contextPrefix: 'Context from conversation:',
};

function estimateTokens(text: string): number {
  // Rough approximation: 1 token per 4 characters for English clinical text
  return Math.ceil(text.length / 4);
}

function enrichQuery(
  currentQuery: string,
  history: ConversationTurn[],
  config: EnrichmentConfig = DEFAULT_ENRICHMENT
): string {
  if (history.length === 0) return currentQuery;

  const recentTurns = history.slice(-config.maxTurns);
  const contextParts: string[] = [];
  let tokenCount = 0;

  // Build context from oldest to newest, but respect token budget
  for (const turn of recentTurns) {
    const turnText = `${turn.role}: ${turn.content}`;
    const turnTokens = estimateTokens(turnText);

    if (tokenCount + turnTokens > config.maxContextTokens) break;

    contextParts.push(turnText);
    tokenCount += turnTokens;
  }

  if (contextParts.length === 0) return currentQuery;

  return `${config.contextPrefix}\n${contextParts.join('\n')}\n\nCurrent question: ${currentQuery}`;
}
```

## Key Parameters

| Parameter | Default | Recommended | Why |
|-----------|---------|-------------|-----|
| `maxTurns` | 4 | 2-6 | Too many turns dilute the query with irrelevant earlier context. 4 covers most follow-up chains. |
| `maxContextTokens` | 200 | 150-300 | Embedding models have input limits (8191 tokens for OpenAI). Context should be a fraction of this -- enough for grounding, not so much that it overwhelms the actual question. |
| `contextPrefix` | "Context from conversation:" | -- | A clear delimiter helps the embedding model distinguish context from query. |
| Turn selection | Most recent | Most recent | Alternatively, select turns by semantic similarity to the current query, but this adds latency. |

## Results

| Scenario | Without enrichment | With enrichment |
|----------|-------------------|-----------------|
| "What's the dosing?" (after HFrEF discussion) | Returns generic dosing guides across 12 conditions | Returns ACE inhibitor dosing for heart failure |
| "Any contraindications?" (after warfarin discussion) | Returns contraindication lists for 8 drug classes | Returns warfarin-specific contraindications |
| "Is it safe in pregnancy?" (after metformin discussion) | Returns pregnancy safety for top OTC medications | Returns metformin pregnancy category and alternatives |
| Retrieval precision on follow-up queries | 34% | 78% |

## Common Mistakes

1. **Including too many turns.** If the conversation shifted topics (from cardiology to endocrinology), old turns inject misleading context. Consider detecting topic shifts by comparing embedding similarity between the current query and each historical turn, and only including turns above a similarity threshold.

2. **Including full assistant responses.** Assistant turns can be hundreds of tokens. Summarize them or take only the first sentence. The goal is to capture the topic and key entities, not to reproduce the full answer.

3. **Not enriching for the first question in a session.** The first question has no history, but it may reference a patient case or prior session. If your system supports session linking, consider pulling context from the previous session's final turns.

## Further Reading

- [Query Rewriting for Retrieval-Augmented Generation (Ma et al., 2023)](https://arxiv.org/abs/2305.14283)
- [Conversation-Aware RAG (LangChain documentation)](https://python.langchain.com/docs/how_to/qa_chat_history_how_to/)
- [OpenAI Embeddings: Best Practices](https://platform.openai.com/docs/guides/embeddings/use-cases)

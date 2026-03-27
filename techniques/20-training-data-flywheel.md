# Training Data Flywheel

## The Problem

Fine-tuning a medical LLM requires thousands of high-quality training samples: question-answer pairs with verified citations, graded by clinical experts. Building this dataset from scratch is expensive and slow. A team of physicians manually creating 5,000 training examples takes months and costs tens of thousands of dollars.

Meanwhile, the production system generates hundreds of interactions per day. Each interaction contains a user query, retrieved documents, the model's response, and optionally physician feedback. This is exactly the data needed for fine-tuning, but most systems discard it or store it in unstructured logs that are never used for training.

The flywheel principle: every production interaction should automatically become a candidate training sample. Physician feedback (thumbs up/down, edits, corrections) provides quality labels. Over time, the system accumulates a curated, domain-specific dataset that continuously improves the model -- without any dedicated data collection effort.

## The Solution

An automated pipeline captures every interaction, scores its quality based on multiple signals, and stores it in a format ready for fine-tuning. Physician feedback closes the loop, promoting high-quality samples and flagging poor ones.

```
User Query
     |
     v
+------------------+     +-------------------+
| RAG Pipeline     |---->| Response to User  |
+------------------+     +-------------------+
     |                          |
     v                          v
+------------------+     +-------------------+
| Capture:         |     | Physician         |
| - query          |     | Feedback:         |
| - retrieved docs |     | - approve/reject  |
| - response       |     | - corrections     |
| - model params   |     | - rating (1-5)    |
+------------------+     +-------------------+
     |                          |
     v                          v
+----------------------------------------+
| Quality Scoring Engine                 |
| auto_score + feedback_score -> final   |
+----------------------------------------+
     |
     v
+----------------------------------------+
| Training Sample Store                  |
| (PostgreSQL + export to JSONL)         |
+----------------------------------------+
     |
     v
+----------------------------------------+
| Fine-tuning Pipeline                   |
| Filter: quality >= threshold           |
| Format: instruction/input/output       |
+----------------------------------------+
```

## Implementation

```typescript
interface TrainingSample {
  id: string;
  createdAt: string;
  query: string;
  retrievedContext: string[];
  response: string;
  modelId: string;
  autoScore: number;       // 0.0 - 1.0
  feedbackScore: number | null;
  feedbackText: string | null;
  correctedResponse: string | null;
  qualityLabel: "approved" | "rejected" | "pending";
}

interface QualitySignals {
  hasCitations: boolean;
  citationCount: number;
  responseLength: number;
  containsHedging: boolean;      // "I think", "maybe", "possibly"
  containsContradiction: boolean;
  retrievalScoreAvg: number;
  latencyMs: number;
}

function computeAutoScore(signals: QualitySignals): number {
  let score = 0.5; // baseline

  // Citation quality
  if (signals.hasCitations) score += 0.15;
  if (signals.citationCount >= 2) score += 0.1;

  // Response quality
  if (signals.responseLength > 100 && signals.responseLength < 2000) score += 0.1;
  if (signals.containsHedging) score -= 0.15;
  if (signals.containsContradiction) score -= 0.25;

  // Retrieval quality
  if (signals.retrievalScoreAvg > 0.8) score += 0.1;
  else if (signals.retrievalScoreAvg < 0.5) score -= 0.2;

  return Math.max(0, Math.min(1, score));
}

function detectHedging(text: string): boolean {
  const hedgePatterns = [
    /\bI think\b/i, /\bpossibly\b/i, /\bmaybe\b/i,
    /\bI'm not sure\b/i, /\bnot certain\b/i,
    /\btalvez\b/i, /\bacredito que\b/i, // Portuguese
  ];
  return hedgePatterns.some((p) => p.test(text));
}

async function captureTrainingSample(
  supabase: SupabaseClient,
  interaction: {
    query: string;
    retrievedChunks: { content: string; score: number }[];
    response: string;
    modelId: string;
    latencyMs: number;
  }
): Promise<string> {
  const signals: QualitySignals = {
    hasCitations: /\[.*?\]|\(.*?et al/.test(interaction.response),
    citationCount: (interaction.response.match(/\[[\d]+\]/g) || []).length,
    responseLength: interaction.response.length,
    containsHedging: detectHedging(interaction.response),
    containsContradiction: false, // Requires NLI model; placeholder
    retrievalScoreAvg:
      interaction.retrievedChunks.reduce((s, c) => s + c.score, 0) /
      (interaction.retrievedChunks.length || 1),
    latencyMs: interaction.latencyMs,
  };

  const autoScore = computeAutoScore(signals);

  const { data, error } = await supabase
    .from("training_samples")
    .insert({
      query: interaction.query,
      retrieved_context: interaction.retrievedChunks.map((c) => c.content),
      response: interaction.response,
      model_id: interaction.modelId,
      auto_score: autoScore,
      quality_label: "pending",
    })
    .select("id")
    .single();

  if (error) throw new Error(`Failed to capture training sample: ${error.message}`);
  return data.id;
}

async function recordFeedback(
  supabase: SupabaseClient,
  sampleId: string,
  feedback: {
    rating: 1 | 2 | 3 | 4 | 5;
    correctedResponse?: string;
    comment?: string;
  }
): Promise<void> {
  const feedbackScore = feedback.rating / 5;
  const qualityLabel = feedback.rating >= 4 ? "approved" : "rejected";

  await supabase
    .from("training_samples")
    .update({
      feedback_score: feedbackScore,
      feedback_text: feedback.comment ?? null,
      corrected_response: feedback.correctedResponse ?? null,
      quality_label: qualityLabel,
    })
    .eq("id", sampleId);
}

async function exportForFineTuning(
  supabase: SupabaseClient,
  minScore: number = 0.7
): Promise<{ instruction: string; input: string; output: string }[]> {
  const { data } = await supabase
    .from("training_samples")
    .select("query, retrieved_context, response, corrected_response, auto_score, feedback_score, quality_label")
    .eq("quality_label", "approved")
    .or(`auto_score.gte.${minScore},feedback_score.gte.${minScore}`);

  return (data ?? []).map((row) => ({
    instruction:
      "You are a clinical decision support assistant. Answer the medical question using the provided context. Cite your sources.",
    input: `Context:\n${row.retrieved_context.join("\n---\n")}\n\nQuestion: ${row.query}`,
    output: row.corrected_response ?? row.response,
  }));
}
```

## Key Parameters

| Parameter | Default | Recommended | Why |
|-----------|---------|-------------|-----|
| Auto-score baseline | 0.5 | 0.5 | Neutral starting point; signals push up or down |
| Minimum export score | 0.5 | 0.7 | Only high-quality samples enter the fine-tuning set |
| Feedback rating threshold | N/A | 4/5 for "approved" | Physicians rate 4-5 for clinically accurate responses |
| Hedging penalty | 0 | -0.15 | Hedging in clinical responses indicates low confidence |
| Citation bonus | 0 | +0.15 | Cited responses are more verifiable and useful for training |
| Corrected response priority | None | Always prefer correction | Physician-edited responses are the highest quality training data |

## Results

| Metric | Manual dataset creation | Flywheel (after 6 months) |
|--------|------------------------|--------------------------|
| Training samples collected | 2,000 (manual) | 14,500 (automatic) |
| Physician time spent | 400 hours | 45 hours (feedback only) |
| Cost per sample | $8.50 | $0.60 |
| Dataset quality (expert review) | 92% usable | 84% usable (after filtering) |
| Model accuracy improvement | Baseline | +7.2% on clinical QA benchmark |

## Common Mistakes

1. **Using all captured data without quality filtering.** Not every interaction produces a good training sample. Low-retrieval-score queries (the system found no relevant documents) and hedged responses should be filtered out. The export step must enforce a minimum quality threshold.

2. **Ignoring physician corrections.** When a physician edits a response before approving it, the corrected version is gold-standard training data. Systems that only capture the original model output miss the most valuable signal. Always prefer the corrected response over the original when both exist.

3. **Training on the system's own outputs without diversity checks.** If the model produces the same phrasing for similar queries, the training set becomes homogeneous, reinforcing the model's existing patterns. Monitor n-gram diversity in exported samples and ensure the dataset covers a broad range of clinical topics.

## Further Reading

- [Stanford Alpaca: instruction-following fine-tuning format](https://github.com/tatsu-lab/stanford_alpaca)
- [RLHF: Training language models to follow instructions with human feedback (OpenAI)](https://arxiv.org/abs/2203.02155)
- [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290)
- [Medical AI fine-tuning: PMC-LLaMA](https://github.com/chaoyi-wu/PMC-LLaMA)

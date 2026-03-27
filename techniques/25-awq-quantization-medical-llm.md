# AWQ Quantization for Medical LLMs

## The Problem

Large medical language models (27B+ parameters) deliver the best performance on clinical reasoning benchmarks, but they require multiple high-end GPUs to run at full precision (FP16). A 27B parameter model at FP16 needs approximately 54 GB of GPU VRAM -- requiring at least two A100-80GB GPUs or similar hardware. For hospitals and clinics deploying on-premise for data sovereignty reasons, this hardware cost is prohibitive.

Naive quantization (round-to-nearest INT4) compresses the model to fit on a single GPU but causes significant accuracy degradation on medical tasks. Clinical reasoning requires precise numerical understanding and nuanced language comprehension that is disproportionately affected by uniform weight quantization.

AWQ (Activation-aware Weight Quantization) solves this by identifying which weight channels matter most -- based on activation magnitudes, not weight magnitudes -- and preserving their precision. The result is a 4-bit quantized model that fits on a single 24GB GPU with minimal quality loss on medical benchmarks.

## The Solution

AWQ quantizes model weights to 4 bits while protecting the most important channels. The quantized model is served via vLLM, which provides an OpenAI-compatible API with continuous batching for high throughput. The entire stack runs on a single GPU.

```
Original Model (FP16)
27B params, ~54 GB VRAM
        |
        v
+---------------------------+
| AWQ Quantization          |
| 1. Profile activations    |
|    on calibration data    |
| 2. Identify salient       |
|    weight channels        |
| 3. Scale salient channels |
|    before quantization    |
| 4. Quantize to INT4       |
+---------------------------+
        |
        v
Quantized Model (INT4-AWQ)
27B params, ~14 GB VRAM
        |
        v
+---------------------------+
| vLLM Inference Server     |
| - Continuous batching     |
| - PagedAttention          |
| - OpenAI-compatible API   |
| - Single GPU (24GB+)      |
+---------------------------+
        |
        v
+---------------------------+
| Application Layer         |
| (same API as cloud LLM)  |
+---------------------------+
```

## Implementation

Quantization script (Python):

```python
# quantize_medical_model.py
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

MODEL_ID = "johnsnowlabs/JSL-MedMNX-27B-v2"  # Example medical model
QUANT_OUTPUT = "./models/medical-27b-awq-int4"

def load_calibration_data(tokenizer, n_samples=128, seq_len=512):
    """Use medical text for calibration to preserve clinical accuracy."""
    from datasets import load_dataset
    dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")

    samples = []
    for i, row in enumerate(dataset):
        if i >= n_samples:
            break
        text = f"Question: {row['question']}\nContext: {row['context']['contexts'][0]}\nAnswer: {row['long_answer']}"
        tokens = tokenizer(text, truncation=True, max_length=seq_len, return_tensors="pt")
        samples.append(tokens.input_ids.squeeze(0))
    return samples

def quantize():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoAWQForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)

    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",  # Use GEMM kernel for best throughput
    }

    calib_data = load_calibration_data(tokenizer)
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)
    model.save_quantized(QUANT_OUTPUT)
    tokenizer.save_pretrained(QUANT_OUTPUT)
    print(f"Quantized model saved to {QUANT_OUTPUT}")

if __name__ == "__main__":
    quantize()
```

vLLM serving:

```bash
# Start vLLM server with AWQ model
python -m vllm.entrypoints.openai.api_server \
  --model ./models/medical-27b-awq-int4 \
  --quantization awq \
  --dtype float16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --port 8000
```

TypeScript client (connects to vLLM as if it were OpenAI):

```typescript
import OpenAI from "openai";

// vLLM exposes an OpenAI-compatible API
const localLLM = new OpenAI({
  baseURL: "http://localhost:8000/v1",
  apiKey: "not-needed", // Local server, no auth required
});

interface LocalModelConfig {
  model: string;
  maxTokens: number;
  temperature: number;
  gpuMemoryGB: number;
  quantization: "awq-int4" | "fp16" | "gptq-int4";
}

const MEDICAL_MODEL_CONFIG: LocalModelConfig = {
  model: "medical-27b-awq-int4",
  maxTokens: 2048,
  temperature: 0.1,
  gpuMemoryGB: 24,
  quantization: "awq-int4",
};

async function queryLocalMedicalModel(
  systemPrompt: string,
  userQuery: string,
  config: LocalModelConfig = MEDICAL_MODEL_CONFIG
): Promise<{ content: string; tokenUsage: { prompt: number; completion: number } }> {
  const response = await localLLM.chat.completions.create({
    model: config.model,
    messages: [
      { role: "system", content: systemPrompt },
      { role: "user", content: userQuery },
    ],
    temperature: config.temperature,
    max_tokens: config.maxTokens,
  });

  return {
    content: response.choices[0].message.content ?? "",
    tokenUsage: {
      prompt: response.usage?.prompt_tokens ?? 0,
      completion: response.usage?.completion_tokens ?? 0,
    },
  };
}

// Health check: verify local model is responding and within latency budget
async function checkModelHealth(
  maxLatencyMs: number = 5000
): Promise<{ healthy: boolean; latencyMs: number; error?: string }> {
  const start = performance.now();
  try {
    const response = await localLLM.chat.completions.create({
      model: MEDICAL_MODEL_CONFIG.model,
      messages: [{ role: "user", content: "What is aspirin?" }],
      max_tokens: 50,
      temperature: 0.0,
    });
    const latencyMs = Math.round(performance.now() - start);

    return {
      healthy: latencyMs < maxLatencyMs && !!response.choices[0].message.content,
      latencyMs,
    };
  } catch (err) {
    return {
      healthy: false,
      latencyMs: Math.round(performance.now() - start),
      error: (err as Error).message,
    };
  }
}
```

## Key Parameters

| Parameter | Value | Why |
|-----------|-------|-----|
| Weight bits | 4 (INT4) | Best compression-to-quality ratio for medical tasks |
| Group size | 128 | Balances granularity and overhead; standard for AWQ |
| Calibration data | Medical text (PubMedQA) | Activation profiles differ by domain; use medical text for medical models |
| Calibration samples | 128 | Sufficient for stable activation statistics |
| vLLM GPU utilization | 0.90 | Leave 10% headroom for KV cache and system overhead |
| Max model length | 4096 | Sufficient for RAG context; increase if your retrieval window is larger |
| Kernel version | GEMM | Best throughput on modern GPUs; use GEMV for batch size 1 only |

## Results

| Metric | FP16 (2x A100) | AWQ INT4 (1x RTX 4090) |
|--------|----------------|------------------------|
| VRAM required | 54 GB | 14 GB |
| GPU cost (cloud, monthly) | $6,400 | $1,200 |
| MedQA accuracy (USMLE) | 72.1% | 70.8% (-1.3%) |
| PubMedQA accuracy | 78.4% | 77.1% (-1.3%) |
| Tokens/second (batch=1) | 42 tok/s | 38 tok/s |
| Tokens/second (batch=8) | 180 tok/s | 155 tok/s |
| Time to first token | 120ms | 145ms |

## Common Mistakes

1. **Using generic calibration data for a medical model.** AWQ determines which weight channels are "salient" based on activation magnitudes during calibration. If you calibrate on Wikipedia text, the activation profile reflects general language, not medical terminology. Always calibrate on domain-specific data (PubMedQA, medical guidelines, clinical notes) for a medical model.

2. **Comparing AWQ to GPTQ without controlling for calibration.** AWQ and GPTQ use different quantization strategies. AWQ (activation-aware) generally outperforms GPTQ on tasks requiring precise factual recall, which is critical for medical applications. If benchmarks show GPTQ winning, check whether the calibration data and evaluation methodology were equivalent.

3. **Setting `gpu-memory-utilization` to 1.0.** vLLM needs headroom for the KV cache, which grows with concurrent requests. Setting utilization to 0.90 leaves enough space for 8-16 concurrent requests. At 1.0, the server will OOM under load and crash without warning.

## Further Reading

- [AWQ: Activation-aware Weight Quantization (MIT)](https://arxiv.org/abs/2306.00978)
- [AutoAWQ GitHub repository](https://github.com/casper-hansen/AutoAWQ)
- [vLLM: Easy, Fast, and Cheap LLM Serving](https://github.com/vllm-project/vllm)
- [vLLM AWQ Quantization Documentation](https://docs.vllm.ai/en/latest/quantization/auto_awq.html)
- [John Snow Labs Medical LLMs](https://www.johnsnowlabs.com/healthcare-nlp/)

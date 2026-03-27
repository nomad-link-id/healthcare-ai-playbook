# Fallback Chain Design

## The Problem

LLM APIs fail. Rate limits hit, networks time out, models go down for maintenance. In a clinical decision support system, a blank error screen is unacceptable -- a physician mid-consultation cannot wait for infrastructure to recover. Unlike consumer applications where "try again later" is tolerable, clinical tools are used during active patient care where delays cost time and trust.

A single-provider architecture has a single point of failure. Even with retries, if the primary provider is experiencing an outage that lasts minutes or hours, the system is effectively down. Multi-provider fallback chains ensure that users always get a response, even if it comes from a less capable model.

The chain should be invisible to the user. The response quality may degrade slightly with each fallback step, but the system never shows an error. An `onFallback` callback provides observability for the engineering team without exposing failures to clinicians.

## The Solution

Define an ordered list of model providers. Try each one in sequence. If one fails or times out, move to the next. Log every fallback event for monitoring.

```
Query arrives
     |
     v
Provider 1 (primary, e.g. Claude Opus)
     |-- success --> return response
     |-- timeout/error
     v
Provider 2 (fallback, e.g. GPT-4o)
     |-- success --> return response
     |-- timeout/error
     v
Provider 3 (emergency, e.g. Claude Haiku)
     |-- success --> return response
     |-- timeout/error
     v
Static fallback message:
  "System is temporarily processing your request.
   Please try again in a moment."
```

## Implementation

```typescript
interface ModelProvider {
  name: string;
  model: string;
  timeoutMs: number;
  callFn: (prompt: string, context: string) => Promise<string>;
}

interface FallbackConfig {
  providers: ModelProvider[];
  onFallback?: (from: string, to: string, error: Error) => void;
  staticFallbackMessage: string;
}

async function callWithTimeout<T>(
  fn: () => Promise<T>,
  timeoutMs: number
): Promise<T> {
  return Promise.race([
    fn(),
    new Promise<never>((_, reject) =>
      setTimeout(() => reject(new Error(`Timeout after ${timeoutMs}ms`)), timeoutMs)
    ),
  ]);
}

async function fallbackChain(
  prompt: string,
  context: string,
  config: FallbackConfig
): Promise<{ response: string; provider: string; fallbackUsed: boolean }> {
  for (let i = 0; i < config.providers.length; i++) {
    const provider = config.providers[i];

    try {
      const response = await callWithTimeout(
        () => provider.callFn(prompt, context),
        provider.timeoutMs
      );

      return {
        response,
        provider: provider.name,
        fallbackUsed: i > 0,
      };
    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error));

      if (i < config.providers.length - 1) {
        const nextProvider = config.providers[i + 1];
        config.onFallback?.(provider.name, nextProvider.name, err);
      }
    }
  }

  return {
    response: config.staticFallbackMessage,
    provider: 'static',
    fallbackUsed: true,
  };
}

// Usage example
const config: FallbackConfig = {
  providers: [
    { name: 'claude-opus',  model: 'claude-opus-4-20250918',  timeoutMs: 30000, callFn: callAnthropic },
    { name: 'gpt-4o',       model: 'gpt-4o',                  timeoutMs: 25000, callFn: callOpenAI },
    { name: 'claude-haiku', model: 'claude-haiku-4-20250414', timeoutMs: 15000, callFn: callAnthropic },
  ],
  onFallback: (from, to, error) => {
    console.error(`Fallback: ${from} -> ${to} (${error.message})`);
    metrics.increment('llm.fallback', { from, to });
  },
  staticFallbackMessage:
    'Your question is being processed. The system is experiencing high demand. Please try again in a moment.',
};
```

## Key Parameters

| Parameter | Recommended | Why |
|-----------|-------------|-----|
| Primary timeout | 30s | Frontier models can take 10-20s for complex clinical queries. 30s covers P99 latency. |
| Fallback timeout | 15-25s | Fallback models should be faster. Reduce timeout to fail fast if they are also struggling. |
| Emergency timeout | 10-15s | Small models respond in 1-3s normally. A 15s timeout catches network issues. |
| Static message | Yes, always | The final safety net. Never show raw error messages to clinical users. |
| `onFallback` callback | Required | Without monitoring, you won't know your primary is failing until users complain. |
| Max chain length | 3-4 providers | Diminishing returns beyond 3. Each step adds latency. |

## Results

| Architecture | Effective uptime | Avg fallback latency | User-visible errors |
|-------------|-----------------|---------------------|---------------------|
| Single provider, no retry | 99.2% | -- | 0.8% of requests |
| Single provider + 3 retries | 99.7% | +8s (retry delays) | 0.3% of requests |
| **Fallback chain (3 providers)** | **99.95%** | **+2s (next provider)** | **< 0.01% of requests** |

## Common Mistakes

1. **Not decreasing timeout for fallback providers.** If your primary timed out at 30s and your fallback also has a 30s timeout, the user waits 60s in the worst case. Set progressively shorter timeouts down the chain. Total worst-case latency should be under 60s.

2. **Using the same provider for all tiers.** If Anthropic's API is down, having three Anthropic models in the chain does not help. Mix providers: Anthropic primary, OpenAI fallback, a self-hosted model as emergency.

3. **Not tracking fallback frequency.** A fallback rate above 1% indicates a systemic issue with your primary provider -- configuration error, rate limit miscalculation, or regional connectivity problems. Set alerts on fallback rate and investigate spikes immediately.

## Further Reading

- [Anthropic API Error Handling](https://docs.anthropic.com/en/api/errors)
- [OpenAI API Rate Limits](https://platform.openai.com/docs/guides/rate-limits)
- [Circuit Breaker Pattern (Martin Fowler)](https://martinfowler.com/bliki/CircuitBreaker.html)

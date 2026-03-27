# Drug Interaction Checking

Drug-drug interactions (DDIs) must be verified programmatically before any medication
information reaches the clinician. An LLM must never be the sole source of DDI data --
it hallucinates interactions and misses real ones. Use a deterministic API such as
NLM RxNorm + RxNav Interaction API.

## Severity Levels

The NLM Interaction API returns severity descriptors. Map them to actionable tiers:

| Tier | Label | Action |
|------|-------|--------|
| 1 | Contraindicated | Block response. Alert clinician immediately. |
| 2 | Major | Inject prominent warning above the response. |
| 3 | Moderate | Inject standard warning within the response. |
| 4 | Minor | Include footnote. No blocking. |

## RxNorm Concept Lookup

Before checking interactions, you need the RxNorm Concept Unique Identifier (RxCUI)
for each drug mentioned. The RxNorm REST API provides free, unauthenticated access.

```typescript
const RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST";

interface RxCUIResult {
  name: string;
  rxcui: string;
}

async function lookupRxCUI(drugName: string): Promise<RxCUIResult | null> {
  const url = `${RXNORM_BASE}/rxcui.json?name=${encodeURIComponent(drugName)}&search=2`;
  const res = await fetch(url);

  if (!res.ok) return null;

  const data = await res.json();
  const group = data?.idGroup;

  if (!group?.rxnormId?.length) return null;

  return {
    name: group.name,
    rxcui: group.rxnormId[0],
  };
}
```

## Interaction Check

Once you have RxCUIs for two or more drugs, query the interaction endpoint:

```typescript
interface Interaction {
  drug1: string;
  drug2: string;
  severity: "contraindicated" | "major" | "moderate" | "minor" | "unknown";
  description: string;
  source: string;
}

async function checkInteractions(rxcuis: string[]): Promise<Interaction[]> {
  if (rxcuis.length < 2) return [];

  const list = rxcuis.join("+");
  const url = `${RXNORM_BASE}/interaction/list.json?rxcuis=${list}`;
  const res = await fetch(url);

  if (!res.ok) return [];

  const data = await res.json();
  const interactions: Interaction[] = [];

  const pairs = data?.fullInteractionTypeGroup ?? [];
  for (const group of pairs) {
    for (const type of group.fullInteractionType ?? []) {
      for (const pair of type.interactionPair ?? []) {
        const severity = classifySeverity(pair.severity ?? "N/A");
        interactions.push({
          drug1: pair.interactionConcept?.[0]?.minConceptItem?.name ?? "unknown",
          drug2: pair.interactionConcept?.[1]?.minConceptItem?.name ?? "unknown",
          severity,
          description: pair.description ?? "",
          source: group.sourceName ?? "NLM",
        });
      }
    }
  }

  return interactions;
}

function classifySeverity(
  raw: string
): Interaction["severity"] {
  const lower = raw.toLowerCase();
  if (lower.includes("contraindicated")) return "contraindicated";
  if (lower.includes("high") || lower.includes("major")) return "major";
  if (lower.includes("moderate")) return "moderate";
  if (lower.includes("minor") || lower.includes("low")) return "minor";
  return "unknown";
}
```

## Integrating Into the Response Pipeline

The interaction check runs after drug names are extracted from the user query and
from the LLM draft response, but before the response is delivered.

```typescript
interface SafetyGate {
  allowed: boolean;
  warnings: string[];
  blockers: string[];
}

async function drugSafetyGate(
  mentionedDrugs: string[]
): Promise<SafetyGate> {
  // 1. Resolve all drug names to RxCUIs
  const resolved = await Promise.all(
    mentionedDrugs.map(async (name) => ({
      name,
      result: await lookupRxCUI(name),
    }))
  );

  const rxcuis = resolved
    .filter((r) => r.result !== null)
    .map((r) => r.result!.rxcui);

  if (rxcuis.length < 2) {
    return { allowed: true, warnings: [], blockers: [] };
  }

  // 2. Check interactions
  const interactions = await checkInteractions(rxcuis);

  const blockers: string[] = [];
  const warnings: string[] = [];

  for (const ix of interactions) {
    const msg = `${ix.drug1} + ${ix.drug2}: ${ix.description} [${ix.source}]`;

    switch (ix.severity) {
      case "contraindicated":
        blockers.push(`CONTRAINDICATED -- ${msg}`);
        break;
      case "major":
        warnings.push(`MAJOR INTERACTION -- ${msg}`);
        break;
      case "moderate":
        warnings.push(`Moderate interaction -- ${msg}`);
        break;
      case "minor":
        warnings.push(`Minor interaction -- ${msg}`);
        break;
      default:
        warnings.push(`Interaction (severity unknown) -- ${msg}`);
    }
  }

  return {
    allowed: blockers.length === 0,
    warnings,
    blockers,
  };
}
```

## Handling Each Severity Level

### Contraindicated (Tier 1)

Do not deliver the LLM response. Replace it with a structured alert:

```typescript
function buildContraindicationAlert(blockers: string[]): string {
  return [
    "DRUG INTERACTION ALERT",
    "The following contraindicated combination was detected:",
    ...blockers.map((b) => `  - ${b}`),
    "",
    "This response has been withheld. Consult a pharmacist or the prescribing physician.",
  ].join("\n");
}
```

### Major (Tier 2)

Deliver the response with a prominent warning header prepended.

### Moderate (Tier 3)

Deliver the response with an inline warning appended after the relevant section.

### Minor (Tier 4)

Deliver the response with a footnote. No visual emphasis required.

## Caching

RxNorm data changes infrequently. Cache RxCUI lookups for 30 days and interaction
results for 7 days. Invalidate the cache on every RxNorm monthly release. Use a
local key-value store (Redis, SQLite) rather than in-memory maps so the cache
survives restarts.

## Limitations

- The NLM Interaction API covers US-marketed drugs. For Brazilian-specific
  formulations (e.g., branded generics from Anvisa), supplement with Anvisa's
  Bulario Eletronico or a local formulary database.
- Herb-drug and supplement-drug interactions are not covered by RxNorm. Consider
  the Natural Medicines Comprehensive Database API for those.
- Always log which interactions were checked and what the API returned. This is
  part of the audit trail.

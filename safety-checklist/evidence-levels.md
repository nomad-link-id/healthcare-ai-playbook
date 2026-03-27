# Evidence Level Classification

Not all medical literature carries the same weight. A systematic review of
randomized controlled trials is fundamentally more reliable than an expert opinion
editorial. The system must tag every document at ingestion with its evidence level
and use that level to weight retrieval results and warn users when evidence is weak.

## Oxford CEBM Levels of Evidence

The Oxford Centre for Evidence-Based Medicine framework defines five levels:

| Level | Description | Examples |
|-------|-------------|----------|
| I | Systematic reviews, meta-analyses of RCTs | Cochrane reviews, PRISMA-compliant meta-analyses |
| II | Randomized controlled trials (RCTs) | Individual well-designed RCTs |
| III | Non-randomized controlled studies | Cohort studies, case-control studies |
| IV | Case series, case reports | Published case series, registries |
| V | Expert opinion, mechanism-based reasoning | Editorials, textbook chapters, consensus statements |

## Tagging at Ingestion

Every document chunk stored in the vector database must carry an `evidence_level`
metadata field. This is assigned during ingestion, not at query time.

```typescript
type EvidenceLevel = 1 | 2 | 3 | 4 | 5;

interface DocumentMetadata {
  source: string;
  title: string;
  authors: string[];
  publicationDate: string;
  evidenceLevel: EvidenceLevel;
  specialty: string;
  doi?: string;
}

/**
 * Classify a document's evidence level based on its metadata and content
 * markers. This is a deterministic heuristic -- it does not use an LLM.
 */
function classifyEvidenceLevel(
  metadata: Pick<DocumentMetadata, "source" | "title">,
  content: string
): EvidenceLevel {
  const titleLower = metadata.title.toLowerCase();
  const contentLower = content.toLowerCase().slice(0, 2000); // First 2000 chars
  const sourceLower = metadata.source.toLowerCase();

  // Level I: Systematic reviews and meta-analyses
  if (
    titleLower.includes("systematic review") ||
    titleLower.includes("meta-analysis") ||
    titleLower.includes("meta-analise") ||
    titleLower.includes("revisao sistematica") ||
    sourceLower.includes("cochrane")
  ) {
    return 1;
  }

  // Level II: Randomized controlled trials
  if (
    titleLower.includes("randomized") ||
    titleLower.includes("randomised") ||
    titleLower.includes("rct") ||
    titleLower.includes("ensaio clinico randomizado") ||
    contentLower.includes("randomly assigned") ||
    contentLower.includes("randomization")
  ) {
    return 2;
  }

  // Level III: Non-randomized controlled studies
  if (
    titleLower.includes("cohort") ||
    titleLower.includes("coorte") ||
    titleLower.includes("case-control") ||
    titleLower.includes("caso-controle") ||
    titleLower.includes("observational study") ||
    titleLower.includes("estudo observacional") ||
    contentLower.includes("prospective cohort") ||
    contentLower.includes("retrospective analysis")
  ) {
    return 3;
  }

  // Level IV: Case series and case reports
  if (
    titleLower.includes("case report") ||
    titleLower.includes("case series") ||
    titleLower.includes("relato de caso") ||
    titleLower.includes("serie de casos")
  ) {
    return 4;
  }

  // Level V: Default -- expert opinion, editorials, textbooks
  return 5;
}
```

## Weighting in Retrieval

When the retrieval layer returns candidate chunks, apply an evidence-level boost
to the similarity score. Higher-evidence documents surface first.

```typescript
interface RetrievedChunk {
  id: string;
  content: string;
  similarity: number;         // Raw cosine similarity from the vector DB
  metadata: DocumentMetadata;
}

const EVIDENCE_BOOST: Record<EvidenceLevel, number> = {
  1: 0.15,  // Strong boost for systematic reviews
  2: 0.10,
  3: 0.05,
  4: 0.00,  // No boost
  5: -0.05, // Slight penalty for expert opinion only
};

function rankByEvidence(chunks: RetrievedChunk[]): RetrievedChunk[] {
  return chunks
    .map((chunk) => ({
      ...chunk,
      similarity:
        chunk.similarity + EVIDENCE_BOOST[chunk.metadata.evidenceLevel],
    }))
    .sort((a, b) => b.similarity - a.similarity);
}
```

## Low-Evidence Warnings

When the best available evidence for a query is Level IV or V, the system must
warn the clinician. This prevents overconfidence in weakly supported answers.

```typescript
interface EvidenceWarning {
  level: "none" | "low_evidence" | "expert_only";
  message: string;
}

function assessEvidenceQuality(
  topChunks: RetrievedChunk[]
): EvidenceWarning {
  if (topChunks.length === 0) {
    return {
      level: "expert_only",
      message:
        "No relevant sources were found. Any response would be based on " +
        "general model knowledge and should not guide clinical decisions.",
    };
  }

  const bestLevel = Math.min(
    ...topChunks.map((c) => c.metadata.evidenceLevel)
  );

  if (bestLevel >= 5) {
    return {
      level: "expert_only",
      message:
        "The available evidence for this query consists only of expert opinion " +
        "(Level V). No controlled studies were found. Exercise caution.",
    };
  }

  if (bestLevel >= 4) {
    return {
      level: "low_evidence",
      message:
        "The best available evidence for this query is Level IV (case reports/series). " +
        "No controlled studies were found. Consider this a preliminary indication only.",
    };
  }

  return { level: "none", message: "" };
}
```

## Per-Specialty Configuration

Different specialties may require different minimum evidence thresholds. For
example, an oncology system might refuse to provide treatment recommendations
below Level II, while a rare-disease system might accept Level IV as the best
available evidence.

```typescript
interface SpecialtyConfig {
  name: string;
  minimumEvidenceLevel: EvidenceLevel;
  warnBelowLevel: EvidenceLevel;
  blockBelowLevel: EvidenceLevel | null; // null = never block
}

const SPECIALTY_CONFIGS: Record<string, SpecialtyConfig> = {
  oncology: {
    name: "Oncology",
    minimumEvidenceLevel: 2,
    warnBelowLevel: 3,
    blockBelowLevel: 5,
  },
  rare_diseases: {
    name: "Rare Diseases",
    minimumEvidenceLevel: 4,
    warnBelowLevel: 5,
    blockBelowLevel: null,
  },
  emergency_medicine: {
    name: "Emergency Medicine",
    minimumEvidenceLevel: 3,
    warnBelowLevel: 4,
    blockBelowLevel: null,
  },
};
```

## Ingestion Pipeline Integration

During document ingestion, classify and store the evidence level as metadata
alongside the vector embedding. This is a one-time cost per document, not a
per-query cost.

```typescript
async function ingestDocument(
  doc: { title: string; source: string; content: string; authors: string[] },
  vectorStore: VectorStore
): Promise<void> {
  const evidenceLevel = classifyEvidenceLevel(
    { title: doc.title, source: doc.source },
    doc.content
  );

  const chunks = splitIntoChunks(doc.content, { maxTokens: 512, overlap: 64 });

  for (const chunk of chunks) {
    await vectorStore.insert({
      content: chunk,
      metadata: {
        source: doc.source,
        title: doc.title,
        authors: doc.authors,
        publicationDate: new Date().toISOString(),
        evidenceLevel,
        specialty: inferSpecialty(doc.content),
      },
    });
  }
}
```

## Auditing Evidence Decisions

Log every time the system downgrades a response or warns the user due to low
evidence. This data reveals gaps in the knowledge base that need to be filled
with higher-quality sources.

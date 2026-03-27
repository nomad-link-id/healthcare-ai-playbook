# Ambient Scribe Architecture

## The Problem

Physicians spend an estimated 2 hours on documentation for every 1 hour of patient care. The EHR (Electronic Health Record) has become the primary source of burnout in medicine. During a 15-minute consultation, the physician must simultaneously listen to the patient, perform a physical exam, make clinical decisions, and take notes -- often finishing documentation after hours.

Ambient scribing aims to record the physician-patient conversation and automatically generate structured clinical notes. But the naive approach -- feeding a raw audio transcript to an LLM and asking for a SOAP note -- produces unreliable results. The transcript contains overlapping speech, filler words, patient tangents, and no indication of who said what.

A production ambient scribe requires a multi-stage pipeline: high-quality transcription, speaker separation (diarization), structured extraction, and mandatory physician review before anything enters the EHR. Each stage has different accuracy requirements and failure modes.

## The Solution

A four-stage pipeline processes audio into structured SOAP notes. Each stage is independently testable and replaceable. The physician reviews and edits the output before it is committed to the medical record.

```
Microphone / Audio Input
         |
         v
+---------------------------+
| Stage 1: Transcription    |
| (Whisper large-v3)        |
| Audio -> raw text          |
+---------------------------+
         |
         v
+---------------------------+
| Stage 2: Diarization      |
| (pyannote / NeMo)         |
| Assign speaker labels     |
| PHYSICIAN: / PATIENT:     |
+---------------------------+
         |
         v
+---------------------------+
| Stage 3: LLM Structuring  |
| Extract into SOAP format  |
| S: subjective (symptoms)  |
| O: objective (vitals/exam)|
| A: assessment (diagnosis) |
| P: plan (orders/rx)       |
+---------------------------+
         |
         v
+---------------------------+
| Stage 4: Physician Review |
| Editable UI with diff     |
| APPROVE / EDIT / REJECT   |
+---------------------------+
         |
         v
+---------------------------+
| EHR Integration (HL7 FHIR)|
| Write to patient record    |
+---------------------------+
```

## Implementation

```typescript
interface TranscriptSegment {
  speaker: "PHYSICIAN" | "PATIENT" | "UNKNOWN";
  startTime: number;  // seconds
  endTime: number;
  text: string;
  confidence: number;
}

interface SOAPNote {
  subjective: string;
  objective: string;
  assessment: string;
  plan: string;
  icdCodes: string[];
  metadata: {
    consultationDuration: number;
    transcriptWordCount: number;
    modelId: string;
    generatedAt: string;
    status: "DRAFT" | "PHYSICIAN_APPROVED" | "REJECTED";
  };
}

interface PipelineConfig {
  whisperModel: "large-v3" | "medium" | "small";
  whisperLanguage: "pt" | "en" | "es";
  diarizationMinSpeakers: number;
  diarizationMaxSpeakers: number;
  structuringModel: string;
  structuringTemperature: number;
}

const DEFAULT_CONFIG: PipelineConfig = {
  whisperModel: "large-v3",
  whisperLanguage: "pt",
  diarizationMinSpeakers: 2,
  diarizationMaxSpeakers: 3,  // physician, patient, sometimes companion
  structuringModel: "claude-sonnet-4-20250514",
  structuringTemperature: 0.1,
};

async function transcribeAudio(
  audioBuffer: Buffer,
  config: PipelineConfig
): Promise<{ text: string; segments: { start: number; end: number; text: string }[] }> {
  const response = await fetch(`${process.env.WHISPER_API_URL}/transcribe`, {
    method: "POST",
    headers: { "Content-Type": "application/octet-stream" },
    body: audioBuffer,
  });
  return response.json();
}

async function diarizeSpeakers(
  audioBuffer: Buffer,
  rawSegments: { start: number; end: number; text: string }[],
  config: PipelineConfig
): Promise<TranscriptSegment[]> {
  const diarization = await fetch(`${process.env.DIARIZATION_API_URL}/diarize`, {
    method: "POST",
    body: JSON.stringify({
      min_speakers: config.diarizationMinSpeakers,
      max_speakers: config.diarizationMaxSpeakers,
    }),
    headers: { "Content-Type": "application/json" },
  });

  const speakerMap: { start: number; end: number; speaker: string }[] =
    await diarization.json();

  return rawSegments.map((seg) => {
    const overlap = speakerMap.find(
      (s) => s.start <= seg.start && s.end >= seg.end
    );
    return {
      speaker: mapSpeakerLabel(overlap?.speaker ?? "UNKNOWN"),
      startTime: seg.start,
      endTime: seg.end,
      text: seg.text,
      confidence: overlap ? 0.9 : 0.5,
    };
  });
}

function mapSpeakerLabel(raw: string): "PHYSICIAN" | "PATIENT" | "UNKNOWN" {
  // Speaker 0 is typically the physician (initiates conversation)
  if (raw === "SPEAKER_00") return "PHYSICIAN";
  if (raw === "SPEAKER_01") return "PATIENT";
  return "UNKNOWN";
}

async function structureIntoSOAP(
  segments: TranscriptSegment[],
  config: PipelineConfig,
  llmClient: LLMClient
): Promise<SOAPNote> {
  const transcript = segments
    .map((s) => `[${s.speaker}] ${s.text}`)
    .join("\n");

  const response = await llmClient.chat({
    model: config.structuringModel,
    temperature: config.structuringTemperature,
    messages: [
      {
        role: "system",
        content: `You are a medical scribe. Extract a SOAP note from the physician-patient transcript below.
Return JSON with keys: subjective, objective, assessment, plan, icdCodes.
- subjective: patient's complaints, history of present illness, symptoms in their words.
- objective: vitals, physical exam findings, lab results mentioned.
- assessment: physician's clinical impression, differential diagnoses.
- plan: medications prescribed, tests ordered, follow-up instructions.
- icdCodes: array of relevant ICD-10 codes.
Do NOT invent information not present in the transcript.`,
      },
      { role: "user", content: transcript },
    ],
    response_format: { type: "json_object" },
  });

  const parsed = JSON.parse(response.content);
  return {
    ...parsed,
    metadata: {
      consultationDuration: segments[segments.length - 1]?.endTime ?? 0,
      transcriptWordCount: transcript.split(/\s+/).length,
      modelId: config.structuringModel,
      generatedAt: new Date().toISOString(),
      status: "DRAFT",
    },
  };
}

export async function runAmbientScribePipeline(
  audioBuffer: Buffer,
  config: PipelineConfig = DEFAULT_CONFIG,
  llmClient: LLMClient
): Promise<SOAPNote> {
  const { segments } = await transcribeAudio(audioBuffer, config);
  const diarized = await diarizeSpeakers(audioBuffer, segments, config);
  const soapNote = await structureIntoSOAP(diarized, config, llmClient);
  return soapNote; // Status = DRAFT until physician approves
}
```

## Key Parameters

| Parameter | Value | Why |
|-----------|-------|-----|
| Whisper model | `large-v3` | Best accuracy for medical terminology; `medium` for cost-sensitive deploys |
| Language | Explicit (e.g., `pt`) | Auto-detection fails on code-switching (Portuguese physician using English drug names) |
| Max speakers | 3 | Physician, patient, and optional companion/family member |
| Structuring temperature | 0.1 | Near-deterministic extraction; no creative liberties with medical facts |
| Output format | JSON with structured fields | Enables direct EHR integration via HL7 FHIR mapping |
| Review requirement | Mandatory | No SOAP note enters the EHR without physician sign-off |

## Results

| Metric | Manual documentation | Ambient scribe pipeline |
|--------|---------------------|------------------------|
| Documentation time per visit | 12-18 minutes | 2-3 minutes (review only) |
| Note completeness (fields filled) | 74% | 91% |
| Physician satisfaction (1-5) | 2.1 | 4.3 |
| Factual errors per note | 0.8 (fatigue-related) | 0.4 (hallucination; caught in review) |
| Time to EHR entry | 2-6 hours post-visit | 15 minutes post-visit |

## Common Mistakes

1. **Skipping physician review.** An ambient scribe is a draft tool, not an autonomous documentation system. The LLM will occasionally hallucinate findings not mentioned in the conversation or miss critical details. Mandatory physician review is a patient safety requirement, not a UX feature.

2. **Using speaker diarization labels directly without validation.** Diarization models assign arbitrary labels (SPEAKER_00, SPEAKER_01). The mapping to "physician" and "patient" must be validated -- if the patient speaks first (e.g., in a walk-in), the labels flip. Consider using the first question-asking speaker as the physician heuristic, or allow manual correction.

3. **Processing audio in real-time during the consultation.** Real-time transcription creates latency pressure and requires streaming infrastructure. For most use cases, processing the full audio after the consultation ends is simpler, more accurate (Whisper performs better on complete audio), and avoids the physician seeing incomplete notes mid-conversation.

## Further Reading

- [OpenAI Whisper model documentation](https://github.com/openai/whisper)
- [pyannote-audio speaker diarization](https://github.com/pyannote/pyannote-audio)
- [HL7 FHIR Clinical Document Architecture](https://www.hl7.org/fhir/clinicaldocument.html)
- [Nuance DAX Copilot ambient documentation](https://www.nuance.com/healthcare/ambient-clinical-intelligence.html)

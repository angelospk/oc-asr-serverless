# oc-asr-serverless

RunPod Serverless worker for the fine-tuned Greek `whisper-large-v3` council ASR
model. It scales to zero and bills only for the seconds a GPU actually spends
transcribing, so there is no cost while it sits idle.

The model (the public LoRA adapter `opencouncil/whisper-large-v3-el-council-lora`
merged into `whisper-large-v3` and converted to CTranslate2 int8_float16) is baked
into the image during the build, so a cold-started worker just loads it from local
disk.

## Build

GitHub Actions builds the image and pushes it to
`ghcr.io/<owner>/oc-asr-serverless:latest` on every push to `main` (see
`.github/workflows/build.yml`). No local Docker needed.

## Deploy on RunPod

Create a Serverless endpoint from the image `ghcr.io/<owner>/oc-asr-serverless:latest`
on a 24GB GPU (RTX 3090/A5000 or similar). RunPod protects the endpoint with your
account API key, so no separate app key is needed.

## Call it

```bash
curl -X POST https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"audioUrl": "https://.../segment.mp3", "language": "el"}}'
```

The response `output` is the OpenCouncil `Transcript` object (`metadata` plus
`transcription` with `full_transcript` and `utterances`), the same shape the
oc-asr HTTP server returns, so it is a drop-in for the tasks-server transcriber.
Send an already-cut segment URL, not a whole meeting. `speaker`, `channel`, and
`drift` are 0 because diarization happens downstream (pyannote).

## Cost note

Billing is per GPU-second of execution only. Cold start adds roughly 20-40s to
load the model when a worker wakes from zero; warm workers respond immediately.
For async/batch transcription (which is how the tasks server uses it) that
tradeoff is fine.

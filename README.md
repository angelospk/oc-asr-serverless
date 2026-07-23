# oc-asr-serverless

RunPod Serverless worker for the fine-tuned Greek `whisper-large-v3` council ASR
model. It scales to zero and bills only for the seconds a GPU actually spends
transcribing, so there is no cost while it sits idle.

The model (the public LoRA adapter `opencouncil/whisper-large-v3-el-council-lora`
merged into `whisper-large-v3` and converted to CTranslate2) is baked into the image
during the build, so a cold-started worker just loads it from local disk.

## Build

GitHub Actions builds the image and pushes it to
`ghcr.io/<owner>/oc-asr-serverless:latest` on every push to `main` (see
`.github/workflows/build.yml`). No local Docker needed. The ct2 model is built in CI
from the public adapter, so nothing large uploads from your machine.

## Deployed endpoint

There is a live endpoint on RunPod:

- **Endpoint ID:** `o1jda6sxo85dnk`
- **GPU:** 24GB Ampere pool, workers min 0 / max 1, 5s idle timeout
- **Image:** `ghcr.io/angelospk/oc-asr-serverless:latest`

It was created with the RunPod GraphQL API (`saveTemplate` then `saveEndpoint`);
`runpodctl` has no serverless subcommand. To recreate it, deploy a Serverless
endpoint from the image above on a 24GB GPU. RunPod protects the endpoint with your
account API key, so there is no separate app key.

Set the compute type to `float16`. The image ships `int8_float16` as the built model,
but that path throws `CUBLAS_STATUS_NOT_SUPPORTED` on RunPod's serverless CUDA image,
so the worker loads the int8 weights as float16 (set via the template env
`COMPUTE=float16`, which is also the current image default).

## Call it

Your RunPod API key lives in `~/.runpod/config.toml`:

```bash
RUNPOD_API_KEY=$(grep apikey ~/.runpod/config.toml | sed "s/.*'\(.*\)'.*/\1/")
```

Synchronous (waits for the result; good for short clips):

```bash
curl -X POST https://api.runpod.ai/v2/o1jda6sxo85dnk/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"audioUrl": "https://data.opencouncil.gr/audio/<segment>.mp3", "language": "el"}}'
```

Asynchronous (submit, then poll; better when a cold start may run long):

```bash
JID=$(curl -s -X POST https://api.runpod.ai/v2/o1jda6sxo85dnk/run \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"input":{"audioUrl":"https://.../segment.mp3","language":"el"}}' | jq -r .id)
curl -s https://api.runpod.ai/v2/o1jda6sxo85dnk/status/$JID \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
```

The response `output` is the OpenCouncil `Transcript` object (`metadata` plus
`transcription` with `full_transcript` and `utterances`), the same shape the oc-asr
HTTP server returns, so it is a drop-in for the tasks-server transcriber. Send an
already-cut segment URL, not a whole meeting. `speaker`, `channel`, and `drift` are 0
because diarization happens downstream (pyannote).

## This endpoint cannot be a benchmark provider as-is

The OpenCouncil benchmark (`bench.opencouncil.gr`) reaches a model through one of its
provider types: `openai-compatible` (a standard `POST {baseURL}/audio/transcriptions`
returning `{text}`), `hf-endpoint`, or `huggingface`. A RunPod Serverless endpoint
speaks none of those. It takes `POST https://api.runpod.ai/v2/<id>/run(sync)` with the
input wrapped in `{"input": {...}}`, authenticates with the RunPod account key, and
replies with RunPod's own `{"status", "output", ...}` envelope rather than the OpenAI
transcription shape. So you cannot point the benchmark at this endpoint directly.

Two ways to benchmark instead:

- Spin up a temporary RunPod GPU **pod** (not serverless) running the oc-asr HTTP
  server, which exposes the `openai-compatible` `/v1/audio/transcriptions` route the
  benchmark expects, register it as a provider, run, then terminate the pod. This is
  the cheapest per-run option (a full 260-clip run is roughly $0.35 of pod time).
- Or put a small `openai-compatible` shim in front of this endpoint that translates
  the OpenAI multipart request into a RunPod `run` call and maps the result back.

## Cost note

Billing is per GPU-second of execution only. Cold start adds roughly 20-40s to load
the model when a worker wakes from zero; warm workers respond immediately. For
async/batch transcription (which is how the tasks server uses it) that tradeoff is
fine.

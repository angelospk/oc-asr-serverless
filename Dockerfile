# Two-stage build:
#   builder — merge a PINNED revision of the public LoRA adapter into a PINNED
#             revision of whisper-large-v3 (CPU torch) and convert to CTranslate2.
#             Uses the exact version pins that are known to build this adapter cleanly.
#   runtime — slim CUDA image with only faster-whisper + the ct2 model + handler.
# The model is baked in, so a cold-started worker just loads it from local disk.
#
# BOTH revisions are full commit SHAs, not tags. A tag can be moved; a commit
# cannot. An output whose producing model is unknown is not evidence, and an
# earlier build of this image floated on `main` and served an adapter nobody
# can now identify.

FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 HF_HUB_DISABLE_TELEMETRY=1
RUN pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install transformers==5.6.2 peft==0.19.1 ctranslate2==4.8.1 accelerate==1.14.0

# whisper-large-v3 @ 06f233fe (2026-08-23)
ARG BASE=openai/whisper-large-v3
ARG BASE_REVISION=06f233fe06e710322aca913c1bc4249a0d71fce1
# opencouncil/whisper-large-v3-el-council-lora tag v2 == commit 1a03f207,
# the clean-pack contiguous seed-47 adapter. Tag v1 is the earlier
# single-utterance adapter, commit f620adc5.
ARG ADAPTER=opencouncil/whisper-large-v3-el-council-lora
ARG ADAPTER_REVISION=1a03f207e7e9bf46253715234a91eee98fa40505
ARG QUANTIZATION=float16

RUN python - "$BASE" "$BASE_REVISION" "$ADAPTER" "$ADAPTER_REVISION" <<'PY'
import json, sys, torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperFeatureExtractor
from peft import PeftModel

base_id, base_rev, adapter, adapter_rev = sys.argv[1:5]
for name, rev in (("base", base_rev), ("adapter", adapter_rev)):
    if len(rev) != 40 or not all(c in "0123456789abcdef" for c in rev):
        raise SystemExit(f"{name} revision must be a full 40-char commit sha, got {rev!r}")

base = WhisperForConditionalGeneration.from_pretrained(
    base_id, revision=base_rev, torch_dtype=torch.float32)
m = PeftModel.from_pretrained(base, adapter, revision=adapter_rev).merge_and_unload()
m.generation_config.language = "greek"; m.generation_config.task = "transcribe"
m.save_pretrained("/merged")
WhisperProcessor.from_pretrained(
    base_id, revision=base_rev, language="greek", task="transcribe").save_pretrained("/merged")
WhisperFeatureExtractor.from_pretrained(base_id, revision=base_rev).save_pretrained("/merged")

json.dump({"base": base_id, "base_revision": base_rev,
           "adapter": adapter, "adapter_revision": adapter_rev},
          open("/provenance.json", "w"), indent=1)
print("merged", base_id, base_rev[:8], "+", adapter, adapter_rev[:8])
PY

RUN ct2-transformers-converter --model /merged --output_dir /model/ct2 \
      --copy_files tokenizer.json preprocessor_config.json \
      --quantization "$QUANTIZATION" --force \
 && python - "$QUANTIZATION" <<'PY'
import hashlib, json, sys
p = json.load(open("/provenance.json"))
h = hashlib.sha256(open("/model/ct2/model.bin", "rb").read()).hexdigest()
p["quantization"] = sys.argv[1]
p["ct2_model_bin_sha256"] = h
p["converter"] = "ctranslate2==4.8.1"
json.dump(p, open("/model/provenance.json", "w"), indent=1)
print("ct2 model.bin sha256", h[:16])
PY
RUN rm -rf /merged /root/.cache

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS runtime
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 MODEL_DIR=/model/ct2 COMPUTE=float16 DEVICE=cuda
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip \
 && rm -rf /var/lib/apt/lists/*
RUN pip3 install faster-whisper==1.2.1 ctranslate2==4.8.1 runpod
COPY --from=builder /model /model
COPY handler.py /handler.py
CMD ["python3", "-u", "/handler.py"]

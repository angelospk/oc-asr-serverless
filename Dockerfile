# Two-stage build:
#   builder — merge the public LoRA adapter into whisper-large-v3 (CPU torch) and
#             convert to CTranslate2 int8_float16. Uses the exact version pins that
#             are known to build this adapter cleanly.
#   runtime — slim CUDA image with only faster-whisper + the ct2 model + handler.
# The model is baked in, so a cold-started worker just loads it from local disk.

FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 HF_HUB_DISABLE_TELEMETRY=1
RUN pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install transformers==5.6.2 peft==0.19.1 ctranslate2==4.8.1 accelerate==1.14.0
ARG BASE=openai/whisper-large-v3
ARG ADAPTER=opencouncil/whisper-large-v3-el-council-lora
RUN python - "$BASE" "$ADAPTER" <<'PY'
import sys, torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperFeatureExtractor
from peft import PeftModel
base_id, adapter = sys.argv[1], sys.argv[2]
base = WhisperForConditionalGeneration.from_pretrained(base_id, torch_dtype=torch.float32)
m = PeftModel.from_pretrained(base, adapter).merge_and_unload()
m.generation_config.language = "greek"; m.generation_config.task = "transcribe"
m.save_pretrained("/merged")
WhisperProcessor.from_pretrained(base_id, language="greek", task="transcribe").save_pretrained("/merged")
WhisperFeatureExtractor.from_pretrained(base_id).save_pretrained("/merged")
print("merged ok")
PY
RUN ct2-transformers-converter --model /merged --output_dir /model/ct2 \
      --copy_files tokenizer.json preprocessor_config.json --quantization int8_float16 --force \
 && rm -rf /merged /root/.cache

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS runtime
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 MODEL_DIR=/model/ct2 COMPUTE=float16 DEVICE=cuda
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip \
 && rm -rf /var/lib/apt/lists/*
RUN pip3 install faster-whisper==1.2.1 ctranslate2==4.8.1 runpod
COPY --from=builder /model /model
COPY handler.py /handler.py
CMD ["python3", "-u", "/handler.py"]

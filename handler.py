"""RunPod Serverless handler: fine-tuned Greek whisper-large-v3 council ASR.

Input  (event["input"]): {"audioUrl": "...", "language": "el"}  (or "audio_url")
Output: the OpenCouncil `Transcript` schema (metadata + transcription with
        full_transcript and utterances), identical to the oc-asr HTTP server.

The model (CTranslate2 int8_float16) is baked into the image at /model/ct2 and
loaded once per worker cold start. RunPod handles auth (endpoint API key), so
there is no separate app key here.
"""
import ipaddress
import math
import os
import socket
import tempfile
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import runpod
from faster_whisper import WhisperModel

MODEL_DIR = os.environ.get("MODEL_DIR", "/model/ct2")
COMPUTE = os.environ.get("COMPUTE", "int8_float16")
DEVICE = os.environ.get("DEVICE", "cuda")
BEAM = int(os.environ.get("BEAM", "2"))
LANGUAGE = os.environ.get("LANGUAGE", "el")
MAX_BYTES = int(os.environ.get("MAX_BYTES", str(500 * 1024 * 1024)))
CHUNK = 1 << 20

# Load once per worker (cold start). Kept at module scope so warm invocations reuse it.
_model = WhisperModel(MODEL_DIR, device=DEVICE, compute_type=COMPUTE)


def _validate_url(url: str) -> None:
    """Basic SSRF guard: only http(s), and the resolved host must be global."""
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise ValueError("audioUrl must be an http(s) URL")
    try:
        infos = socket.getaddrinfo(p.hostname, p.port or (443 if p.scheme == "https" else 80))
    except socket.gaierror as e:
        raise ValueError(f"cannot resolve host: {e}")
    for *_, sockaddr in infos:
        ip = ipaddress.ip_address(sockaddr[0])
        if not ip.is_global:
            raise ValueError("audioUrl resolves to a non-global address")


def _download(url: str) -> Path:
    _validate_url(url)
    suffix = Path(urlparse(url).path).suffix or ".mp3"
    fd, tmp = tempfile.mkstemp(prefix="oc_asr_", suffix=suffix)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "oc-asr-serverless"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=180) as r, os.fdopen(fd, "wb") as f:
            fd = None
            read = 0
            while True:
                chunk = r.read(CHUNK)
                if not chunk:
                    break
                read += len(chunk)
                if read > MAX_BYTES:
                    raise ValueError("audio exceeds size limit")
                f.write(chunk)
    except BaseException:
        if fd is not None:
            os.close(fd)
        Path(tmp).unlink(missing_ok=True)
        raise
    return Path(tmp)


def _transcribe(path: str, language: str) -> dict:
    t0 = time.time()
    utterances, full = [], []
    segments, info = _model.transcribe(
        path, language=language, word_timestamps=True, beam_size=BEAM,
        condition_on_previous_text=False)
    for seg in segments:
        words = [{
            "word": w.word,
            "start": round(float(w.start), 3),
            "end": round(float(w.end), 3),
            "confidence": round(float(w.probability), 4),
        } for w in (seg.words or [])]
        seg_conf = round(math.exp(seg.avg_logprob), 4) if seg.avg_logprob is not None else 0.0
        text = seg.text.strip()
        full.append(text)
        utterances.append({
            "text": text,
            "language": info.language or language,
            "start": round(float(seg.start), 3),
            "end": round(float(seg.end), 3),
            "confidence": seg_conf,
            "channel": 0,
            "speaker": 0,   # diarization is downstream (pyannote)
            "drift": 0,
            "words": words,
        })
    elapsed = time.time() - t0
    return {
        "metadata": {
            "audio_duration": round(float(info.duration), 3),
            "number_of_distinct_channels": 1,
            "billing_time": 0,
            "transcription_time": round(elapsed, 3),
            "notes": ("self-hosted whisper-large-v3+LoRA (CTranslate2 int8_float16, GPU); "
                      "speaker/channel/drift=0 (diarization downstream/pyannote)"),
        },
        "transcription": {
            "languages": [info.language or language],
            "full_transcript": " ".join(full).strip(),
            "utterances": utterances,
        },
    }


def handler(event):
    inp = event.get("input") or {}
    url = inp.get("audioUrl") or inp.get("audio_url")
    if not url:
        return {"error": "missing 'audioUrl' in input"}
    language = inp.get("language") or LANGUAGE
    path = None
    try:
        path = _download(url)
        return _transcribe(str(path), language)
    except Exception as e:
        return {"error": str(e)}
    finally:
        if path is not None:
            Path(path).unlink(missing_ok=True)


runpod.serverless.start({"handler": handler})

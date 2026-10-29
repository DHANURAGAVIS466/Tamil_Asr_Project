import os
import sys
import time
import re
import numpy as np
import soundfile as sf
import librosa
import torch
from datasets import load_dataset
from transformers import pipeline, AutoProcessor, AutoModelForCTC
from jiwer import wer, cer

# UTF-8 for console output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

TARGET_SAMPLE_RATE = 16000
os.makedirs("static/samples", exist_ok=True)

print("=" * 70, flush=True)
print("TAMIL ASR: REAL-TIME COMPARISON (WHISPER vs WAV2VEC2)", flush=True)
print("=" * 70, flush=True)

# 1. Load Whisper-Tiny
print("\n[1/4] Loading Whisper-Tiny model...", flush=True)
whisper_pipe = pipeline(
    task="automatic-speech-recognition",
    model="openai/whisper-tiny",
    device=-1
)
print("Whisper-Tiny loaded successfully.", flush=True)

# 2. Load Wav2Vec2 Tamil
print("\n[2/4] Loading Wav2Vec2 Tamil (Amrrs/wav2vec2-large-xlsr-53-tamil)...", flush=True)
wav2vec_model_id = "Amrrs/wav2vec2-large-xlsr-53-tamil"
wav2vec_processor = AutoProcessor.from_pretrained(wav2vec_model_id)
wav2vec_model = AutoModelForCTC.from_pretrained(wav2vec_model_id)
wav2vec_model.eval()
print("Wav2Vec2 Tamil loaded successfully.", flush=True)

# 3. Stream 3 real speech samples
print("\n[3/4] Streaming 3 real human speech samples from dataset...", flush=True)
try:
    dataset = load_dataset("SPRINGLab/IndicVoices-R_Tamil", split="train", streaming=True)
except Exception:
    dataset = load_dataset("deepdml/openslr65-tamil", split="train", streaming=True)

samples = []
for idx, item in enumerate(dataset):
    audio_info = item.get("audio", {})
    if isinstance(audio_info, dict):
        arr = audio_info.get("array")
        sr = audio_info.get("sampling_rate", 16000)
    else:
        arr = audio_info
        sr = 16000

    ref = (
        item.get("normalized")
        or item.get("text")
        or item.get("transcription")
        or item.get("normalized_text")
        or item.get("sentence")
        or ""
    )

    if arr is None or len(arr) < int(sr * 1.5):
        continue

    # Resample to 16kHz
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim > 1:
        arr = np.mean(arr, axis=0)
    if sr != TARGET_SAMPLE_RATE:
        arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)

    # Save to disk
    wav_path = f"static/samples/real_tamil_sample_{len(samples)+1}.wav"
    sf.write(wav_path, arr, TARGET_SAMPLE_RATE)

    samples.append({
        "id": f"sample_{len(samples)+1}",
        "path": wav_path,
        "audio": arr,
        "reference": str(ref).strip(),
        "duration": len(arr) / TARGET_SAMPLE_RATE
    })
    print(f"Extracted sample {len(samples)}: {len(arr)/TARGET_SAMPLE_RATE:.2f}s | Ref: {ref[:40]}...", flush=True)
    if len(samples) >= 3:
        break

# 4. Transcribe each sample with BOTH models
print("\n[4/4] Transcribing and comparing Whisper vs Wav2Vec2...\n", flush=True)

for s in samples:
    audio_array = s["audio"]
    ref = s["reference"]
    dur = s["duration"]

    print("=" * 70, flush=True)
    print(f"SAMPLE {s['id']} (Duration: {dur:.2f}s)", flush=True)
    print(f"REFERENCE : {ref}", flush=True)

    # Whisper
    t0 = time.time()
    w_out = whisper_pipe(
        {"raw": audio_array, "sampling_rate": TARGET_SAMPLE_RATE},
        return_timestamps=True,
        chunk_length_s=30,
        generate_kwargs={"language": "tamil", "task": "transcribe", "no_repeat_ngram_size": 3}
    )
    t_whisper = time.time() - t0
    w_text = w_out.get("text", "").strip() if isinstance(w_out, dict) else str(w_out).strip()

    # Wav2Vec2
    t0 = time.time()
    inputs = wav2vec_processor(
        audio_array,
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
        padding=True
    )
    with torch.no_grad():
        logits = wav2vec_model(inputs.input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    v_text = wav2vec_processor.batch_decode(pred_ids)[0].strip()
    t_wav2vec = time.time() - t0

    # Calculate WER / CER if reference exists
    w_wer = wer(ref, w_text) if ref and w_text else 0.0
    w_cer = cer(ref, w_text) if ref and w_text else 0.0
    v_wer = wer(ref, v_text) if ref and v_text else 0.0
    v_cer = cer(ref, v_text) if ref and v_text else 0.0

    print(f"\n[Whisper-Tiny] ({t_whisper:.2f}s):", flush=True)
    print(f"  OUTPUT : {w_text}", flush=True)
    print(f"  METRICS: WER = {w_wer*100:.1f}% | CER = {w_cer*100:.1f}%", flush=True)

    print(f"\n[Wav2Vec2 Tamil] ({t_wav2vec:.2f}s):", flush=True)
    print(f"  OUTPUT : {v_text}", flush=True)
    print(f"  METRICS: WER = {v_wer*100:.1f}% | CER = {v_cer*100:.1f}%", flush=True)
    print("=" * 70, flush=True)

print("\nDual evaluation complete!", flush=True)

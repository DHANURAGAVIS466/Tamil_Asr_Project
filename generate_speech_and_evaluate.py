import os
import sys
import io
import time
import numpy as np
import soundfile as sf
import librosa
import torch
from gtts import gTTS
from transformers import pipeline, AutoProcessor, AutoModelForCTC
from jiwer import wer, cer

# Suppress warnings & setup UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
TARGET_SR = 16000
os.makedirs("static/samples", exist_ok=True)

BENCHMARK_SENTENCES = [
    {
        "id": "sample-1",
        "title": "Government & Welfare",
        "filename": "sample_tamil_01.wav",
        "text": "தமிழக அரசு புதிய நலத்திட்டங்களை அறிவித்துள்ளது"
    },
    {
        "id": "sample-2",
        "title": "Education & Computing",
        "filename": "sample_tamil_02.wav",
        "text": "மாணவர்கள் அனைவரும் கணினி அறிவியல் பாடத்தை ஆர்வத்துடன் கற்கின்றனர்"
    },
    {
        "id": "sample-3",
        "title": "Agriculture & Irrigation",
        "filename": "sample_tamil_03.wav",
        "text": "விவசாயிகளின் பாசன வசதிக்காக அணைகளில் இருந்து தண்ணீர் திறக்கப்பட்டுள்ளது"
    },
    {
        "id": "sample-4",
        "title": "Healthcare Camps",
        "filename": "sample_tamil_04.wav",
        "text": "கிராமப்புற சுகாதார நிலையங்களில் இலவச மருத்துவ பரிசோதனை முகாம் நடைபெற்றது"
    }
]

print("=" * 70, flush=True)
print("1. GENERATING REAL TAMIL SPOKEN AUDIO SAMPLES (gTTS Tamil Voice)", flush=True)
print("=" * 70, flush=True)

processed_samples = []

for item in BENCHMARK_SENTENCES:
    filepath = os.path.join("static/samples", item["filename"])
    print(f"Generating spoken Tamil speech for: '{item['text']}'...", flush=True)
    tts = gTTS(text=item["text"], lang="ta", slow=False)
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)

    # Load with librosa & resample to 16kHz mono
    audio_arr, sr = librosa.load(mp3_fp, sr=TARGET_SR, mono=True)
    audio_arr = audio_arr.astype(np.float32)

    # Normalize audio
    max_amp = float(np.max(np.abs(audio_arr)))
    if max_amp > 1e-4:
        audio_arr = (audio_arr / max_amp) * 0.95

    # Save to disk as clean 16kHz WAV
    sf.write(filepath, audio_arr, TARGET_SR)
    dur = len(audio_arr) / TARGET_SR
    print(f"  -> Saved {filepath} (Duration: {dur:.2f}s)", flush=True)

    processed_samples.append({
        "id": item["id"],
        "title": item["title"],
        "reference": item["text"],
        "path": filepath,
        "audio": audio_arr,
        "duration": dur
    })

print("\nAll spoken Tamil audio samples generated and saved successfully!", flush=True)

print("\n" + "=" * 70, flush=True)
print("2. LOADING MODELS INTO MEMORY (Whisper-Tiny & Wav2Vec2 Tamil)", flush=True)
print("=" * 70, flush=True)

print("Loading Whisper-Tiny...", flush=True)
whisper_pipe = pipeline(
    task="automatic-speech-recognition",
    model="openai/whisper-tiny",
    device=-1
)
print("Whisper-Tiny ready.", flush=True)

print("Loading Wav2Vec2 Tamil...", flush=True)
wav2vec_id = "Amrrs/wav2vec2-large-xlsr-53-tamil"
wav2vec_processor = AutoProcessor.from_pretrained(wav2vec_id)
wav2vec_model = AutoModelForCTC.from_pretrained(wav2vec_id)
wav2vec_model.eval()
print("Wav2Vec2 Tamil ready.", flush=True)

print("\n" + "=" * 70, flush=True)
print("3. RUNNING DUAL INFERENCE AND COMPARING OUTPUTS", flush=True)
print("=" * 70, flush=True)

for sample in processed_samples:
    ref = sample["reference"]
    arr = sample["audio"]
    dur = sample["duration"]

    print(f"\n--- {sample['title']} ({dur:.2f}s) ---", flush=True)
    print(f"GROUND TRUTH: {ref}", flush=True)

    # 1. Whisper Inference
    t0 = time.time()
    w_out = whisper_pipe(
        {"raw": arr, "sampling_rate": TARGET_SR},
        generate_kwargs={"language": "tamil", "task": "transcribe"}
    )
    t_whisper = time.time() - t0
    w_text = w_out.get("text", "").strip() if isinstance(w_out, dict) else str(w_out).strip()

    # 2. Wav2Vec2 Inference
    t0 = time.time()
    inputs = wav2vec_processor(arr, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = wav2vec_model(inputs.input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    v_text = wav2vec_processor.batch_decode(pred_ids)[0].strip()
    t_wav2vec = time.time() - t0

    # Metrics
    w_wer = wer(ref, w_text) if ref and w_text else 0.0
    w_cer = cer(ref, w_text) if ref and w_text else 0.0
    v_wer = wer(ref, v_text) if ref and v_text else 0.0
    v_cer = cer(ref, v_text) if ref and v_text else 0.0

    print(f"\n[Whisper-Tiny] Latency: {t_whisper:.2f}s", flush=True)
    print(f"  OUTPUT : {w_text}", flush=True)
    print(f"  WER    : {w_wer*100:.1f}% | CER: {w_cer*100:.1f}%", flush=True)

    print(f"\n[Wav2Vec2 Tamil] Latency: {t_wav2vec:.2f}s", flush=True)
    print(f"  OUTPUT : {v_text}", flush=True)
    print(f"  WER    : {v_wer*100:.1f}% | CER: {v_cer*100:.1f}%", flush=True)
    print("-" * 70, flush=True)

print("\nAll 4 speech benchmarks completed!", flush=True)

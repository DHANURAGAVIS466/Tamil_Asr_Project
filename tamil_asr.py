import os
import sys
import re
import warnings

# Configure UTF-8 for console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Configure environment for Hugging Face and suppress warnings
os.environ["HF_HUB_HTTP_TIMEOUT"] = "120"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import librosa
import torch

from datasets import load_dataset
from transformers import (
    pipeline,
    AutoProcessor,
    AutoModelForCTC
)

from jiwer import wer, cer


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_NAME = "SPRINGLab/IndicVoices-R_Tamil"

WHISPER_MODEL = "openai/whisper-tiny"

WAV2VEC_MODEL = "Amrrs/wav2vec2-large-xlsr-53-tamil"

TARGET_MINUTES = 10

TARGET_SAMPLE_RATE = 16000

OUTPUT_DIR = "results"

# Set this to True if you want to run Wav2Vec2.
# Wav2Vec2-Large can be slow on CPU.
RUN_WAV2VEC2 = False


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():
    DEVICE = 0
    DEVICE_NAME = "CUDA GPU"
else:
    DEVICE = -1
    DEVICE_NAME = "CPU"


print("=" * 70)
print("TAMIL ASR EXPERIMENT")
print("=" * 70)

print(f"Device: {DEVICE_NAME}")
print(f"Dataset: {DATASET_NAME}")
print(f"Target duration: {TARGET_MINUTES} minutes")

if DEVICE == -1:
    print("\nWARNING:")
    print("No GPU detected.")
    print("The models will run on CPU and may be slow.")

print("=" * 70)


# ============================================================
# LOAD DATASET
# ============================================================

print("\n[1/7] Loading Tamil dataset...")

print("Using streaming mode so the complete dataset is not downloaded.")

try:
    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True
    )
except Exception as e:
    print(f"Notice loading {DATASET_NAME}: {e}")
    print("Falling back to deepdml/openslr65-tamil...")
    DATASET_NAME = "deepdml/openslr65-tamil"
    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True
    )


# ============================================================
# SELECT APPROXIMATELY 10 MINUTES
# ============================================================

print("\n[2/7] Selecting approximately 10 minutes of speech...")

target_seconds = TARGET_MINUTES * 60

selected_samples = []

total_seconds = 0.0

def extract_sample_info(item, idx):
    audio_info = item.get("audio", {})
    if isinstance(audio_info, dict):
        audio_array = audio_info.get("array")
        sr = audio_info.get("sampling_rate", 16000)
    else:
        audio_array = audio_info
        sr = 16000

    text = (
        item.get("normalized")
        or item.get("text")
        or item.get("transcription")
        or item.get("normalized_text")
        or item.get("sentence")
        or item.get("transcript")
        or ""
    )

    if "duration" in item and item["duration"] is not None:
        duration = float(item["duration"])
    elif audio_array is not None and len(audio_array) > 0:
        duration = float(len(audio_array)) / float(sr)
    else:
        duration = 0.0

    file_id = (
        item.get("file_id")
        or item.get("id")
        or (audio_info.get("path") if isinstance(audio_info, dict) else None)
        or f"sample_{idx + 1}"
    )

    return {
        "file_id": str(file_id),
        "duration": duration,
        "reference": str(text).strip(),
        "audio_array": audio_array,
        "sampling_rate": sr
    }

for index, item in enumerate(dataset):
    sample = extract_sample_info(item, index)
    duration = sample["duration"]

    # Skip unusually long, empty or invalid samples
    if duration <= 0.2 or sample["audio_array"] is None or len(sample["audio_array"]) == 0:
        continue

    selected_samples.append(sample)
    total_seconds += duration

    if total_seconds >= target_seconds:
        break


print(f"Selected samples: {len(selected_samples)}")
print(f"Total duration: {total_seconds:.2f} seconds")
print(f"Total duration: {total_seconds / 60:.2f} minutes")


if len(selected_samples) == 0:
    raise RuntimeError("No audio samples were loaded.")


# ============================================================
# AUDIO PREPROCESSING
# ============================================================

def preprocess_audio(audio_array, sampling_rate):
    """
    Convert audio to mono float32 and resample to 16 kHz.
    """

    audio_array = np.asarray(audio_array, dtype=np.float32)

    # Convert stereo to mono
    if audio_array.ndim > 1:
        audio_array = np.mean(audio_array, axis=0)

    # Resample if required
    if sampling_rate != TARGET_SAMPLE_RATE and len(audio_array) > 0:

        audio_array = librosa.resample(
            audio_array,
            orig_sr=sampling_rate,
            target_sr=TARGET_SAMPLE_RATE
        )

    return audio_array.astype(np.float32)


# ============================================================
# LOAD WHISPER
# ============================================================

print("\n[3/7] Loading Whisper model...")

whisper = pipeline(
    task="automatic-speech-recognition",
    model=WHISPER_MODEL,
    device=DEVICE
)

print("Whisper loaded successfully.")


# ============================================================
# WHISPER TRANSCRIPTION
# ============================================================

print("\n[4/7] Running Whisper transcription...")

results = []

for index, item in enumerate(selected_samples):

    print(
        f"Whisper: {index + 1}/{len(selected_samples)}",
        end="\r",
        flush=True
    )

    audio_array = preprocess_audio(
        item["audio_array"],
        item["sampling_rate"]
    )

    try:

        output = whisper(
            {
                "raw": audio_array,
                "sampling_rate": TARGET_SAMPLE_RATE
            },
            return_timestamps=True,
            chunk_length_s=30,
            generate_kwargs={
                "language": "tamil",
                "task": "transcribe"
            }
        )

        if isinstance(output, dict):
            prediction = output.get("text", "").strip()
        elif isinstance(output, list) and len(output) > 0:
            prediction = " ".join([x.get("text", "").strip() for x in output if isinstance(x, dict)])
        else:
            prediction = str(output).strip()

    except Exception as error:

        print(f"\nWhisper error on sample {index}: {error}")

        prediction = ""

    results.append(
        {
            "file_id": item["file_id"],
            "duration": item["duration"],
            "reference": item["reference"],
            "whisper_prediction": prediction,
            "wav2vec_prediction": ""
        }
    )


print("\nWhisper transcription completed.")


# ============================================================
# LOAD WAV2VEC2
# ============================================================

if RUN_WAV2VEC2:

    print("\n[5/7] Loading Wav2Vec2 Tamil model...")

    print(
        "NOTE: Wav2Vec2-Large is much heavier than Whisper-Tiny "
        "and can be slow on CPU."
    )

    processor = AutoProcessor.from_pretrained(
        WAV2VEC_MODEL
    )

    wav2vec_model = AutoModelForCTC.from_pretrained(
        WAV2VEC_MODEL
    )

    wav2vec_model.eval()

    print("Wav2Vec2 loaded successfully.")


    # ========================================================
    # WAV2VEC2 TRANSCRIPTION
    # ========================================================

    print("\nRunning Wav2Vec2 transcription...")

    for index, item in enumerate(selected_samples):

        print(
            f"Wav2Vec2: {index + 1}/{len(selected_samples)}",
            end="\r",
            flush=True
        )

        audio_array = preprocess_audio(
            item["audio_array"],
            item["sampling_rate"]
        )

        try:

            inputs = processor(
                audio_array,
                sampling_rate=TARGET_SAMPLE_RATE,
                return_tensors="pt",
                padding=True
            )

            with torch.no_grad():

                logits = wav2vec_model(
                    inputs.input_values
                ).logits

            predicted_ids = torch.argmax(
                logits,
                dim=-1
            )

            prediction = processor.batch_decode(
                predicted_ids
            )[0].strip()

            results[index]["wav2vec_prediction"] = prediction

        except Exception as error:

            print(
                f"\nWav2Vec2 error on sample {index}: {error}"
            )

            results[index]["wav2vec_prediction"] = ""


    print("\nWav2Vec2 transcription completed.")


# ============================================================
# BASIC TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Basic normalization for WER/CER calculation.
    """

    if text is None:
        return ""

    text = str(text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# CALCULATE METRICS
# ============================================================

print("\n[6/7] Calculating WER and CER...")


references = [
    normalize_text(x["reference"])
    for x in results
]


whisper_predictions = [
    normalize_text(x["whisper_prediction"])
    for x in results
]


whisper_wer = wer(
    references,
    whisper_predictions
)


whisper_cer = cer(
    references,
    whisper_predictions
)


print("\n" + "=" * 70)
print("WHISPER RESULTS")
print("=" * 70)

print(f"WER: {whisper_wer:.4f}")
print(f"WER (%): {whisper_wer * 100:.2f}%")

print(f"CER: {whisper_cer:.4f}")
print(f"CER (%): {whisper_cer * 100:.2f}%")


# ============================================================
# WAV2VEC2 METRICS
# ============================================================

if RUN_WAV2VEC2:

    wav2vec_predictions = [
        normalize_text(x["wav2vec_prediction"])
        for x in results
    ]

    wav2vec_wer = wer(
        references,
        wav2vec_predictions
    )

    wav2vec_cer = cer(
        references,
        wav2vec_predictions
    )

    print("\n" + "=" * 70)
    print("WAV2VEC2 RESULTS")
    print("=" * 70)

    print(f"WER: {wav2vec_wer:.4f}")
    print(f"WER (%): {wav2vec_wer * 100:.2f}%")

    print(f"CER: {wav2vec_cer:.4f}")
    print(f"CER (%): {wav2vec_cer * 100:.2f}%")


# ============================================================
# SAVE RESULTS
# ============================================================

print("\n[7/7] Saving results...")

results_df = pd.DataFrame(results)

output_file = os.path.join(
    OUTPUT_DIR,
    "tamil_asr_results.csv"
)

results_df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nResults saved to:")
print(output_file)


# ============================================================
# DISPLAY SAMPLE RESULTS
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE TRANSCRIPTIONS")
print("=" * 70)

def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        try:
            print(str(text).encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8"))
        except Exception:
            print(repr(text))

for i, result in enumerate(results[:5]):
    print(f"\nSample {i + 1}")
    print("-" * 50)

    print("REFERENCE:")
    safe_print(result["reference"])

    print("\nWHISPER:")
    safe_print(result["whisper_prediction"])

    if RUN_WAV2VEC2:
        print("\nWAV2VEC2:")
        safe_print(result["wav2vec_prediction"])

print("\n" + "=" * 70)
print("EXPERIMENT COMPLETED")
print("=" * 70)

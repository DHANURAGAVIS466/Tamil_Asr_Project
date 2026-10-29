"""
Fine-tuning Whisper Speech-to-Text on SPRINGLab/IndicVoices-R_Tamil dataset.
Supports both CPU and CUDA GPU environments.
"""

import os
import sys
import warnings
import dataclasses
from typing import Any, Dict, List, Union
import torch
import numpy as np
import librosa
from datasets import load_dataset, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

# Suppress warnings & configure UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
DATASET_NAME = "SPRINGLab/IndicVoices-R_Tamil"
MODEL_NAME = "openai/whisper-tiny"
OUTPUT_DIR = "./tamil_asr_model"
NUM_TRAIN_SAMPLES = 50   # Adjust as needed (e.g. 100, 500, 1000)
NUM_EVAL_SAMPLES = 10
BATCH_SIZE = 4 if torch.cuda.is_available() else 2
LEARNING_RATE = 1e-5
NUM_EPOCHS = 3
TARGET_SAMPLE_RATE = 16000

print("=" * 70)
print("TAMIL ASR TRAINING PIPELINE")
print("=" * 70)
print(f"Dataset: {DATASET_NAME}")
print(f"Base Model: {MODEL_NAME}")
print(f"Device: {'CUDA GPU' if torch.cuda.is_available() else 'CPU'}")
print(f"Output Directory: {OUTPUT_DIR}")
print("=" * 70)

# ============================================================
# 1. LOAD PROCESSOR & MODEL
# ============================================================
print("\n[1/5] Loading Feature Extractor, Tokenizer & Model...")
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, language="tamil", task="transcribe")
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="tamil", task="transcribe")

model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.use_cache = False

# ============================================================
# 2. LOAD & PREPARE DATASET
# ============================================================
def load_and_collect_samples(dataset_name, num_train, num_eval):
    print(f"\n[2/5] Streaming {dataset_name} dataset...")
    collected_train = []
    collected_eval = []
    
    ds = None
    # 1. Direct parquet data_files for IndicVoices
    if "IndicVoices" in dataset_name:
        try:
            ds = load_dataset(
                dataset_name,
                data_files={"train": "data/train-00000-of-00014.parquet"},
                split="train",
                streaming=True
            )
        except Exception as e:
            print(f"Direct parquet data_files notice: {e}")
            
    # 2. Standard stream fallback
    if ds is None:
        try:
            ds = load_dataset(dataset_name, split="train", streaming=True)
        except Exception as e:
            print(f"Standard stream notice: {e}")
            
    if ds is None:
        return collected_train, collected_eval
        
    try:
        iterator = iter(ds)
        idx = 0
        consecutive_errors = 0
        
        while len(collected_train) < num_train or len(collected_eval) < num_eval:
            try:
                item = next(iterator)
                consecutive_errors = 0
                arr, sr, txt = extract_sample(item, idx)
                idx += 1
                
                if arr is None or len(arr) == 0 or not txt:
                    continue
                
                # Resample to 16kHz
                arr = np.asarray(arr, dtype=np.float32)
                if arr.ndim > 1:
                    arr = np.mean(arr, axis=0)
                if sr != TARGET_SAMPLE_RATE and len(arr) > 0:
                    arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
                
                # Filter out overly long audio for memory efficiency (>15 sec)
                if len(arr) > TARGET_SAMPLE_RATE * 15:
                    arr = arr[:TARGET_SAMPLE_RATE * 15]
                    
                input_features = feature_extractor(arr, sampling_rate=TARGET_SAMPLE_RATE).input_features[0]
                labels = tokenizer(txt).input_ids
                
                sample_dict = {
                    "input_features": input_features,
                    "labels": labels,
                    "raw_audio": arr,
                    "reference": txt
                }
                
                if len(collected_train) < num_train:
                    collected_train.append(sample_dict)
                    print(f"Loaded Train Sample: {len(collected_train)}/{num_train}", end="\r", flush=True)
                elif len(collected_eval) < num_eval:
                    collected_eval.append(sample_dict)
                    print(f"Loaded Eval Sample: {len(collected_eval)}/{num_eval}", end="\r", flush=True)
                    
            except StopIteration:
                break
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors > 5:
                    print(f"\nStream completed/interrupted ({e}). Proceeding with {len(collected_train)} samples.")
                    break
    except Exception as e:
        print(f"\nCould not stream {dataset_name}: {e}")
        
    return collected_train, collected_eval

def extract_sample(item, idx):
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
    return audio_array, sr, str(text).strip()

CACHE_TRAIN = "cached_indicvoices_train.pt"
CACHE_EVAL = "cached_indicvoices_eval.pt"

if os.path.exists(CACHE_TRAIN) and os.path.exists(CACHE_EVAL):
    print(f"\n[2/5] Loading preprocessed samples from local cache ({CACHE_TRAIN})...")
    train_items = torch.load(CACHE_TRAIN, weights_only=False)
    eval_items = torch.load(CACHE_EVAL, weights_only=False)
    print(f"Loaded {len(train_items)} train samples, {len(eval_items)} eval samples from cache.")
else:
    print(f"Collecting {NUM_TRAIN_SAMPLES} training and {NUM_EVAL_SAMPLES} validation samples...")
    train_items, eval_items = load_and_collect_samples(DATASET_NAME, NUM_TRAIN_SAMPLES, NUM_EVAL_SAMPLES)

    if len(train_items) < 10:
        print("\nAttempting backup dataset: deepdml/openslr65-tamil...")
        fallback_train, fallback_eval = load_and_collect_samples("deepdml/openslr65-tamil", NUM_TRAIN_SAMPLES, NUM_EVAL_SAMPLES)
        if fallback_train:
            train_items = fallback_train
            eval_items = fallback_eval

    if len(train_items) > 0:
        torch.save(train_items, CACHE_TRAIN)
        torch.save(eval_items, CACHE_EVAL)

print(f"\nSuccessfully prepared {len(train_items)} train samples, {len(eval_items)} eval samples.")

if len(train_items) == 0:
    raise RuntimeError("Could not load sufficient samples for training.")

# ============================================================
# 3. DATA COLLATOR
# ============================================================
@dataclasses.dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# ============================================================
# 4. TRAINING ARGUMENTS & TRAINER
# ============================================================
print("\n[3/5] Setting up Training Arguments...")
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=2,
    learning_rate=LEARNING_RATE,
    warmup_steps=5,
    max_steps=25 if not torch.cuda.is_available() else 100,
    gradient_checkpointing=torch.cuda.is_available(),
    fp16=torch.cuda.is_available(),
    save_steps=25,
    logging_steps=5,
    report_to=["none"],
    load_best_model_at_end=False,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_items,
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
)

print("\n[4/5] Starting Training Loop...")
trainer.train()

print(f"\n[5/5] Saving fine-tuned model and processor to {OUTPUT_DIR}...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Run Post-Training Evaluation
if len(eval_items) > 0:
    print("\nRunning Evaluation on Validation Set...")
    try:
        from jiwer import wer, cer
        refs = []
        hyps = []
        model.eval()
        for s in eval_items[:10]:
            refs.append(s["reference"])
            inputs = processor(s["raw_audio"], sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt")
            with torch.no_grad():
                pred_ids = model.generate(inputs.input_features, language="tamil", task="transcribe")
            hyp_text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
            hyps.append(hyp_text)
            
        final_wer = wer(refs, hyps) * 100
        final_cer = cer(refs, hyps) * 100
        print(f"Final Validation WER: {final_wer:.2f}% | CER: {final_cer:.2f}%")
    except Exception as e:
        print(f"Eval note: {e}")

print("\n" + "=" * 70)
print("TAMIL ASR TRAINING COMPLETE!")
print(f"Trained model saved to: {os.path.abspath(OUTPUT_DIR)}")
print("=" * 70)

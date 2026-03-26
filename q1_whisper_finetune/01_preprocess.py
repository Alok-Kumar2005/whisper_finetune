# %%writefile q1_whisper_finetune/01_preprocess.py
import os
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from datasets import Dataset, DatasetDict, Audio
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.data_loader import download_dataset
from shared.utils import setup_logging, normalize_hindi_text

setup_logging()
logger = logging.getLogger(__name__)

# Config - Assuming you upload the file as 'dataset.csv'
METADATA_CSV      = "dataset.csv"
DATA_DIR          = Path("q1_whisper_finetune/data")
PROCESSED_DIR     = Path("q1_whisper_finetune/data/processed")
HF_DATASET_DIR    = Path("q1_whisper_finetune/data/hf_dataset")

TARGET_SR         = 16_000
MAX_DURATION_SEC  = 30.0
MIN_DURATION_SEC  = 0.5
MIN_WORDS         = 2
SILENCE_THRESHOLD = 0.005
TRAIN_RATIO       = 0.80
VAL_RATIO         = 0.10
RANDOM_SEED       = 42

def step1_download(csv_path: str) -> pd.DataFrame:
    logger.info("=== Step 1: Downloading dataset ===")
    if (DATA_DIR / "manifest.csv").exists():
        return pd.read_csv(DATA_DIR / "manifest.csv")
    return download_dataset(
        tsv_path=csv_path,
        output_dir=str(DATA_DIR),
        download_audio=True,
        download_transcription=True,
        download_metadata=True,
    )

def is_silent(audio: np.ndarray, threshold: float = SILENCE_THRESHOLD) -> bool:
    rms = np.sqrt(np.mean(audio ** 2))
    return rms < threshold

def validate_audio(audio_path: str) -> tuple:
    try:
        audio, sr = sf.read(audio_path)
    except Exception as exc:
        return False, f"read error: {exc}", None, None

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    duration = len(audio) / sr
    if duration < MIN_DURATION_SEC:
        return False, f"too short ({duration:.2f}s)", audio, sr
    if is_silent(audio):
        return False, "silence", audio, sr

    return True, "ok", audio, sr

def segment_audio(audio: np.ndarray, sr: int, transcript: str, recording_id: str, out_dir: Path) -> list:
    duration = len(audio) / sr
    out_dir.mkdir(parents=True, exist_ok=True)

    if duration <= MAX_DURATION_SEC:
        seg_path = out_dir / f"{recording_id}_seg0.wav"
        sf.write(str(seg_path), audio, sr)
        return [{"audio_path": str(seg_path), "transcript": transcript, "duration": duration}]

    chunk_samples = int(MAX_DURATION_SEC * sr)
    segments = []
    for i, start in enumerate(range(0, len(audio), chunk_samples)):
        chunk = audio[start: start + chunk_samples]
        if len(chunk) / sr < MIN_DURATION_SEC:
            continue
        seg_path = out_dir / f"{recording_id}_seg{i}.wav"
        sf.write(str(seg_path), chunk, sr)
        segments.append({
            "audio_path": str(seg_path),
            "transcript": transcript,
            "duration": len(chunk) / sr,
        })
    return segments

def build_hf_dataset(records: list) -> DatasetDict:
    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    n = len(df)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    splits = {
        "train":      df.iloc[:n_train],
        "validation": df.iloc[n_train: n_train + n_val],
        "test":       df.iloc[n_train + n_val:],
    }

    datasets = {}
    for split_name, split_df in splits.items():
        hf_ds = Dataset.from_dict({
            "audio":       split_df["audio_path"].tolist(),
            "sentence":    split_df["transcript"].tolist(),
            "duration":    split_df["duration"].tolist(),
        })
        hf_ds = hf_ds.cast_column("audio", Audio(sampling_rate=TARGET_SR))
        datasets[split_name] = hf_ds

    return DatasetDict(datasets)

def main():
    manifest_df = step1_download(METADATA_CSV)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_records = []
    for _, row in tqdm(manifest_df.iterrows(), total=len(manifest_df)):
        audio_path = row.get("audio_path", "")
        transcript = row.get("transcript_text", "")

        if not isinstance(transcript, str) or len(transcript.split()) < MIN_WORDS:
            continue
        transcript = normalize_hindi_text(transcript)

        valid, reason, audio, sr = validate_audio(audio_path)
        if not valid:
            continue

        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR) if sr != TARGET_SR else audio
        rec_id = str(row.get("recording_id", Path(audio_path).stem))
        segs = segment_audio(audio, TARGET_SR, transcript, rec_id, PROCESSED_DIR / rec_id)
        all_records.extend(segs)

    dataset = build_hf_dataset(all_records)
    dataset.save_to_disk(str(HF_DATASET_DIR))
    logger.info(f"Dataset saved to {HF_DATASET_DIR}")

if __name__ == "__main__":
    main()
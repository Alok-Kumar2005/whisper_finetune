import sys
import logging
import requests
import json
import time
import torch
import pandas as pd
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Paths
ORIGINAL_DATASET_CSV = Path("dataset.csv") 
SAMPLE_DIR = Path("q2_asr_cleanup/sample_audio")
OUTPUT_MANIFEST = Path("q2_asr_cleanup/raw_asr_manifest.csv")
MODEL_NAME = "openai/whisper-small"
NUM_SAMPLES = 5 # 5 files is perfect!

def download_file(url: str, dest: Path):
    if dest.exists(): return True
    try:
        response = requests.get(url, stream=True, timeout=15)
        response.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
        return True
    except Exception as e:
        logger.warning(f"Failed to download {url}: {e}")
        return False

def load_transcription(json_path: Path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key in ("transcript", "transcription", "text", "sentence"):
            if key in data: return data[key]
        return ""
    except Exception:
        return ""

def main():
    if not ORIGINAL_DATASET_CSV.exists():
        logger.error(f"Could not find {ORIGINAL_DATASET_CSV} in the root folder!")
        return

    logger.info("Loading original dataset and extracting a sample...")
    df = pd.read_csv(ORIGINAL_DATASET_CSV).head(NUM_SAMPLES).copy()
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    
    records = []
    
    logger.info(f"Checking {NUM_SAMPLES} sample files...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading"):
        rec_id = row["recording_id"]
        
        audio_url = str(row["rec_url_gcp"]).replace("joshtalks-data-collection/hq_data/hi", "upload_goai")
        trans_url = str(row["transcription_url_gcp"]).replace("joshtalks-data-collection/hq_data/hi", "upload_goai")
        
        audio_path = SAMPLE_DIR / f"{rec_id}_audio.wav"
        trans_path = SAMPLE_DIR / f"{rec_id}_transcription.json"
        
        if download_file(audio_url, audio_path) and download_file(trans_url, trans_path):
            ref_text = load_transcription(trans_path)
            records.append({
                "recording_id": rec_id,
                "audio_path": str(audio_path),
                "transcript_text": ref_text
            })
            
    sample_df = pd.DataFrame(records)
    
    logger.info(f"Loading baseline model {MODEL_NAME}...")
    device = 0 if torch.cuda.is_available() else -1
    
    # FIX: Use batch_size 1 for local CPU so it doesn't freeze your RAM
    asr_pipe = pipeline(
        "automatic-speech-recognition", 
        model=MODEL_NAME, 
        device=device,
        chunk_length_s=30,
        batch_size=1 
    )
    
    asr_pipe.model.generation_config.forced_decoder_ids = None
    asr_pipe.model.generation_config.suppress_tokens = []
    asr_pipe.model.generation_config.language = "hindi"
    asr_pipe.model.generation_config.task = "transcribe"

    raw_asr_results = []
    def data_generator():
        for path in sample_df["audio_path"]:
            try:
                audio, sr = sf.read(path)
                if audio.ndim > 1: audio = audio.mean(axis=1)
                
                # THE SPEED FIX: Chop the audio to ONLY the first 30 seconds
                max_samples = sr * 30
                if len(audio) > max_samples:
                    audio = audio[:max_samples]
                    
                yield {"array": audio.astype(np.float32), "sampling_rate": sr}
            except Exception:
                yield {"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000}

    logger.info("Generating raw ASR transcripts for the 30-second clips...")
    for out in tqdm(asr_pipe(data_generator()), total=len(sample_df), desc="Transcribing"):
        raw_asr_results.append(out["text"])

    sample_df["raw_asr"] = raw_asr_results
    
    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(OUTPUT_MANIFEST, index=False, encoding="utf-8")
    logger.info(f"Done! Saved {len(sample_df)} raw transcripts to {OUTPUT_MANIFEST}")

if __name__ == "__main__":
    main()
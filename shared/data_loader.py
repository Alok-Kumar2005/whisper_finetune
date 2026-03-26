# %%writefile shared/data_loader.py
import os, json, time, logging, requests
from pathlib import Path
from typing import Optional
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

def build_urls(row: pd.Series) -> dict:
    old_base = "joshtalks-data-collection/hq_data/hi"
    new_base = "upload_goai"
    return {
        "audio": str(row["rec_url_gcp"]).replace(old_base, new_base),
        "transcription": str(row["transcription_url_gcp"]).replace(old_base, new_base),
        "metadata": str(row["metadata_url_gcp"]).replace(old_base, new_base),
    }

def download_file(url: str, dest: Path, retries: int = 3, timeout: int = 60) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists(): return True
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
            return True
        except Exception as exc:
            logger.warning(f"Attempt {attempt}/{retries} failed for {url}: {exc}")
            time.sleep(2 ** attempt)
    return False

def load_transcription(json_path: Path) -> Optional[str]:
    if not json_path.exists(): return None
    try:
        with open(json_path, "r", encoding="utf-8") as f: data = json.load(f)
        for key in ("transcript", "transcription", "text", "sentence", "utterance"):
            if key in data: return data[key]
        return json.dumps(data, ensure_ascii=False)
    except Exception: return None

def load_metadata(json_path: Path) -> dict:
    if not json_path.exists(): return {}
    try:
        with open(json_path, "r", encoding="utf-8") as f: return json.load(f)
    except Exception: return {}

def download_dataset(tsv_path: str, output_dir: str = "q1_whisper_finetune/data", download_audio: bool = True,
                     download_transcription: bool = True, download_metadata: bool = False ) -> pd.DataFrame:
    df = pd.read_csv(tsv_path) if tsv_path.endswith(".csv") else pd.read_csv(tsv_path, sep="\t")
    out = Path(output_dir)
    records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading dataset"):
        urls = build_urls(row)
        rec_id = row["recording_id"]
        try: folder_id = urls["audio"].split("/upload_goai/")[1].split("/")[0]
        except IndexError: folder_id = "unknown_folder"

        rec_dir = out / folder_id
        rec_dir.mkdir(parents=True, exist_ok=True)
        audio_path, trans_path, meta_path = rec_dir / f"{rec_id}_audio.wav", rec_dir / f"{rec_id}_transcription.json", rec_dir / f"{rec_id}_metadata.json"

        if download_audio: download_file(urls["audio"], audio_path)
        if download_transcription: download_file(urls["transcription"], trans_path)
        if download_metadata: download_file(urls["metadata"], meta_path)

        records.append({
            **row.to_dict(),
            "audio_path": str(audio_path),
            "transcription_path": str(trans_path),
            "transcript_text": load_transcription(trans_path) if trans_path.exists() else None,
            "metadata_extra": load_metadata(meta_path) if meta_path.exists() else {},
        })

    result_df = pd.DataFrame(records)
    result_df.to_csv(out / "manifest.csv", index=False)
    return result_df
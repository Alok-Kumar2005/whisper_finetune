# %%writefile q2_asr_cleanup/pipeline.py
import sys, json, logging
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from q2_asr_cleanup.number_normalization.normalizer  import normalize_numbers
from q2_asr_cleanup.english_detection.detector       import tag_english_words, extract_english_words
from shared.utils import setup_logging, normalize_hindi_text, compute_wer

setup_logging()
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("q2_asr_cleanup/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def run_pipeline(raw_asr_text: str) -> dict:
    after_num  = normalize_numbers(raw_asr_text)
    after_eng  = tag_english_words(after_num)
    en_words   = extract_english_words(after_num)
    return {
        "raw": raw_asr_text, "number_norm": after_num, "english_tagged": after_eng,
        "final": after_num, "english_words": en_words,
    }

def process_dataset(manifest_csv: str, raw_asr_col: str = "raw_asr", ref_col: str = "transcript_text") -> pd.DataFrame:
    df = pd.read_csv(manifest_csv)
    results = []
    for _, row in df.iterrows():
        out = run_pipeline(str(row.get(raw_asr_col, "")))
        results.append({**row.to_dict(), "pipeline_number_norm": out["number_norm"], "pipeline_english_tagged": out["english_tagged"], "pipeline_final": out["final"], "detected_en_words": json.dumps(out["english_words"], ensure_ascii=False)})

    result_df = pd.DataFrame(results)
    valid = result_df.dropna(subset=[raw_asr_col, ref_col])
    if len(valid) > 0:
        refs = valid[ref_col].apply(normalize_hindi_text).tolist()
        raw_hyps = valid[raw_asr_col].apply(normalize_hindi_text).tolist()
        norm_hyps = valid["pipeline_final"].apply(normalize_hindi_text).tolist()
        wer_raw, wer_norm = compute_wer(refs, raw_hyps), compute_wer(refs, norm_hyps)
        print(f"\nWER before normalisation: {wer_raw:.4f}\nWER after  normalisation: {wer_norm:.4f}")
        delta = wer_norm - wer_raw
        print(f"Δ WER: {'+' if delta>0 else ''}{delta:.4f} ({'worse' if delta>0 else 'improvement'})")

    out_path = RESULTS_DIR / "pipeline_output.csv"
    result_df.to_csv(out_path, index=False, encoding="utf-8")
    return result_df

def run_demo():
    print("\n" + "=" * 75 + "\nASR CLEANUP PIPELINE – DEMO\n" + "=" * 75)
    examples = [
        ("उसने चौदह किताबें खरीदीं", "उसने 14 किताबें खरीदीं", "Simple number conversion"),
        ("दो-चार बातें बोलनी थीं", "दो-चार बातें बोलनी थीं", "EDGE: Idiom – should NOT convert")
    ]
    for raw, ref, desc in examples:
        out = run_pipeline(raw)
        print(f"\n  [{desc}]\n  Raw ASR  : {raw}\n  Num norm : {out['number_norm']}\n  EN tagged: {out['english_tagged']}")

if __name__ == "__main__":
    run_demo()
    manifest = Path("q2_asr_cleanup/raw_asr_manifest.csv")
    if manifest.exists(): process_dataset(str(manifest))
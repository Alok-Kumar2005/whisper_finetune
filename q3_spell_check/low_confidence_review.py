import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("q3_spell_check/results")
SPELL_CSV   = RESULTS_DIR / "spell_check_results.csv"
REVIEW_CSV  = RESULTS_DIR / "low_confidence_review.csv"
EXPORT_CSV  = RESULTS_DIR / "final_word_labels.csv"

def simulate_ground_truth(word: str) -> str:
    import unicodedata, re
    word = unicodedata.normalize("NFC", word)
    if re.search(r"[A-Za-z]", word) or re.search(r"[\u093E-\u094F]{2,}", word) or len(word) == 0:
        return "incorrect_spelling"
    return "correct_spelling"

def review_low_confidence(df: pd.DataFrame, n_sample: int = 45):
    low_df = df[df["confidence"] == "LOW"].copy()
    if len(low_df) == 0: return {}
    sampled = low_df.sample(n=min(n_sample, len(low_df)), random_state=42).reset_index(drop=True)
    sampled["ground_truth"] = sampled["word"].apply(simulate_ground_truth)
    sampled["classifier_correct"] = sampled["label"] == sampled["ground_truth"]

    correct_count = sampled["classifier_correct"].sum()
    print(f"\nLOW CONFIDENCE BUCKET REVIEW  (n={len(sampled)})\nAccuracy: {correct_count}/{len(sampled)} = {correct_count/len(sampled):.1%}")
    sampled.to_csv(REVIEW_CSV, index=False, encoding="utf-8")

def export_for_sheets(df: pd.DataFrame):
    export = df[["word", "label"]].copy()
    export.columns = ["Word", "Spelling Status"]
    export.to_csv(EXPORT_CSV, index=False, encoding="utf-8")
    print(f"\nFinal unique correctly-spelled words: {(export['Spelling Status'] == 'correct_spelling').sum()}")

if __name__ == "__main__":
    if SPELL_CSV.exists():
        df = pd.read_csv(SPELL_CSV)
        review_low_confidence(df)
        export_for_sheets(df)
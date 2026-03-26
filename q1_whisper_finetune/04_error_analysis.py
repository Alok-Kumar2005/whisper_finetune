# %%writefile q1_whisper_finetune/04_error_analysis.py
import sys, re, json, logging
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Tuple
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.utils import setup_logging, compute_wer

setup_logging()
logger = logging.getLogger(__name__)

RESULTS_DIR  = Path("q1_whisper_finetune/results")
EVAL_CSV     = RESULTS_DIR / "fleurs_evaluation.csv"
ANALYSIS_OUT = RESULTS_DIR / "error_analysis.json"
RANDOM_SEED  = 42
N_SAMPLE     = 25

def utt_wer(ref: str, hyp: str) -> float:
    return 1.0 if not ref.split() else compute_wer([ref], [hyp])

def word_diff(ref: str, hyp: str) -> List[Tuple[str, str, str]]:
    ref_tokens, hyp_tokens = ref.split(), hyp.split()
    r, h = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (h + 1) for _ in range(r + 1)]
    for i in range(r + 1): dp[i][0] = i
    for j in range(h + 1): dp[0][j] = j
    for i in range(1, r + 1):
        for j in range(1, h + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]: dp[i][j] = dp[i - 1][j - 1]
            else: dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    
    ops = []
    i, j = r, h
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_tokens[i - 1] == hyp_tokens[j - 1]:
            ops.append(("=", ref_tokens[i - 1], hyp_tokens[j - 1]))
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("sub", ref_tokens[i - 1], hyp_tokens[j - 1]))
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", ref_tokens[i - 1], ""))
            i -= 1
        else:
            ops.append(("ins", "", hyp_tokens[j - 1]))
            j -= 1
    return list(reversed(ops))

HINDI_DIGIT_WORDS = {"एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ", "दस", "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस", "बीस", "तीस", "चालीस", "पचास", "साठ", "सत्तर", "अस्सी", "नब्बे", "सौ", "हज़ार", "लाख", "करोड़"}
LATIN_RE = re.compile(r"[A-Za-z]")

def classify_error(ops: List[Tuple]) -> Counter:
    cats = Counter()
    for op, ref_w, hyp_w in ops:
        if op == "=": continue
        elif op == "del": cats["B_deletion"] += 1
        elif op == "ins": cats["C_insertion"] += 1
        elif op == "sub":
            if ref_w in HINDI_DIGIT_WORDS or hyp_w in HINDI_DIGIT_WORDS or ref_w.isdigit() or hyp_w.isdigit(): cats["D_number"] += 1
            elif LATIN_RE.search(ref_w) or LATIN_RE.search(hyp_w): cats["E_english_script"] += 1
            else: cats["A_substitution"] += 1
    return cats

def stratified_sample(df: pd.DataFrame, n_total: int = N_SAMPLE) -> pd.DataFrame:
    mild, medium, severe = df[df["utt_wer"] <= 0.33], df[(df["utt_wer"] > 0.33) & (df["utt_wer"] <= 0.66)], df[df["utt_wer"] > 0.66]
    parts = []
    for tier_df, tier_n in [(mild, 8), (medium, 9), (severe, 8)]:
        if len(tier_df) == 0: continue
        n = min(tier_n, len(tier_df))
        step = max(1, len(tier_df) // n)
        parts.append(tier_df.iloc[list(range(0, len(tier_df), step))[:n]])
    
    sampled = pd.concat(parts).drop_duplicates()
    if len(sampled) < n_total:
        remaining = df[~df.index.isin(sampled.index)]
        extra = remaining.sample(n=min(n_total - len(sampled), len(remaining)), random_state=RANDOM_SEED)
        sampled = pd.concat([sampled, extra])
    return sampled.reset_index(drop=True)

TAXONOMY = {
    "A_substitution": {"label": "Phonetic / lexical substitution"},
    "B_deletion": {"label": "Word deletion"},
    "C_insertion": {"label": "Word insertion (hallucination)"},
    "D_number": {"label": "Number normalisation mismatch"},
    "E_english_script": {"label": "English word / script mismatch"},
    "F_oov_proper_noun": {"label": "OOV / proper nouns"},
    "G_grammatical_variant": {"label": "Valid grammatical variant"},
    "H_truncation": {"label": "Incomplete / truncated output"}
}

def main():
    if not EVAL_CSV.exists(): return
    df = pd.read_csv(EVAL_CSV)
    df = df[df["finetuned_correct"] == False].copy()
    df["utt_wer"] = df.apply(lambda r: utt_wer(str(r["reference"]), str(r["finetuned_hyp"])), axis=1)
    
    sampled = stratified_sample(df)
    category_examples = defaultdict(list)
    all_cats = Counter()

    for _, row in sampled.iterrows():
        ref, hyp = str(row["reference"]), str(row["finetuned_hyp"])
        ops = word_diff(ref, hyp)
        cats = classify_error(ops)
        if len(hyp.split()) < len(ref.split()) * 0.5: cats["H_truncation"] += 1
        all_cats += cats

        for cat in cats:
            category_examples[cat].append({
                "reference": ref, "hypothesis": hyp, "utt_wer": round(row["utt_wer"], 3),
                "ops": [(op, rw, hw) for op, rw, hw in ops if op != "="][:5]
            })

    output = {
        "total_sampled": len(sampled),
        "category_counts": dict(all_cats.most_common()),
        "taxonomy": TAXONOMY,
        "category_examples": {k: v[:5] for k, v in category_examples.items()}
    }
    with open(ANALYSIS_OUT, "w", encoding="utf-8") as f: json.dump(output, f, ensure_ascii=False, indent=2)
    sampled.to_csv(RESULTS_DIR / "sampled_errors.csv", index=False)
    logger.info(f"Analysis saved to {ANALYSIS_OUT}")

if __name__ == "__main__":
    main()
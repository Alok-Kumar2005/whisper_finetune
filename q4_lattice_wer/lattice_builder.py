import re
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from q2_asr_cleanup.number_normalization.normalizer import normalize_numbers

AGREEMENT_THRESHOLD = 0.6
MORPHOLOGICAL_VARIANTS = {
    "खरीदीं": ["खरीदी", "ख़रीदीं"],
    "किताबें": ["किताबे", "पुस्तकें", "किताबों"],
    "मौनता": ["मौन", "मोनता"],
    "रक्षाबंधन": ["रक्षा बंधन", "रक्षा-बंधन"],
    "बहनों": ["बहनो", "बहनें"],
    "खेतीबाड़ी": ["खेती बाड़ी", "खेती-बाड़ी"],
    "क्या": ["क्या?"],
    "होता": ["होती", "होतई"],
    "है": ["हे", "हैं"],
}

_VARIANT_REVERSE = {}
for canonical, variants in MORPHOLOGICAL_VARIANTS.items():
    for v in variants:
        if v not in _VARIANT_REVERSE: _VARIANT_REVERSE[v] = []
        _VARIANT_REVERSE[v].append(canonical)
        for other in variants:
            if other != v: _VARIANT_REVERSE[v].append(other)

def get_variants(token: str) -> List[str]:
    variants = {token}
    if token in MORPHOLOGICAL_VARIANTS: variants.update(MORPHOLOGICAL_VARIANTS[token])
    if token in _VARIANT_REVERSE: variants.update(_VARIANT_REVERSE[token])
    return list(variants)

def add_number_variants(token: str) -> List[str]:
    variants = [token]
    if not token.isdigit():
        normalised = normalize_numbers(token)
        if normalised != token: variants.append(normalised)
    return variants

GAP = None

def align_sequences(ref: List[str], hyp: List[str]) -> Tuple[List, List]:
    r, h = len(ref), len(hyp)
    dp = [[0] * (h + 1) for _ in range(r + 1)]
    for i in range(r + 1): dp[i][0] = i
    for j in range(h + 1): dp[0][j] = j

    for i in range(1, r + 1):
        for j in range(1, h + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + cost, dp[i - 1][j] + 1, dp[i][j - 1] + 1)

    aligned_ref, aligned_hyp = [], []
    i, j = r, h
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (0 if ref[i-1] == hyp[j-1] else 1):
            aligned_ref.append(ref[i - 1])
            aligned_hyp.append(hyp[j - 1])
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            aligned_ref.append(ref[i - 1])
            aligned_hyp.append(GAP)
            i -= 1
        else:
            aligned_ref.append(GAP)
            aligned_hyp.append(hyp[j - 1])
            j -= 1
    return list(reversed(aligned_ref)), list(reversed(aligned_hyp))

class Lattice:
    def __init__(self):
        self.bins: List[set] = []
        self.reference: List[str] = []
        self.model_names: List[str] = []
        self.trusted_cols: set = set()

    def pretty_print(self):
        print("\nLATTICE\n" + "=" * 60)
        for i, bin_set in enumerate(self.bins):
            trust = " ★ (model-trusted)" if i in self.trusted_cols else ""
            print(f"  Bin {i+1:>3}: {sorted(bin_set)}{trust}")
        print("=" * 60)

def build_lattice(reference: List[str], model_outputs: Dict[str, List[str]]) -> Lattice:
    lattice = Lattice()
    lattice.reference = reference
    lattice.model_names = list(model_outputs.keys())
    
    model_aligned = {}
    for name, hyp in model_outputs.items():
        aligned_ref, aligned_hyp = align_sequences(reference, hyp)
        model_aligned[name] = (aligned_ref, aligned_hyp)

    anchor_ref, anchor_hyp = model_aligned[lattice.model_names[0]] if lattice.model_names else (reference, reference)

    bins = []
    for col_idx in range(len(anchor_ref)):
        ref_tok_at_col = anchor_ref[col_idx]
        model_toks_at_col = {name: (model_aligned[name][1][col_idx] if col_idx < len(model_aligned[name][1]) else GAP) for name in lattice.model_names}
        
        bin_set = set()
        if ref_tok_at_col is not GAP:
            bin_set.add(ref_tok_at_col)
            for v in get_variants(ref_tok_at_col): bin_set.add(v)
            for v in add_number_variants(ref_tok_at_col): bin_set.add(v)

        non_gap_model_toks = [t for t in model_toks_at_col.values() if t is not GAP]
        for tok in non_gap_model_toks:
            bin_set.add(tok)
            for v in get_variants(tok): bin_set.add(v)
            for v in add_number_variants(tok): bin_set.add(v)

        if non_gap_model_toks:
            most_common_tok, count = Counter(non_gap_model_toks).most_common(1)[0]
            if (count / len(model_outputs)) >= AGREEMENT_THRESHOLD and ref_tok_at_col is not GAP and most_common_tok != ref_tok_at_col:
                bin_set.add(most_common_tok)
                lattice.trusted_cols.add(col_idx)

        bin_set.discard(None)
        bins.append(bin_set)

    lattice.bins = bins
    return lattice
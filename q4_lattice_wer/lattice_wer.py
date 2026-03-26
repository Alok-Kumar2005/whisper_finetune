import sys
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from q4_lattice_wer.lattice_builder import Lattice, GAP

def lattice_edit_distance(lattice: Lattice, hyp: List[str]) -> Tuple[int, int, int, int]:
    bins, N, H = lattice.bins, len(lattice.bins), len(hyp)
    dp = [[0] * (H + 1) for _ in range(N + 1)]

    for i in range(N + 1): dp[i][0] = i
    for j in range(H + 1): dp[0][j] = j

    for i in range(1, N + 1):
        for j in range(1, H + 1):
            sub_cost = 0 if hyp[j - 1] in bins[i - 1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + sub_cost, dp[i - 1][j] + 1, dp[i][j - 1] + 1)

    S = D = I = 0
    i, j = N, H
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            sub_cost = 0 if hyp[j - 1] in bins[i - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + sub_cost:
                if sub_cost == 1: S += 1
                i -= 1; j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            D += 1; i -= 1
        else:
            I += 1; j -= 1
    return S, D, I, N

def compute_lattice_wer(lattice: Lattice, hyp: List[str]) -> float:
    S, D, I, N = lattice_edit_distance(lattice, hyp)
    return (S + D + I) / N if N > 0 else 0.0

def rigid_wer(ref: List[str], hyp: List[str]) -> float:
    from jiwer import wer as jiwer_wer
    ref_str, hyp_str = " ".join(ref), " ".join(hyp)
    return jiwer_wer(ref_str, hyp_str) if ref_str.strip() else 0.0

def evaluate_all_models(lattice: Lattice, model_outputs: Dict[str, List[str]], reference: List[str]) -> Dict[str, dict]:
    results = {}
    for name, hyp in model_outputs.items():
        S, D, I, N = lattice_edit_distance(lattice, hyp)
        lat_wer = (S + D + I) / max(N, 1)
        rig_wer = rigid_wer(reference, hyp)
        results[name] = {"rigid_wer": round(rig_wer, 4), "lattice_wer": round(lat_wer, 4), "delta": round(lat_wer - rig_wer, 4), "S": S, "D": D, "I": I, "N": N}
    return results

def print_comparison_table(results: Dict[str, dict], human_ref_name: str = "Human"):
    header = f"{'Model':<18} | {'Rigid WER':>10} | {'Lattice WER':>12} | {'Δ WER':>8} | {'Note'}"
    print("\n" + "─" * len(header) + "\n" + header + "\n" + "─" * len(header))
    for model, r in results.items():
        note = "▼ improved" if r["delta"] < -0.005 else ("▲ slightly higher" if r["delta"] > 0.005 else "≈ unchanged")
        print(f"{model:<18} | {r['rigid_wer']:>10.4f} | {r['lattice_wer']:>12.4f} | {r['delta']:>+8.4f} | {note}")
    print("─" * len(header))
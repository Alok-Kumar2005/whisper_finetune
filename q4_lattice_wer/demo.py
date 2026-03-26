import sys
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from q4_lattice_wer.lattice_builder import build_lattice
from q4_lattice_wer.lattice_wer import evaluate_all_models, print_comparison_table
from shared.utils import setup_logging

setup_logging()

SEGMENTS = [
    {"id": "seg_1", "url": "ch_1726922_866.16_868.20.wav", "human": "वही अपना खेती बाड़ी और क्या", "model_H": "वही अपना खेती बाड़ी और क्या", "model_i": "वही अपना खेती बाड़ी और क्या", "model_k": "वही अपना खेती बाड़ी और क्या?", "model_l": "वही अपना खेती बाड़ी और क्या", "model_m": "वही अपना खेतीबाड़ी और क्या", "model_n": "वही अपना खेती बाड़ी और क्या"},
    {"id": "seg_2", "url": "ch_2052946_52.62_54.36.wav", "human": "मौनता का अर्थ क्या होता है", "model_H": "मौनता का अर्थ क्या होता है", "model_i": "मौनता का अर्थ क्या होता है?", "model_k": "मौन तागार थके होतई।", "model_l": "मोनता का अर्थ है क्या होता है", "model_m": "मोन ताका हर थक्या होताहए", "model_n": "मौनता का हर थका होता है"},
    {"id": "seg_3", "url": "ch_2054042_186.66_189.27.wav", "human": "और रक्षाबंधन पे चलो बहनों को", "model_H": "और रक्षाबंधन पे चलो बहनों को", "model_i": "और रक्षाबंधन पे चलो बहनों को --", "model_k": "और रक्षाबंधन पे चलो बहनों को?", "model_l": "और रक्षाबंधन पे चलो बहनों को", "model_m": "और रक्षा बंधन पे चलो बहनों को", "model_n": "और रक्षा बंधन पे चलो बहनों को"},
]

MODEL_KEYS = ["model_H", "model_i", "model_k", "model_l", "model_m", "model_n"]

def tokenise(text: str) -> List[str]:
    import re
    cleaned = []
    for tok in text.strip().split():
        tok = re.sub(r"[।\?\!\.\,\;\:\"\'\(\)\[\]\{\}\-\_]+$", "", tok)
        if tok: cleaned.append(tok)
    return cleaned

def run_demo():
    all_segment_results = {}
    for seg in SEGMENTS:
        print(f"\n{'='*70}\nSEGMENT: {seg['id']}  ({seg['url']})\n{'='*70}\nHuman reference: {seg['human']}")
        reference = tokenise(seg["human"])
        model_outputs = {key.replace("model_", ""): tokenise(seg[key]) for key in MODEL_KEYS}
        
        lattice = build_lattice(reference, model_outputs)
        lattice.pretty_print()
        
        results = evaluate_all_models(lattice, model_outputs, reference)
        print_comparison_table(results, human_ref_name="Human")
        all_segment_results[seg["id"]] = results

    print("\n" + "="*70 + "\nAGGREGATE WER ACROSS ALL 3 SEGMENTS\n" + "="*70)
    model_names = [k.replace("model_", "") for k in MODEL_KEYS]
    agg_rigid = {m: 0.0 for m in model_names}
    agg_lattice = {m: 0.0 for m in model_names}

    for seg_results in all_segment_results.values():
        for model, r in seg_results.items():
            agg_rigid[model] += r["rigid_wer"]
            agg_lattice[model] += r["lattice_wer"]

    print(f"\n{'Model':<12} | {'Avg Rigid WER':>14} | {'Avg Lattice WER':>16} | {'Avg Δ':>8}\n" + "─" * 60)
    for m in model_names:
        r, l = agg_rigid[m] / len(SEGMENTS), agg_lattice[m] / len(SEGMENTS)
        flag = "▼ improved" if (l - r) < -0.01 else ("▲ worse" if (l - r) > 0.01 else "≈ same")
        print(f"{m:<12} | {r:>14.4f} | {l:>16.4f} | {(l-r):>+8.4f}  {flag}")

if __name__ == "__main__":
    run_demo()
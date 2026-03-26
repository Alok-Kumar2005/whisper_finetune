# %%writefile q1_whisper_finetune/03_evaluate.py
import sys, logging, torch, gc
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from transformers import WhisperProcessor, pipeline
import evaluate
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.utils import setup_logging, normalize_hindi_text, print_wer_table

setup_logging()
logger = logging.getLogger(__name__)

PRETRAINED_MODEL = "openai/whisper-small"
FINETUNED_MODEL  = "q1_whisper_finetune/models/whisper-small-hindi/final"
RESULTS_DIR      = Path("q1_whisper_finetune/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# FIX 1: Cut memory usage in half by using float16 if on GPU
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

def load_fleurs_hindi():
    logger.info("Loading FLEURS hi_in test split...")
    return load_dataset("google/fleurs", "hi_in", split="test")

def transcribe_dataset(model_path: str, dataset) -> list:
    logger.info(f"Transcribing with: {model_path} on {DEVICE.upper()}")
    
    # FIX 2: Reduced batch size to 4 to prevent 15GB RAM limit crashing
    asr_pipe = pipeline(
        "automatic-speech-recognition", 
        model=model_path, 
        device=0 if DEVICE == "cuda" else -1, 
        torch_dtype=TORCH_DTYPE,
        chunk_length_s=30, 
        batch_size=4 
    )
    
    asr_pipe.model.generation_config.forced_decoder_ids = None
    asr_pipe.model.generation_config.suppress_tokens = []
    asr_pipe.model.generation_config.language = "hindi"
    asr_pipe.model.generation_config.task = "transcribe"
    
    def data():
        for ex in dataset:
            yield {"array": ex["audio"]["array"], "sampling_rate": ex["audio"]["sampling_rate"]}
    
    hyps = []
    for res in tqdm(asr_pipe(data()), total=len(dataset), desc=f"Evaluating {model_path.split('/')[-1]}"):
        hyps.append(normalize_hindi_text(res["text"]))
    
    # FIX 3: Force the system to delete the model from RAM when it finishes
    del asr_pipe
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        
    return hyps

def score(refs: list, hyps: list) -> float:
    metric = evaluate.load("wer")
    return metric.compute(predictions=[normalize_hindi_text(h) for h in hyps], references=[normalize_hindi_text(r) for r in refs])

def main():
    fleurs = load_fleurs_hindi()
    refs = [ex["transcription"] for ex in fleurs]
    
    hyps_p = transcribe_dataset(PRETRAINED_MODEL, fleurs)
    wer_p = score(refs, hyps_p)
    
    hyps_f = transcribe_dataset(FINETUNED_MODEL, fleurs)
    wer_f = score(refs, hyps_f)
    
    results = {"Whisper Small (Pretrained)": wer_p, "FT Whisper Small (ours)": wer_f}
    print("\n")
    print_wer_table(results)
    
    df = pd.DataFrame({
        "id": [ex.get("id", i) for i, ex in enumerate(fleurs)],
        "reference": [normalize_hindi_text(r) for r in refs],
        "pretrained_hyp": hyps_p,
        "finetuned_hyp": hyps_f,
        "pretrained_correct": [normalize_hindi_text(r) == h for r, h in zip(refs, hyps_p)],
        "finetuned_correct": [normalize_hindi_text(r) == h for r, h in zip(refs, hyps_f)]
    })
    df.to_csv(RESULTS_DIR / "fleurs_evaluation.csv", index=False)
    pd.DataFrame([{"Model": k, "WER": v, "WER_%": round(v * 100, 2)} for k, v in results.items()]).to_csv(RESULTS_DIR / "wer_summary.csv", index=False)

if __name__ == "__main__":
    main()
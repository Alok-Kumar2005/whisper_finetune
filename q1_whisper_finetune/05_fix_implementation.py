# %%writefile q1_whisper_finetune/05_fix_implementation.py
import sys, logging, random, gc, json
from pathlib import Path
from typing import List
import torch, numpy as np, pandas as pd, soundfile as sf, librosa
from datasets import Dataset, Audio, load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.utils import setup_logging, normalize_hindi_text, DataCollatorSpeechSeq2SeqWithPadding

setup_logging()
logger = logging.getLogger(__name__)

FINETUNED_MODEL = "q1_whisper_finetune/models/whisper-small-hindi/final"
FIX_OUTPUT_DIR  = Path("q1_whisper_finetune/models/whisper-small-hindi-fix1")
RESULTS_DIR     = Path("q1_whisper_finetune/results")
SAMPLED_CSV     = RESULTS_DIR / "sampled_errors.csv"
AUGMENTED_DIR   = Path("q1_whisper_finetune/data/augmented")
AUGMENTED_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR = 16_000
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

def speed_perturb(audio: np.ndarray, sr: int, rates: List[float] = [0.9, 1.0, 1.1]) -> np.ndarray:
    rate = random.choice(rates)
    return audio if rate == 1.0 else librosa.effects.time_stretch(audio, rate=rate)

def spec_augment_audio(audio: np.ndarray, sr: int, freq_mask_pct: float = 0.15, time_mask_pct: float = 0.10) -> np.ndarray:
    augmented = audio.copy()
    mask_len = int(len(augmented) * time_mask_pct)
    start = random.randint(0, max(0, len(augmented) - mask_len))
    augmented[start: start + mask_len] = 0.0
    return augmented

def augment_array_and_save(audio: np.ndarray, sr: int, stem: str, out_dir: Path, n_augments: int = 3) -> List[str]:
    if audio.ndim > 1: audio = audio.mean(axis=1)
    if sr != TARGET_SR: audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

    saved = []
    for i in range(n_augments):
        aug = speed_perturb(audio.copy(), TARGET_SR)
        if random.random() > 0.5: aug = spec_augment_audio(aug, TARGET_SR)
        out_path = out_dir / f"{stem}_aug{i}.wav"
        sf.write(str(out_path), aug, TARGET_SR)
        saved.append(str(out_path))
    return saved

def build_augmented_dataset(sampled_df: pd.DataFrame, fleurs_ds) -> Dataset:
    records = []
    if "error_cats" in sampled_df.columns:
        deletion_df = sampled_df[sampled_df["error_cats"].str.contains("B_deletion", na=False)]
    else:
        deletion_df = sampled_df

    if len(deletion_df) == 0:
        deletion_df = sampled_df

    fleurs_dict = {ex.get("id", i): ex["audio"] for i, ex in enumerate(fleurs_ds)}

    for _, row in deletion_df.iterrows():
        uid = row["id"]
        if uid not in fleurs_dict: continue
        
        audio_data = fleurs_dict[uid]
        audio_array = audio_data["array"]
        sr = audio_data["sampling_rate"]
        
        aug_paths = augment_array_and_save(audio_array, sr, str(uid), AUGMENTED_DIR)
        for ap in aug_paths:
            records.append({"audio": ap, "sentence": normalize_hindi_text(str(row.get("reference", "")))})

    if not records: return None
    ds = Dataset.from_dict({"audio": [r["audio"] for r in records], "sentence": [r["sentence"] for r in records]})
    return ds.cast_column("audio", Audio(sampling_rate=TARGET_SR))

def prepare_dataset_fn(batch, processor):
    batch["input_features"] = processor.feature_extractor(batch["audio"]["array"], sampling_rate=batch["audio"]["sampling_rate"]).input_features[0]
    tokenized_text = processor.tokenizer(normalize_hindi_text(batch["sentence"]), truncation=True, max_length=448)
    batch["labels"] = tokenized_text.input_ids
    return batch

def evaluate_model_on_subset(model_path: str, subset_df: pd.DataFrame, fleurs_ds) -> float:
    from transformers import pipeline as hf_pipeline
    
    # THE FIX: Added chunk_length_s and batch_size
    asr_pipe = hf_pipeline(
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

    metric = evaluate.load("wer")
    fleurs_dict = {ex.get("id", i): ex["audio"] for i, ex in enumerate(fleurs_ds)}
    
    valid_rows = [row for _, row in subset_df.iterrows() if row["id"] in fleurs_dict]
    refs = [normalize_hindi_text(str(row["reference"])) for row in valid_rows]
    hyps = []

    def data_generator():
        for row in valid_rows:
            audio_data = fleurs_dict[row["id"]]
            yield {"array": audio_data["array"].astype(np.float32), "sampling_rate": audio_data["sampling_rate"]}

    # Added progress bar so it doesn't look frozen
    for result in tqdm(asr_pipe(data_generator()), total=len(valid_rows), desc="Evaluating subset"):
        hyps.append(normalize_hindi_text(result["text"]))
        
    del asr_pipe
    gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
        
    return metric.compute(predictions=hyps, references=refs) if refs else float("nan")

def main():
    if not SAMPLED_CSV.exists(): 
        logger.error(f"{SAMPLED_CSV} not found!")
        return
        
    sampled_df = pd.read_csv(SAMPLED_CSV)
    
    logger.info("Loading FLEURS dataset to extract audio arrays...")
    fleurs_ds = load_dataset("google/fleurs", "hi_in", split="test")
    
    logger.info("Evaluating baseline on subset...")
    wer_before = evaluate_model_on_subset(FINETUNED_MODEL, sampled_df, fleurs_ds)
    
    logger.info("Building augmented dataset...")
    aug_ds = build_augmented_dataset(sampled_df, fleurs_ds)
    if aug_ds is None: 
        logger.error("Failed to build augmented dataset.")
        return

    processor = WhisperProcessor.from_pretrained(FINETUNED_MODEL, language="Hindi", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(FINETUNED_MODEL)
    
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens    = []
    model.generation_config.language           = "hindi"
    model.generation_config.task               = "transcribe"

    aug_ds = aug_ds.map(lambda b: prepare_dataset_fn(b, processor), remove_columns=aug_ds.column_names)
    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(FIX_OUTPUT_DIR), per_device_train_batch_size=4, gradient_accumulation_steps=4,
        learning_rate=5e-6, max_steps=50, gradient_checkpointing=True, fp16=DEVICE == "cuda",
        eval_strategy="no", save_steps=50, logging_steps=10, predict_with_generate=True, report_to=[]
    )

    trainer = Seq2SeqTrainer(
        model           = model, 
        args            = training_args, 
        train_dataset   = aug_ds, 
        processing_class= processor.feature_extractor,
        data_collator   = collator
    )
    
    logger.info("Training fix model...")
    trainer.train()
    trainer.save_model(str(FIX_OUTPUT_DIR / "final"))
    processor.save_pretrained(str(FIX_OUTPUT_DIR / "final"))

    logger.info("Evaluating fixed model on subset...")
    wer_after = evaluate_model_on_subset(str(FIX_OUTPUT_DIR / "final"), sampled_df, fleurs_ds)
    
    results_text = f"RESULTS (Subset Evaluation)\n{'='*50}\nBefore WER: {wer_before:.4f}\nAfter WER:  {wer_after:.4f}\n{'='*50}"
    print("\n" + results_text)
    
    with open(RESULTS_DIR / "fix_results.txt", "w") as f:
        f.write(results_text)
    logger.info(f"Saved results to {RESULTS_DIR / 'fix_results.txt'}")

if __name__ == "__main__":
    main()
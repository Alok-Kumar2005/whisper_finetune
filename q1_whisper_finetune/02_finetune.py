# %%writefile q1_whisper_finetune/02_finetune.py
import sys
import logging
from pathlib import Path

from datasets import DatasetDict, load_from_disk
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.utils import setup_logging, normalize_hindi_text, DataCollatorSpeechSeq2SeqWithPadding

setup_logging()
logger = logging.getLogger(__name__)

HF_DATASET_DIR = Path("q1_whisper_finetune/data/hf_dataset")
MODEL_NAME     = "openai/whisper-small"
OUTPUT_DIR     = Path("q1_whisper_finetune/models/whisper-small-hindi")
LANGUAGE       = "Hindi"
TASK           = "transcribe"

TRAIN_ARGS = dict(
    output_dir                  = str(OUTPUT_DIR),
    per_device_train_batch_size = 8,
    per_device_eval_batch_size  = 8,
    gradient_accumulation_steps = 2,
    learning_rate             = 1e-5,
    lr_scheduler_type         = "constant_with_warmup",
    warmup_steps              = 100,
    max_steps                 = 1500,
    gradient_checkpointing    = True,
    fp16                      = True,
    eval_strategy             = "steps", 
    eval_steps                = 300,
    save_steps                = 300,
    logging_steps             = 50,
    load_best_model_at_end    = True,
    metric_for_best_model     = "wer",
    greater_is_better         = False,
    predict_with_generate     = True,
    generation_max_length     = 225,
    report_to                 = ["tensorboard"],
    save_total_limit          = 2,
)

def prepare_dataset(batch, processor: WhisperProcessor):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    
    tokenized_text = processor.tokenizer(
        normalize_hindi_text(batch["sentence"]),
        truncation=True, 
        max_length=448   
    )
    
    batch["labels"] = tokenized_text.input_ids
    return batch

def build_compute_metrics(processor: WhisperProcessor):
    wer_metric = evaluate.load("wer")
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str  = processor.tokenizer.batch_decode(pred_ids,   skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids,  skip_special_tokens=True)
        pred_str  = [normalize_hindi_text(s) for s in pred_str]
        label_str = [normalize_hindi_text(s) for s in label_str]
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}
    return compute_metrics

def main():
    logger.info("Loading preprocessed dataset from disk...")
    dataset: DatasetDict = load_from_disk(str(HF_DATASET_DIR))
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
    
    logger.info("Extracting features and applying truncation...")
    dataset = dataset.map(
        lambda b: prepare_dataset(b, processor),
        remove_columns=dataset["train"].column_names,
        num_proc=1,
    )

    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    # --- THE FIX: ALL settings moved to generation_config ---
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens    = []
    model.generation_config.language           = LANGUAGE.lower()
    model.generation_config.task               = TASK

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    training_args = Seq2SeqTrainingArguments(**TRAIN_ARGS)

    trainer = Seq2SeqTrainer(
        model           = model,
        args            = training_args,
        train_dataset   = dataset["train"],
        eval_dataset    = dataset["validation"],
        processing_class= processor.feature_extractor,
        data_collator   = data_collator,
        compute_metrics = build_compute_metrics(processor),
    )

    logger.info("Starting training...")
    # Resume from checkpoint if it exists from the previous 300 steps!
    try:
        trainer.train(resume_from_checkpoint=True)
    except ValueError:
        trainer.train()
    
    final_dir = OUTPUT_DIR / "final"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    logger.info(f"Final model saved to {final_dir}")

if __name__ == "__main__":
    main()
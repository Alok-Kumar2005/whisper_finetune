# %%writefile shared/utils.py
import re, unicodedata, logging, torch
from dataclasses import dataclass
from typing import List, Dict, Union, Any, Tuple
import evaluate

_wer_metric = None

def get_wer_metric():
    global _wer_metric
    if _wer_metric is None:
        _wer_metric = evaluate.load("wer")
    return _wer_metric

def compute_wer(references: List[str], hypotheses: List[str]) -> float:
    metric = get_wer_metric()
    return metric.compute(references=references, predictions=hypotheses)

def normalize_hindi_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[।\?\!\.\,\;\:\"\'\(\)\[\]\{\}\-\_]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def setup_logging(level: str = "INFO"):
    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", level=getattr(logging, level.upper()))

def print_wer_table(results: dict):
    header = f"{'Model':<35} | {'WER (Hindi)':<12}"
    sep    = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for model, wer in results.items():
        print(f"{model:<35} | {wer:.4f} ({wer*100:.1f}%)")
    print(sep)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all(): labels = labels[:, 1:]
        batch["labels"] = labels
        return batch
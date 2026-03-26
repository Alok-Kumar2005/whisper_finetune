import sys
import re
import json
import math
import logging
import unicodedata
from pathlib import Path
from collections import Counter
from typing import Dict, Tuple, List, Optional
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from q2_asr_cleanup.english_detection.script_utils import classify_word_script
from q2_asr_cleanup.english_detection.detector import _is_devanagari_english
from shared.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("q3_spell_check/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DICT_PATHS = [
    Path("resources/hindi_wordlist.txt"),
    Path("/usr/share/dict/hindi"),
]

SEED_WORDS = {
    "है", "हैं", "था", "थी", "थे", "हो", "हूं", "हुआ", "हुई", "हुए",
    "में", "से", "को", "के", "की", "का", "पर", "और", "या", "तो",
    "भी", "नहीं", "जी", "ना", "हां", "कि", "जो", "यह", "वह", "इस",
    "उस", "इन", "उन", "मैं", "हम", "आप", "वो", "ये", "वे", "मेरा",
    "मेरी", "मेरे", "आपका", "आपकी", "आपके", "हमारा", "उनका",
    "करना", "करता", "करती", "करते", "करें", "किया", "की", "कर",
    "होना", "होता", "होती", "होते", "जाना", "जाता", "जाती", "गया",
    "गई", "गए", "आना", "आता", "आती", "आए", "देना", "देता", "दिया",
    "लेना", "लेता", "लिया", "बोलना", "बताना", "समझना", "सोचना",
    "लोग", "बात", "काम", "दिन", "साल", "घर", "देश", "सरकार", "समय",
    "जगह", "तरीका", "चीज", "बच्चा", "आदमी", "औरत", "परिवार", "पैसा",
    "रुपये", "नाम", "जिंदगी", "दुनिया", "सोच", "मन", "दिल",
    "अच्छा", "अच्छी", "अच्छे", "बड़ा", "बड़ी", "बड़े", "छोटा", "छोटी",
    "नया", "नई", "पुराना", "सही", "गलत", "सच", "झूठ", "ज्यादा",
    "कम", "बहुत", "बस", "सिर्फ", "अभी", "बाद", "पहले", "फिर",
    "कभी", "हमेशा", "कहाँ", "क्यों", "कैसे", "क्या", "कब", "कौन",
    "एक", "दो", "तीन", "चार", "पाँच", "पांच", "छह", "सात", "आठ", "नौ",
    "दस", "बीस", "तीस", "सौ", "हज़ार", "लाख", "करोड़",
}

def load_dictionary() -> set:
    words = set(SEED_WORDS)
    for path in DICT_PATHS:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    w = line.strip()
                    if w: words.add(w)
            logger.info(f"Loaded dictionary from {path}: {len(words)} words total")
            break
    else:
        logger.info(f"Using seed dictionary: {len(words)} words")
    return words

VALID_DEVA_RE = re.compile(r"^[\u0900-\u097F\u0966-\u096F\s]+$")
INVALID_SEQUENCE_RE = re.compile(r"[\u093E-\u094F\u0955-\u0963]{3,}|\u094D\u094D|^\u094D")

def is_valid_unicode_sequence(word: str) -> bool:
    try: normalized = unicodedata.normalize("NFC", word)
    except Exception: return False
    return not bool(INVALID_SEQUENCE_RE.search(normalized))

def has_valid_devanagari_structure(word: str) -> bool:
    return bool(VALID_DEVA_RE.match(word))

VALID_SUFFIXES = {"ना", "ता", "ती", "ते", "या", "यी", "ए", "एं", "ओ", "ओं", "इया", "वाला", "वाली", "वाले", "पन", "त्व", "ई", "ी", "ा", "े", "ें", "ों"}
SUSPECT_PATTERNS = [re.compile(r"आा"), re.compile(r"इई"), re.compile(r"ओउ"), re.compile(r"कख|खग"), re.compile(r"ंं"), re.compile(r"ः\S")]

def morphological_plausibility(word: str) -> float:
    if len(word) <= 1: return 0.5
    score = 1.0
    for pat in SUSPECT_PATTERNS:
        if pat.search(word): score -= 0.3
    for suf in VALID_SUFFIXES:
        if word.endswith(suf):
            score += 0.1
            break
    return max(0.0, min(1.0, score))

class CharNgramModel:
    def __init__(self, n: int = 3):
        self.n = n
        self.counts = Counter()
        self.context_counts = Counter()
        self.trained = False

    def train(self, words: List[str]):
        for word in words:
            padded = "^" * (self.n - 1) + word + "$"
            for i in range(len(padded) - self.n + 1):
                ngram = padded[i: i + self.n]
                context = padded[i: i + self.n - 1]
                self.counts[ngram] += 1
                self.context_counts[context] += 1
        self.trained = True

    def log_prob(self, word: str) -> float:
        if not self.trained: return 0.0
        padded = "^" * (self.n - 1) + word + "$"
        total_lp, count = 0.0, 0
        vocab_size = len(self.counts)
        for i in range(len(padded) - self.n + 1):
            ngram = padded[i: i + self.n]
            context = padded[i: i + self.n - 1]
            num = self.counts.get(ngram, 0) + 1
            den = self.context_counts.get(context, 0) + vocab_size
            total_lp += math.log(num / den)
            count += 1
        return total_lp / max(count, 1)

CORRECT = "correct_spelling"
INCORRECT = "incorrect_spelling"

def classify_word(word: str, dictionary: set, ngram_model: CharNgramModel, ngram_threshold: float = -2.5) -> Tuple[str, str, str]:
    word = unicodedata.normalize("NFC", word.strip())
    if not word: return INCORRECT, "HIGH", "empty string"
    script = classify_word_script(word)

    if script == "numeric": return CORRECT, "HIGH", "pure digit string"
    if script == "latin": return INCORRECT, "MEDIUM", "Latin script word in Devanagari corpus"
    if script == "mixed": return INCORRECT, "HIGH", "mixed-script word (Devanagari + Latin)"
    if script not in ("devanagari", "unknown"): return INCORRECT, "MEDIUM", f"unexpected script: {script}"

    signals = {
        "dict": word in dictionary,
        "unicode": is_valid_unicode_sequence(word) and has_valid_devanagari_structure(word),
        "morph": morphological_plausibility(word) >= 0.6,
        "ngram": ngram_model.log_prob(word) >= ngram_threshold,
        "loanword": _is_devanagari_english(word)
    }

    n_positive = sum(signals.values())
    if signals["loanword"]: return CORRECT, "HIGH", "Devanagari transliteration of English word"
    if not signals["unicode"]: return INCORRECT, "HIGH", "invalid Unicode/Devanagari sequence"
    if signals["dict"] and signals["morph"]: return CORRECT, "HIGH", "in dictionary + morphologically valid"
    if signals["dict"] and not signals["ngram"]: return CORRECT, "MEDIUM", "in dictionary but low n-gram score"
    if signals["dict"]: return CORRECT, "HIGH", "in dictionary"
    
    if n_positive >= 3: return CORRECT, "MEDIUM", "not in dictionary but multiple signals positive"
    if n_positive == 2 and signals["morph"] and signals["ngram"]: return CORRECT, "LOW", "not in dict; morphology + n-gram plausible"
    if n_positive <= 1: return INCORRECT, "MEDIUM", "fails most signals; likely misspelling"
    return INCORRECT, "LOW", "ambiguous; more signals negative than positive"

def classify_wordlist(words: List[str], output_csv: str = None) -> pd.DataFrame:
    dictionary = load_dictionary()
    ngram_model = CharNgramModel(n=3)
    ngram_model.train(list(dictionary))

    records = []
    for word in tqdm(words, desc="Classifying words"):
        label, conf, reason = classify_word(word, dictionary, ngram_model)
        records.append({"word": word, "label": label, "confidence": conf, "reason": reason})

    df = pd.DataFrame(records)
    if output_csv: df.to_csv(output_csv, index=False, encoding="utf-8")
    
    correct = (df["label"] == CORRECT).sum()
    incorrect = (df["label"] == INCORRECT).sum()
    print(f"\nSPELL CHECK SUMMARY\nTotal unique words: {len(df)}\nCorrect spelling: {correct}\nIncorrect spelling: {incorrect}")
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hindi spell checker")
    parser.add_argument("--wordlist", help="Path to newline-delimited word list file")
    parser.add_argument("--out", default=str(RESULTS_DIR / "spell_check_results.csv"))
    args = parser.parse_args()

    if args.wordlist:
        with open(args.wordlist, encoding="utf-8") as f:
            words = [line.strip() for line in f if line.strip()]
    else:
        words = list(SEED_WORDS) + ["हैिं", "करतासा", "कंप्यूटर", "aapka", "थाा", "बोलनाा", "समझाया"]
    df = classify_wordlist(words, output_csv=args.out)
# %%writefile q2_asr_cleanup/number_normalization/normalizer.py
import re
from typing import List, Optional, Tuple

ONES = {"शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पाँच": 5, "पांच": 5, "छह": 6, "छः": 6, "सात": 7, "आठ": 8, "नौ": 9}
TEENS = {"दस": 10, "ग्यारह": 11, "बारह": 12, "तेरह": 13, "चौदह": 14, "पंद्रह": 15, "पन्द्रह": 15, "सोलह": 16, "सत्रह": 17, "अठारह": 18, "उन्नीस": 19}
TENS = {"बीस": 20, "तीस": 30, "चालीस": 40, "पचास": 50, "साठ": 60, "सत्तर": 70, "अस्सी": 80, "नब्बे": 90}
COMPOUND_TENS = {"इक्कीस": 21, "बाईस": 22, "तेईस": 23, "चौबीस": 24, "पच्चीस": 25, "छब्बीस": 26, "सत्ताईस": 27, "अट्ठाईस": 28, "उनतीस": 29, "इकत्तीस": 31, "बत्तीस": 32, "तैंतीस": 33, "चौंतीस": 34, "पैंतीस": 35, "छत्तीस": 36, "सैंतीस": 37, "अड़तीस": 38, "उनतालीस": 39, "इकतालीस": 41, "बयालीस": 42, "तेतालीस": 43, "चवालीस": 44, "पैंतालीस": 45, "छियालीस": 46, "सैंतालीस": 47, "अड़तालीस": 48, "उनचास": 49, "इक्यावन": 51, "बावन": 52, "तिरेपन": 53, "चौवन": 54, "पचपन": 55, "छप्पन": 56, "सत्तावन": 57, "अट्ठावन": 58, "उनसठ": 59, "इकसठ": 61, "बासठ": 62, "तिरेसठ": 63, "चौंसठ": 64, "पैंसठ": 65, "छियासठ": 66, "सड़सठ": 67, "अड़सठ": 68, "उनहत्तर": 69, "इकहत्तर": 71, "बहत्तर": 72, "तिहत्तर": 73, "चौहत्तर": 74, "पचहत्तर": 75, "छिहत्तर": 76, "सतहत्तर": 77, "अठहत्तर": 78, "उनासी": 79, "इक्यासी": 81, "बयासी": 82, "तिरासी": 83, "चौरासी": 84, "पचासी": 85, "छियासी": 86, "सत्तासी": 87, "अट्ठासी": 88, "नवासी": 89, "इक्यानवे": 91, "बानवे": 92, "तिरानवे": 93, "चौरानवे": 94, "पचानवे": 95, "छियानवे": 96, "सत्तानवे": 97, "अट्ठानवे": 98, "निन्यानवे": 99}
MULTIPLIERS = {"सौ": 100, "हज़ार": 1_000, "हजार": 1_000, "लाख": 100_000, "करोड़": 10_000_000, "करोड": 10_000_000}
ALL_NUMBER_TOKENS = set(ONES) | set(TEENS) | set(TENS) | set(COMPOUND_TENS) | set(MULTIPLIERS)
CONNECTORS = {"और", "व"}

def _token_value(tok: str) -> Optional[int]:
    for d in (ONES, TEENS, TENS, COMPOUND_TENS, MULTIPLIERS):
        if tok in d: return d[tok]
    return None

def _is_number_token(tok: str) -> bool: return tok in ALL_NUMBER_TOKENS

def _parse_number_span(tokens: List[str]) -> Optional[int]:
    if not tokens: return None
    total, current = 0, 0
    def flush(multiplier: int) -> int:
        nonlocal current
        if current == 0: current = 1
        res = current * multiplier
        current = 0
        return res

    for tok in tokens:
        val = _token_value(tok)
        if val is None: return None
        if tok in MULTIPLIERS:
            multiplier = MULTIPLIERS[tok]
            if multiplier >= 1_000: total += flush(multiplier)
            else: current = flush(100)
        else: current += val
    total += current
    return total if total > 0 else None

def _extract_number_spans(tokens: List[str]) -> List[Tuple[int, int]]:
    spans, i, n = [], 0, len(tokens)
    while i < n:
        if _is_number_token(tokens[i]):
            start = i
            while i < n and (_is_number_token(tokens[i]) or (tokens[i] in CONNECTORS and i + 1 < n and _is_number_token(tokens[i + 1]))):
                i += 1
            end = i
            while end > start and tokens[end - 1] in CONNECTORS: end -= 1
            spans.append((start, end))
        else: i += 1
    return spans

def normalize_numbers(text: str, apply_idiom_guard: bool = True) -> str:
    if apply_idiom_guard:
        from q2_asr_cleanup.number_normalization.edge_cases import should_skip_conversion
        if should_skip_conversion(text): return text

    tokens = text.split()
    spans  = _extract_number_spans(tokens)

    for start, end in reversed(spans):
        span_tokens = [t for t in tokens[start:end] if t not in CONNECTORS]
        value = _parse_number_span(span_tokens)
        if value is not None: tokens[start:end] = [str(value)]
    return " ".join(tokens)
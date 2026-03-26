# %%writefile q2_asr_cleanup/number_normalization/edge_cases.py
import re
from typing import List

HYPHEN_PAIR_PATTERN = re.compile(
    r"(एक|दो|तीन|चार|पाँच|पांच|छह|सात|आठ|नौ|दस)"
    r"[-–]"
    r"(एक|दो|तीन|चार|पाँच|पांच|छह|सात|आठ|नौ|दस|बीस|तीस)"
)

IDIOMATIC_FOLLOWING = {"दिन", "बातें", "चाँद", "समंदर", "दिशाएं", "यार", "आँसू"}
IDIOM_BLACKLIST = {"दो दिन", "दो-चार", "दो चार", "चार चाँद", "दस बीस", "सात समंदर"}

def should_skip_conversion(text: str) -> bool:
    if HYPHEN_PAIR_PATTERN.search(text): return True
    for idiom in IDIOM_BLACKLIST:
        if idiom in text: return True
    return False

def should_skip_span(span_tokens: List[str], following_token: str = "") -> bool:
    if len(span_tokens) == 1 and following_token in IDIOMATIC_FOLLOWING: return True
    span_text = " ".join(span_tokens)
    for idiom in IDIOM_BLACKLIST:
        if idiom in span_text: return True
    return False
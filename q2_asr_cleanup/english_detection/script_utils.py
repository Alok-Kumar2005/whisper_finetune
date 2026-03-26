# %%writefile q2_asr_cleanup/english_detection/script_utils.py
def char_script(ch: str) -> str:
    cp = ord(ch)
    if 0x0900 <= cp <= 0x097F: return "devanagari"
    if (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A): return "latin"
    if ch.isdigit(): return "digit"
    if ch in ".,!?;:'\"-–—()[]{}": return "punct"
    return "other"

def classify_word_script(word: str) -> str:
    if not word: return "unknown"
    if word.isdigit(): return "numeric"
    deva = sum(1 for c in word if char_script(c) == "devanagari")
    lat  = sum(1 for c in word if char_script(c) == "latin")
    alpha = deva + lat
    if alpha == 0: return "punct" if all(c in ".,!?;:'\"-–—()[]{}/ " for c in word) else "unknown"
    if deva / alpha >= 0.80: return "devanagari"
    if lat  / alpha >= 0.80: return "latin"
    return "mixed"
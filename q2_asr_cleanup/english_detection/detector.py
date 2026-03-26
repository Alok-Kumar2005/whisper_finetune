# %%writefile q2_asr_cleanup/english_detection/detector.py
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from q2_asr_cleanup.english_detection.script_utils import classify_word_script

DEVA_ENGLISH_LOANWORDS = {
    "इंटरव्यू", "इंटरव्यूअर", "कंप्यूटर", "कंप्युटर", "लैपटॉप", "मोबाइल", "फोन", "ऑफिस", "मीटिंग", 
    "प्रेजेंटेशन", "प्रोजेक्ट", "टीम", "मैनेजर", "बॉस", "सैलरी", "जॉब", "इंटर्नशिप", "रिज्यूमे",
    "एचआर", "सीवी", "ऑफर लेटर", "पार्टी", "बर्थडे", "वेबसाइट", "सोशल मीडिया", "फेसबुक", "इंस्टाग्राम",
    "व्हाट्सएप", "ट्विटर", "यूट्यूब", "गूगल", "सर्च", "स्कूल", "कॉलेज", "यूनिवर्सिटी", "एग्जाम", 
    "टेस्ट", "रिजल्ट", "पासवर्ड", "लॉगिन", "बैंक", "लोन", "ईएमआई", "क्रेडिट", "डेबिट", "प्रॉब्लम", 
    "सॉल्यूशन", "इशू", "एरर", "बग", "डेटा", "बस", "ट्रेन", "फ्लाइट", "एयरपोर्ट", "होटल",
}
DEVA_ENGLISH_NGRAMS = {"ंप्य", "ल्ट", "ंट", "व्य", "र्व", "फ्", "ड्", "क्स", "ट्व", "न्स", "ब्ज"}

def _is_devanagari_english(word: str) -> bool:
    if word in DEVA_ENGLISH_LOANWORDS: return True
    if len(word) >= 4 and any(ng in word for ng in DEVA_ENGLISH_NGRAMS): return True
    return False

def tag_english_words(text: str) -> str:
    tokens, tagged = text.split(), []
    for tok in tokens:
        script = classify_word_script(tok)
        if script == "latin" or (script == "devanagari" and _is_devanagari_english(tok)):
            tagged.append(f"[EN]{tok}[/EN]")
        else: tagged.append(tok)
    return " ".join(tagged)

def extract_english_words(text: str) -> List[tuple]:
    results = []
    for tok in text.split():
        script = classify_word_script(tok)
        if script == "latin": results.append((tok, "roman"))
        elif script == "devanagari" and _is_devanagari_english(tok): results.append((tok, "devanagari_transliterated"))
    return results
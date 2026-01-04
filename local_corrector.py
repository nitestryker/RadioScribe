import re
from pathlib import Path
from difflib import SequenceMatcher

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

MODEL_DIR_DEFAULT = "model_corrector_focus"

FORBIDDEN_PATTERNS = [r"\$"]
NON_ENGLISH_BLOCKLIST = re.compile(r"\b(stimme|bitte|danke|bonjour|hola)\b", re.IGNORECASE)

CALLSIGN_RE = re.compile(
    r"^\s*(boy|adam|charles|david|edward|frank|george|henry|ida|john|king|lincoln|mary|"
    r"nancy|nora|ocean|paul|queen|robert|sam|tom|union|victor|william|x-ray|xray|young|yellow|zebra)\s+\d+\b",
    re.IGNORECASE
)

def normalize(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def extract_callsign(s: str):
    m = CALLSIGN_RE.search(s or "")
    return m.group(0).lower() if m else None

def safety_accept(raw: str, pred: str) -> bool:
    raw_n = normalize(raw)
    pred_n = normalize(pred)

    if not pred_n:
        return False

    if any(re.search(p, pred_n) for p in FORBIDDEN_PATTERNS):
        return False

    if NON_ENGLISH_BLOCKLIST.search(pred_n):
        return False

    # Length guards (avoid rambles / repeats)
    if len(pred_n) > max(40, int(len(raw_n) * 1.25)):
        return False

    raw_words = raw_n.split()
    pred_words = pred_n.split()
    if len(pred_words) > max(8, int(len(raw_words) * 1.25)):
        return False

    # Don't drop leading callsign if present
    raw_cs = extract_callsign(raw_n)
    pred_cs = extract_callsign(pred_n)
    if raw_cs and not pred_cs:
        return False

    # Detect simple duplication (first half == second half)
    if len(pred_words) >= 12:
        half = len(pred_words) // 2
        if pred_words[:half] == pred_words[half:half * 2]:
            return False

    # Don't introduce too many novel alphabetic tokens
    raw_alpha = set(re.findall(r"[A-Za-z']+", raw_n.lower()))
    pred_alpha = re.findall(r"[A-Za-z']+", pred_n.lower())
    if len([w for w in pred_alpha if w not in raw_alpha]) > 3:
        return False

    # Decimal comma flip guard
    if re.search(r"\d+\.\d+", raw_n) and re.search(r"\d+,\d+", pred_n):
        return False

    return True

class LocalCorrector:
    def __init__(self, model_dir: str = MODEL_DIR_DEFAULT):
        self.model_dir = model_dir
        self.tokenizer = None
        self.model = None
        self._load()

    def _load(self):
        model_path = Path(self.model_dir)
        # allow relative paths
        if not model_path.exists():
            # still try HF-style resolution; if it fails, raise
            model_path = Path(self.model_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))

        adapter_cfg = model_path / "adapter_config.json"
        if adapter_cfg.exists():
            base = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
            self.model = PeftModel.from_pretrained(base, str(model_path))
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path))

        self.model.eval()

    @torch.inference_mode()
    def _generate(self, text: str) -> str:
        inputs = self.tokenizer([text], return_tensors="pt", truncation=True, max_length=128)
        out = self.model.generate(
            **inputs,
            max_new_tokens=64,
            num_beams=4,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            do_sample=False,
        )
        return normalize(self.tokenizer.decode(out[0], skip_special_tokens=True))

    def correct(self, raw: str) -> str:
        raw_n = normalize(raw)
        if not raw_n:
            return raw_n

        try:
            pred = self._generate(raw_n)
        except Exception:
            return raw_n

        if not safety_accept(raw_n, pred):
            return raw_n

        # Accept only if very close (tiny formatting/punct fixes), not big rewrites
        if similarity(pred.lower(), raw_n.lower()) <= 0.98:
            return raw_n

        return pred

import json
import re
from pathlib import Path
from difflib import SequenceMatcher

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# Change this to "model_corrector" or "model_corrector_focus" depending on which you trained
MODEL_DIR = "model_corrector_focus"

FORBIDDEN_PATTERNS = [r"\$"]

CALLSIGN_RE = re.compile(
    r"^\s*(boy|adam|charles|david|edward|frank|george|henry|ida|john|king|lincoln|mary|"
    r"nancy|ocean|paul|queen|robert|sam|tom|union|victor|william|x-ray|young|zebra)\s+\d+\b",
    re.IGNORECASE
)

NON_ENGLISH_BLOCKLIST = re.compile(r"\b(stimme|bitte|danke|bonjour|hola)\b", re.IGNORECASE)


def normalize(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def has_forbidden(s: str) -> bool:
    for pat in FORBIDDEN_PATTERNS:
        if re.search(pat, s):
            return True
    return False


def extract_leading_callsign(s: str):
    m = CALLSIGN_RE.search(s)
    return m.group(0).lower() if m else None


def safety_accept(raw: str, pred: str) -> bool:
    raw_n = normalize(raw)
    pred_n = normalize(pred)

    # forbid obvious junk
    if has_forbidden(pred_n):
        return False

    # forbid non-English words (we never expect these in radio logs)
    if NON_ENGLISH_BLOCKLIST.search(pred_n):
        return False

    # length explosion guard (tight)
    if len(pred_n) > max(40, int(len(raw_n) * 1.25)):
        return False

    # word explosion guard
    raw_words = raw_n.split()
    pred_words = pred_n.split()
    if len(pred_words) > max(8, int(len(raw_words) * 1.25)):
        return False

    # preserve leading callsign if present in raw
    raw_cs = extract_leading_callsign(raw_n)
    pred_cs = extract_leading_callsign(pred_n)
    if raw_cs and not pred_cs:
        return False

    # crude duplication detector: repeated halves
    if len(pred_words) >= 12:
        half = max(6, len(pred_words) // 2)
        if pred_words[:half] == pred_words[half:half * 2]:
            return False

    # reject if it introduces too many new alphabetic tokens not in raw
    raw_alpha = set(w.lower() for w in re.findall(r"[A-Za-z']+", raw_n))
    pred_alpha = [w.lower() for w in re.findall(r"[A-Za-z']+", pred_n)]
    if pred_alpha:
        new = [w for w in pred_alpha if w not in raw_alpha]
        if len(new) > 3:
            return False

    # decimal comma protection: if raw had 12.3 style, don't allow 12,3
    if re.search(r"\d+\.\d+", raw_n) and re.search(r"\d+,\d+", pred_n):
        return False

    return True


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


class Corrector:
    def __init__(self, model_dir: str):
        self.device = "cpu"

        model_dir_path = Path(model_dir)
        if not model_dir_path.exists():
            raise FileNotFoundError(f"{model_dir} not found")

        # Tokenizer can be loaded from adapter dir if saved there
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # If this is a PEFT adapter directory, load base model + adapter
        adapter_cfg = model_dir_path / "adapter_config.json"
        if adapter_cfg.exists():
            # PEFT adapter: base model will likely be t5-small (or whatever you trained from)
            base_name = "t5-small"
            base_model = AutoModelForSeq2SeqLM.from_pretrained(base_name)
            self.model = PeftModel.from_pretrained(base_model, model_dir)
        else:
            # Full merged model directory
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

        self.model.eval()

    @torch.inference_mode()
    def correct(self, text: str) -> str:
        text = normalize(text)
        inputs = self.tokenizer([text], return_tensors="pt", truncation=True, max_length=128)

        out = self.model.generate(
            **inputs,
            max_new_tokens=64,
            num_beams=4,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            do_sample=False,
        )
        pred = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return normalize(pred)


def main():
      # Prefer val_focus.jsonl from prep_data.py, fallback to val.jsonl
    val_path = Path("val_focus.jsonl") if Path("val_focus.jsonl").exists() else Path("val.jsonl")
    if not val_path.exists():
        raise FileNotFoundError("val.jsonl not found in current folder.")

    rows = []
    with val_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    if not rows:
        raise ValueError("val.jsonl is empty")

    corrector = Corrector(MODEL_DIR)

    total = 0
    accepted = 0
    sum_sim_raw = 0.0
    sum_sim_used = 0.0
    preview = 8

    for i, r in enumerate(rows):
        raw = r["input"]
        target = r["target"]

        pred = corrector.correct(raw)
        ok = safety_accept(raw, pred)

        sim_raw = similarity(normalize(raw).lower(), normalize(target).lower())
        sim_pred = similarity(normalize(pred).lower(), normalize(target).lower())

        # ✅ Accept ONLY if it is safe AND measurably better than RAW
        if ok and (sim_pred >= sim_raw + 0.01):
            used = pred
            accepted += 1
            flag = "accepted+better"
        else:
            used = raw
            flag = "fallback"

        sim_used = similarity(normalize(used).lower(), normalize(target).lower())

        total += 1
        sum_sim_raw += sim_raw
        sum_sim_used += sim_used

        if i < preview:
            print("-" * 70)
            print(f"RAW:    {raw}")
            print(f"TARGET: {target}")
            print(f"PRED:   {pred}")
            print(f"USED:   {used}  ({flag})")
            print(f"sim(raw,target)={sim_raw:.3f}  sim(used,target)={sim_used:.3f}")

    print("\n" + "=" * 70)
    print(f"VAL rows: {total}")
    print(f"Accepted (safe+better): {accepted}/{total} ({accepted/total*100:.1f}%)")
    print(f"Avg similarity RAW→TARGET:  {sum_sim_raw/total:.3f}")
    print(f"Avg similarity USED→TARGET: {sum_sim_used/total:.3f}")


if __name__ == "__main__":
    main()

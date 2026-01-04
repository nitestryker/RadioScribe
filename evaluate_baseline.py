import json
import re
from pathlib import Path
from difflib import SequenceMatcher

# --- Safety rules (conservative) ---
FORBIDDEN_PATTERNS = [
    r"\$",          # no dollar signs
]

CALLSIGN_RE = re.compile(r"^\s*(boy|adam|charles|david|edward|frank|george|henry|ida|john|king|lincoln|mary|nancy|ocean|paul|queen|robert|sam|tom|union|victor|william|x-ray|young|zebra)\s+\d+\b",
                         re.IGNORECASE)

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

    # forbid patterns
    if has_forbidden(pred_n):
        return False

    # don't allow huge length explosions
    if len(pred_n) > max(40, int(len(raw_n) * 1.8)):
        return False

    # if raw begins with a callsign, prediction must also
    raw_cs = extract_leading_callsign(raw_n)
    pred_cs = extract_leading_callsign(pred_n)
    if raw_cs and not pred_cs:
        return False

    return True

# --- Baseline corrector (very simple; replace later with your trained model) ---
def baseline_corrector(raw: str) -> str:
    s = normalize(raw)

    # clear31 -> clear 31 (only when it's "clear" followed by digits)
    s = re.sub(r"\b(clear)(\d+)\b", r"\1 \2", s, flags=re.IGNORECASE)

    # IMPORTANT: Do NOT do Tom->to replacements here.
    # "Tom" is a valid radio callsign word and we don't want false corrections.

    return s

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def main():
    val_path = Path("val.jsonl")
    if not val_path.exists():
        raise FileNotFoundError("val.jsonl not found in current folder.")

    rows = []
    with val_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    if not rows:
        raise ValueError("val.jsonl is empty")

    total = 0
    accepted = 0
    sum_sim_raw = 0.0
    sum_sim_pred = 0.0

    # Show a few examples
    preview = 6

    for i, r in enumerate(rows):
        raw = r["input"]
        target = r["target"]

        pred = baseline_corrector(raw)
        ok = safety_accept(raw, pred)
        final_pred = pred if ok else raw  # fallback

        sim_raw = similarity(normalize(raw).lower(), normalize(target).lower())
        sim_pred = similarity(normalize(final_pred).lower(), normalize(target).lower())

        total += 1
        accepted += 1 if ok else 0
        sum_sim_raw += sim_raw
        sum_sim_pred += sim_pred

        if i < preview:
            print("-" * 70)
            print(f"RAW:    {raw}")
            print(f"TARGET: {target}")
            print(f"PRED:   {pred}")
            print(f"USED:   {final_pred}  ({'accepted' if ok else 'rejected->fallback'})")
            print(f"sim(raw,target)={sim_raw:.3f}  sim(pred,target)={sim_pred:.3f}")

    print("\n" + "=" * 70)
    print(f"VAL rows: {total}")
    print(f"Accepted by safety rules: {accepted}/{total} ({accepted/total*100:.1f}%)")
    print(f"Avg similarity RAW→TARGET:  {sum_sim_raw/total:.3f}")
    print(f"Avg similarity PRED→TARGET: {sum_sim_pred/total:.3f}")

if __name__ == "__main__":
    main()

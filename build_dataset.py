import json
import re
from pathlib import Path

RAW_RE = re.compile(r"^\[RAW\]\s*(.*)\s*$", re.MULTILINE)
ENH_RE = re.compile(r"^\[ENHANCED\]\s*(.*)\s*$", re.MULTILINE)
FIN_RE = re.compile(r"^\[FINAL\]\s*(.*)\s*$", re.MULTILINE)

def normalize_line(s: str) -> str:
    # Keep it conservative: just trim and collapse whitespace
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def split_blocks(text: str):
    # Split on TRAINING MODE markers but keep robust if formatting is messy
    parts = re.split(r"===\s*TRAINING MODE\s*===", text)
    # First chunk before the first marker is not a block
    return parts[1:] if len(parts) > 1 else []

def extract_example(block: str):
    raw_m = RAW_RE.search(block)
    enh_m = ENH_RE.search(block)
    fin_m = FIN_RE.search(block)

    if not raw_m or not enh_m:
        return None

    raw = normalize_line(raw_m.group(1))
    enh = normalize_line(enh_m.group(1))
    fin = normalize_line(fin_m.group(1)) if fin_m else ""

    # Drop useless pairs (empty, or super short noise)
    if len(raw) < 2 or len(enh) < 2:
        return None

    # Avoid training on identical lines that teach nothing (optional but helpful)
    # Keep some identical pairs, but not tons of them.
    identical = (raw.lower() == enh.lower())

    return {
        "input": raw,
        "target": enh,
        "final": fin,
        "identical": identical,
    }

def main():
    in_path = Path("pasted.txt")
    if not in_path.exists():
        raise FileNotFoundError("pasted.txt not found in current folder. Put pasted.txt next to this script.")

    text = in_path.read_text(encoding="utf-8", errors="ignore")
    blocks = split_blocks(text)

    examples = []
    for b in blocks:
        ex = extract_example(b)
        if ex:
            examples.append(ex)

    # Reduce identical pairs so they don't dominate
    non_ident = [e for e in examples if not e["identical"]]
    ident = [e for e in examples if e["identical"]]

    # Keep up to 100% as many identical examples as non-identical ones (1:1 max)
    keep_ident = min(len(ident), len(non_ident))
    ident = ident[:keep_ident]

    final_examples = non_ident + ident

    out_path = Path("dataset.jsonl")
    with out_path.open("w", encoding="utf-8") as f:
        for e in final_examples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print("Done.")
    print(f"Blocks found: {len(blocks)}")
    print(f"Examples extracted (before balancing): {len(examples)}")
    print(f"Examples written (after balancing): {len(final_examples)}")
    print(f"Wrote: {out_path.resolve()}")

if __name__ == "__main__":
    main()

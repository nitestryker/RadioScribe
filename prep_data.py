import json
import re
import random
from pathlib import Path

SEED = 1337

RAW_RE = re.compile(r"^\[RAW\]\s*(.*)\s*$", re.MULTILINE)
ENH_RE = re.compile(r"^\[ENHANCED\]\s*(.*)\s*$", re.MULTILINE)
FIN_RE = re.compile(r"^\[FINAL\]\s*(.*)\s*$", re.MULTILINE)

TRAINING_SPLIT_RE = re.compile(r"===\s*TRAINING MODE\s*===")

def normalize_line(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def split_blocks(text: str):
    parts = re.split(TRAINING_SPLIT_RE, text)
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

    if len(raw) < 2 or len(enh) < 2:
        return None

    identical = (raw.lower() == enh.lower())

    return {
        "input": raw,
        "target": enh,
        "final": fin,
        "identical": identical,
    }

def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def import_incoming(pasted_path: Path, incoming_path: Path):
    """
    If incoming_blocks.txt exists and has content, append only *new* blocks into pasted.txt,
    then back it up and clear it.
    Dedupe is done by exact block text (string match).
    """
    if not incoming_path.exists():
        return {"imported_blocks": 0, "incoming_blocks": 0, "skipped_blocks": 0, "did_import": False}

    incoming_text = incoming_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not incoming_text:
        return {"imported_blocks": 0, "incoming_blocks": 0, "skipped_blocks": 0, "did_import": False}

    pasted_text = pasted_path.read_text(encoding="utf-8", errors="ignore") if pasted_path.exists() else ""

    # Normalize: ensure incoming starts with the marker so split_blocks works reliably
    if not TRAINING_SPLIT_RE.search(incoming_text):
        # if collector is correct, this shouldn't happen, but don't import junk
        return {"imported_blocks": 0, "incoming_blocks": 0, "skipped_blocks": 0, "did_import": False}

    existing_blocks = set(split_blocks(pasted_text))
    incoming_blocks = split_blocks(incoming_text)

    to_add = []
    skipped = 0
    for b in incoming_blocks:
        if b in existing_blocks:
            skipped += 1
        else:
            to_add.append(b)
            existing_blocks.add(b)

    if to_add:
        # Append with marker + blocks
        with pasted_path.open("a", encoding="utf-8") as f:
            for b in to_add:
                f.write("\n=== TRAINING MODE ===\n")
                f.write(b.strip() + "\n")

        # Backup incoming then clear it
        backup = incoming_path.with_suffix(incoming_path.suffix + ".bak")
        backup.write_text(incoming_text, encoding="utf-8")
        incoming_path.write_text("", encoding="utf-8")

    return {
        "imported_blocks": len(to_add),
        "incoming_blocks": len(incoming_blocks),
        "skipped_blocks": skipped,
        "did_import": bool(to_add),
    }

def main():
    pasted_path = Path("pasted.txt")
    incoming_path = Path("incoming_blocks.txt")

    # 1) Import new blocks automatically (if present)
    imp = import_incoming(pasted_path, incoming_path)

    # 2) Build dataset from pasted.txt
    if not pasted_path.exists():
        raise FileNotFoundError("pasted.txt not found in current folder (and no incoming_blocks.txt to create it).")

    text = pasted_path.read_text(encoding="utf-8", errors="ignore")
    blocks = split_blocks(text)

    examples = []
    for b in blocks:
        ex = extract_example(b)
        if ex:
            examples.append(ex)

    # 3) De-dupe exact (input,target) pairs
    seen = set()
    deduped = []
    for e in examples:
        key = (e["input"], e["target"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(e)

    # 4) Write full dataset
    dataset_path = Path("dataset.jsonl")
    write_jsonl(dataset_path, deduped)

    # 5) Build focused sets (mostly non-identical + small amount of identical)
    non_ident = [r for r in deduped if not r.get("identical")]
    ident = [r for r in deduped if r.get("identical")]

    random.seed(SEED)
    random.shuffle(non_ident)
    random.shuffle(ident)

    keep_ident = min(len(ident), max(1, int(0.20 * max(1, len(non_ident)))))
    ident_kept = ident[:keep_ident]

    mixed = non_ident + ident_kept
    random.shuffle(mixed)

    val_size = max(1, int(0.15 * len(mixed)))
    val = mixed[:val_size]
    train = mixed[val_size:]

    write_jsonl(Path("train_focus.jsonl"), train)
    write_jsonl(Path("val_focus.jsonl"), val)

    print("Done.")
    if imp["incoming_blocks"] > 0:
        print(f"Import: incoming_blocks={imp['incoming_blocks']} imported={imp['imported_blocks']} skipped_dupes={imp['skipped_blocks']}")
        if imp["did_import"]:
            print("Import: incoming_blocks.txt was backed up to incoming_blocks.txt.bak and then cleared.")
    else:
        print("Import: no incoming_blocks.txt content to import.")

    print(f"Blocks found (pasted.txt): {len(blocks)}")
    print(f"Examples extracted: {len(examples)}")
    print(f"Examples deduped written: {len(deduped)} -> {dataset_path.resolve()}")
    print(f"Non-identical: {len(non_ident)}")
    print(f"Identical kept: {len(ident_kept)} (from {len(ident)})")
    print(f"Train_focus: {len(train)} -> train_focus.jsonl")
    print(f"Val_focus:   {len(val)} -> val_focus.jsonl")

if __name__ == "__main__":
    main()

import json
import random
from pathlib import Path

SEED = 1337

def read_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def write_jsonl(path: str, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    rows = read_jsonl("dataset.jsonl")
    if not rows:
        raise ValueError("dataset.jsonl empty")

    non_ident = [r for r in rows if not r.get("identical")]
    ident = [r for r in rows if r.get("identical")]

    # Keep some identical, but not too many (20% of non-identical)
    keep_ident = min(len(ident), max(1, int(0.20 * max(1, len(non_ident)))))

    random.seed(SEED)
    random.shuffle(non_ident)
    random.shuffle(ident)

    ident = ident[:keep_ident]
    mixed = non_ident + ident
    random.shuffle(mixed)

    # Split 85/15
    val_size = max(1, int(0.15 * len(mixed)))
    val = mixed[:val_size]
    train = mixed[val_size:]

    write_jsonl("train_focus.jsonl", train)
    write_jsonl("val_focus.jsonl", val)

    print("Done.")
    print(f"Total rows: {len(rows)}")
    print(f"Non-identical: {len(non_ident)}")
    print(f"Identical: {len(ident)} (kept)")
    print(f"Train: {len(train)} -> train_focus.jsonl")
    print(f"Val:   {len(val)} -> val_focus.jsonl")

if __name__ == "__main__":
    main()

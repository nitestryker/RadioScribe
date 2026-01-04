import json
import random
from pathlib import Path

SEED = 1337
VAL_RATIO = 0.15  # 15% validation

def main():
    in_path = Path("dataset.jsonl")
    if not in_path.exists():
        raise FileNotFoundError("dataset.jsonl not found in current folder.")

    rows = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if len(rows) < 20:
        raise ValueError(f"Dataset too small to split reliably: {len(rows)} rows")

    random.seed(SEED)
    random.shuffle(rows)

    val_size = max(1, int(len(rows) * VAL_RATIO))
    val = rows[:val_size]
    train = rows[val_size:]

    train_path = Path("train.jsonl")
    val_path = Path("val.jsonl")

    with train_path.open("w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with val_path.open("w", encoding="utf-8") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # quick stats
    def pct_ident(rows_):
        if not rows_:
            return 0.0
        return 100.0 * sum(1 for r in rows_ if r.get("identical")) / len(rows_)

    print("Done.")
    print(f"Total: {len(rows)}")
    print(f"Train: {len(train)} (identical: {pct_ident(train):.1f}%) -> {train_path.resolve()}")
    print(f"Val:   {len(val)} (identical: {pct_ident(val):.1f}%) -> {val_path.resolve()}")

if __name__ == "__main__":
    main()

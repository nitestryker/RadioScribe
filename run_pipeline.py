import argparse
import subprocess
import sys
from pathlib import Path

def run(cmd):
    print("\n> " + " ".join(cmd))
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        sys.exit(r.returncode)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="Prep data then train the model")
    ap.add_argument("--eval", action="store_true", help="Run evaluation after prep/train")
    args = ap.parse_args()

    print(f"Flags: train={args.train}, eval={args.eval}")

    # Always prep first
    run([sys.executable, "prep_data.py"])

    if args.train:
        if not Path("train_t5_lora.py").exists():
            raise FileNotFoundError("train_t5_lora.py not found in this folder.")
        run([sys.executable, "train_t5_lora.py"])

    if args.eval:
        if not Path("evaluate_model.py").exists():
            raise FileNotFoundError("evaluate_model.py not found in this folder.")
        run([sys.executable, "evaluate_model.py"])

    print("\nPipeline complete.")

if __name__ == "__main__":
    main()

import os
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from peft import LoraConfig, get_peft_model, TaskType


MODEL_NAME = "t5-small"
OUT_DIR = "model_corrector_focus"
MAX_INPUT_LEN = 128
MAX_TARGET_LEN = 128


def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    train_path = Path("train_focus.jsonl")
    val_path = Path("val_focus.jsonl")
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("train.jsonl and/or val.jsonl not found in current folder.")

    train_rows = load_jsonl(str(train_path))
    val_rows = load_jsonl(str(val_path))

    # Prepare datasets with just the fields we train on
    train_ds = Dataset.from_list([{"input": r["input"], "target": r["target"]} for r in train_rows])
    val_ds = Dataset.from_list([{"input": r["input"], "target": r["target"]} for r in val_rows])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # LoRA config for seq2seq
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)

    def preprocess(batch):
        inputs = tokenizer(
            batch["input"],
            max_length=MAX_INPUT_LEN,
            truncation=True,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["target"],
                max_length=MAX_TARGET_LEN,
                truncation=True,
            )
        inputs["labels"] = labels["input_ids"]
        return inputs

    train_tok = train_ds.map(preprocess, batched=True, remove_columns=["input", "target"])
    val_tok = val_ds.map(preprocess, batched=True, remove_columns=["input", "target"])

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    use_cuda = torch.cuda.is_available()
    print(f"CUDA available: {use_cuda}")

    args = Seq2SeqTrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=3e-4,
    num_train_epochs=12,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    predict_with_generate=True,
    fp16=use_cuda,
    report_to="none",
)

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save tokenizer + LoRA adapter
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(OUT_DIR)
    model.save_pretrained(OUT_DIR)

    print(f"\nSaved model to: {Path(OUT_DIR).resolve()}")


if __name__ == "__main__":
    # Avoid tokenizer parallelism warning spam
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()

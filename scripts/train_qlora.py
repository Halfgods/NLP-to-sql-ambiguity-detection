"""QLoRA fine-tuning: Qwen2.5-Coder-1.5B-Instruct on 5k SQL teacher pairs.

4-bit NF4 base, bf16 LoRA adapters (r=16, all attention + MLP projections),
completion-only loss (schema/question tokens masked out), 2 epochs.

Usage:
    .venv/bin/python scripts/train_qlora.py [--epochs 2] [--out adapters/sql-lora]
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          DataCollatorForSeq2Seq, Trainer, TrainingArguments)
from peft import (LoraConfig, PeftModel, get_peft_model,
                  prepare_model_for_kbit_training)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_runtime import SQL_SYSTEM_PROMPT  # shared with serving/UI  # noqa: E402

BASE = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MAX_LEN = 1024


def build_examples(path: Path, tok, max_len: int):
    skipped = 0
    rows = [json.loads(line) for line in open(path)]
    random.Random(42).shuffle(rows)
    out = []
    for r in rows:
        mode = str(r.get("mode", "READ_ONLY")).upper()
        if mode not in {"READ_ONLY", "WRITE"}:
            raise ValueError(f"unsupported operation mode {mode!r}")
        user_text = (f"Operation mode: {mode}\n\n{r['schema']}\n\n"
                     f"Question: {r['question']}\n\nWrite one SQLite statement.")
        msgs = [
            {"role": "system", "content": SQL_SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": r["sql"]},
        ]
        prefix = tok.apply_chat_template(msgs[:2], add_generation_prompt=True, tokenize=False)
        full = tok.apply_chat_template(msgs, tokenize=False)
        ids_p = tok(prefix, add_special_tokens=False).input_ids
        ids_f = tok(full, add_special_tokens=False).input_ids
        if len(ids_f) > max_len or not full.startswith(prefix):
            skipped += 1
            continue
        labels = [-100] * len(ids_p) + ids_f[len(ids_p):]
        out.append({"input_ids": ids_f, "attention_mask": [1] * len(ids_f),
                    "labels": labels, "length": len(ids_f)})
    print(f"examples kept: {len(out)}  skipped(too long): {skipped}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "train_pairs.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "adapters" / "qwen25c-1.5b-sql-lora"))
    ap.add_argument("--resume-adapter", default=None,
                    help="Continue fine-tuning from an existing LoRA adapter")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1024)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map={"": 0}, torch_dtype="auto")
    model.config.use_cache = False

    # simple torch dataset of_dicts for Trainer + DataCollatorForSeq2Seq
    class ListDataset(TorchDataset):
        def __init__(self, rows): self.rows = rows
        def __len__(self): return len(self.rows)
        def __getitem__(self, i): return self.rows[i]


    train_ds = ListDataset(build_examples(Path(args.data), tok, args.max_len))
    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    if args.resume_adapter:
        model = PeftModel.from_pretrained(model, args.resume_adapter,
                                          is_trainable=True)
    else:
        model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    args_tr = TrainingArguments(
        output_dir=str(ROOT / "runs" / "tmp-trainer"), per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum, num_train_epochs=args.epochs,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=15,
        max_grad_norm=0.3, bf16=True, optim="adamw_torch_fused",
        logging_steps=10, save_strategy="no", report_to=[], seed=42,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=2)
    trainer = Trainer(model=model, args=args_tr, train_dataset=train_ds,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True,
                                                           label_pad_token_id=-100))
    t0 = time.time()
    trainer.train()
    print(f"train wall time: {(time.time() - t0) / 60:.1f} min", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print("adapter saved ->", out, flush=True)


if __name__ == "__main__":
    main()

"""Merge the QLoRA adapter into the base model (bf16, CPU) for GGUF export."""

import argparse
import gc
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", default=str(ROOT / "adapters" / "qwen25c-1.5b-sql-lora"))
ap.add_argument("--out", default=str(ROOT / "models" / "merged-sql"))
ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
args = ap.parse_args()

out = Path(args.out)
if (out / "model.safetensors").exists() or any(out.glob("*.safetensors")):
    print("merged model already present ->", out)
    sys.exit(0)

print("loading base (bf16, cpu)...", flush=True)
base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16,
                                            low_cpu_mem_usage=True)
model = PeftModel.from_pretrained(base, args.adapter)
model.eval()
print("merging adapter...", flush=True)
model = model.merge_and_unload(progressbar=True)
del base
gc.collect()

out.mkdir(parents=True, exist_ok=True)
model.save_pretrained(out, safe_serialization=True)
AutoTokenizer.from_pretrained(args.base).save_pretrained(out)
print("merged ->", out, flush=True)

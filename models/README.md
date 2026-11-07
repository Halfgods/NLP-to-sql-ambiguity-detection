# Model checkpoints

Model weights are **not committed** to this repository. They are large
(0.9–3.1 GB each) and are rebuilt locally from the base model and adapters.

Base model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`

Local build path:

```sh
# train / merge adapters (see scripts/train_qlora.py and scripts/merge_lora.py)
scripts/build_gguf.sh        # SQL-only Q4_K_M
scripts/build_write_gguf.sh  # write-adapted Q4_K_M
```

The report's comparison checkpoints are the two Q4_K_M GGUF files produced by
those scripts.

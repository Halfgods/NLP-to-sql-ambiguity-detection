"""Runtime for the fine-tuned, quantized Qwen2.5-Coder-1.5B SQL model.

    base:      Qwen2.5-Coder-1.5B-Instruct (HF)
    fine-tune: QLoRA (4-bit NF4), 5,000 SQL teacher pairs  -> adapters/qwen25c-1.5b-sql-lora
    artifact:  merged -> GGUF F16 -> llama-quantize Q4_K_M (941 MB)
    serving:   llama.cpp llama-server, CPU + mmap, ~1.06 GB resident set

`llama-server` is spawned once (HTTP OpenAI-style endpoint, JSON out) and reused
by the UI worker, the eval script, and anything else. Only this module starts /
stops the server process.
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
GGUF = Path(os.environ.get(
    "CLARION_GGUF",
    str(ROOT / "models" / "qwen25-coder-1.5b-sql-write-q4_k_m.gguf"),
))
SERVER_BIN = ROOT / "llama.cpp" / "build" / "bin" / "llama-server"
PORT = int(os.environ.get("SQL_MODEL_PORT", "18766"))
BASE_URL = f"http://127.0.0.1:{PORT}"
MODEL_NAME = "Qwen2.5-Coder-1.5B-Instruct (SQL + explicit-write QLoRA, GGUF Q4_K_M)"

# One canonical system prompt shared by training, eval and the UI so the merge
# of fine-tuning and runtime prompts is exact.

# The 'give up' sentinel the system prompt mandates for unanswerable asks —
# one contract shared by the prompt and the give-up detector.
GIVE_UP_SQL = "The Answer is not clear. Please try better"

SQL_SYSTEM_PROMPT = (
    "You are an expert SQLite assistant. Follow the operation mode supplied in "
    "the user message. In READ_ONLY mode, output exactly one SELECT or WITH query. "
    "In WRITE mode, output exactly one INSERT or UPDATE statement, and only when "
    "the user explicitly requested that data change. Never output DELETE, DDL, "
    "or multiple statements. If the requested mode does not match the user's "
    f"explicit intent, output {GIVE_UP_SQL}. Reply with SQL only, without prose "
    "or markdown.")

CTX_SIZE = 2048


def _dbg(*a):
    print("[model_runtime]", *a, file=sys.stderr, flush=True)


class LlamaServerRuntime:
    """Owns the llama-server subprocess and exposes a tiny chat API."""

    def __init__(self, gguf_path: Path = GGUF, port: int = PORT):
        self.gguf = Path(gguf_path)
        self.port = port
        self.base = f"http://127.0.0.1:{port}"
        self.proc = None
        self.model_name = MODEL_NAME

    # -- lifecycle -----------------------------------------------------------
    def is_healthy(self, timeout=1.0):
        if self.proc and self.proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"{self.base}/health", timeout=timeout) as r:
                return json.loads(r.read().decode()).get("status") == "ok"
        except Exception:
            return False

    def start(self, wait_s=180.0):
        if self.is_healthy():
            return self
        if not self.gguf.exists():
            raise FileNotFoundError(f"quantized model not found: {self.gguf}")
        if not SERVER_BIN.exists():
            raise FileNotFoundError(
                f"llama-server not built: {SERVER_BIN} (run scripts/build_llama.sh)")
        cmd = [str(SERVER_BIN), "-m", str(self.gguf), "--port", str(self.port),
               "-c", str(CTX_SIZE), "-t", "8", "--no-webui", "-b", "512", "-ub", "256"]
        _dbg("starting llama-server:", " ".join(cmd))
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t0 = time.time()
        while time.time() - t0 < wait_s:
            if self.is_healthy():
                _dbg(f"llama-server ready on :{self.port}, pid={self.proc.pid}")
                return self
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {self.proc.returncode}")
            time.sleep(0.4)
        self.stop()
        raise TimeoutError("llama-server did not become healthy")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=10)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        self.proc = None

    # -- inference -----------------------------------------------------------
    def generate_chat(self, messages, max_tokens=160, temperature=0.0, timeout=120.0):
        payload = {"messages": messages, "max_tokens": max_tokens,
                   "temperature": temperature, "top_p": 1.0, "stream": False}
        req = urllib.request.Request(
            f"{self.base}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode())
        return data["choices"][0]["message"]["content"] or ""

    # -- duck-typing for the UI warmup report ---------------------------------
    def info(self):
        """(tokenizer_like, model_like) for model_worker._warmup()."""
        meta = SimpleNamespace(vocab_size=151936, chat_template="qwen2 chat")
        model = SimpleNamespace(
            device="cpu (llama-server, mmap)", dtype="4-bit Q4_K_M gguf",
            peft_config=None, model=self.gguf.name)
        return meta, model


_RUNTIME = None


def get_runtime(gguf_path: Path = None) -> LlamaServerRuntime:
    """Module-level singleton: start once per process, reuse everywhere."""
    global _RUNTIME
    if _RUNTIME is None or gguf_path is not None:
        _RUNTIME = LlamaServerRuntime(gguf_path=gguf_path or GGUF)
    if not _RUNTIME.is_healthy():
        _RUNTIME.start()
    return _RUNTIME


def shutdown_runtime():
    global _RUNTIME
    if _RUNTIME is not None:
        _RUNTIME.stop()
        _RUNTIME = None


def load_model():
    """Compatibility shim for UI/server/model_worker.py warm-up.

    Starts the quantized model server and returns (tokenizer_like, model_like).
    """
    rt = get_runtime()
    return rt.info()

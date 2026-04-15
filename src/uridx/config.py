import os
import socket
from pathlib import Path

URIDX_DB_PATH = Path(os.getenv("URIDX_DB_PATH", Path.home() / ".local/share/uridx/uridx.db"))

URIDX_EMBEDDINGS = os.getenv("URIDX_EMBEDDINGS", "fastembed")  # "fastembed" or "ollama"
FASTEMBED_MODEL = os.getenv("FASTEMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b")

URIDX_MIN_SCORE: float | None = float(v) if (v := os.getenv("URIDX_MIN_SCORE")) else None


def get_machine_id() -> str:
    return os.getenv("URIDX_MACHINE") or socket.gethostname()

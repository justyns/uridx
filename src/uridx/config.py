import os
from pathlib import Path

URIDX_DB_PATH = Path(os.getenv("URIDX_DB_PATH", Path.home() / ".local/share/uridx/uridx.db"))
URIDX_API_URL: str | None = os.getenv("URIDX_API_URL")
URIDX_AUTH_TOKEN: str | None = os.getenv("URIDX_AUTH_TOKEN")
URIDX_OAUTH_PASSWORD: str | None = os.getenv("URIDX_OAUTH_PASSWORD")
URIDX_BASE_URL: str | None = os.getenv("URIDX_BASE_URL")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b")

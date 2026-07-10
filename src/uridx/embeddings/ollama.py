import sys
import time

import httpx

from uridx.config import OLLAMA_BASE_URL, normalize_ollama_url
from uridx.embeddings import EmbeddingBackendError

# Read timeout must cover a full 64-chunk batch on a contended server (model
# reload / other models hogging it); connect stays short so a down host fails fast.
DEFAULT_TIMEOUT = httpx.Timeout(300.0, connect=10.0)
ATTEMPTS = 3
# Cap inputs per /api/embed request: one giant record (a several-MB email can
# yield thousands of chunks) must not outrun the read timeout in a single call.
MAX_INPUTS_PER_REQUEST = 64


def _url(base_url: str, path: str) -> str:
    """Build a native Ollama API URL, tolerating an OpenAI-compat /v1 suffix on base_url."""
    return f"{normalize_ollama_url(base_url)}{path}"


def _warn_retry(attempt: int, exc: Exception) -> None:
    print(f"Warning: {type(exc).__name__} from Ollama, retrying ({attempt}/{ATTEMPTS - 1})...", file=sys.stderr)


def _post(client: httpx.Client, url: str, payload: dict) -> dict:
    """POST with retries on transient network errors (timeouts, dropped connections)."""
    for attempt in range(1, ATTEMPTS + 1):
        try:
            response = client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise EmbeddingBackendError(
                f"HTTP {e.response.status_code} from {e.request.url.path}: {e.response.text.strip()}"
            ) from e
        except httpx.TransportError as e:
            if attempt == ATTEMPTS:
                raise EmbeddingBackendError(
                    f"embedding backend unreachable: {type(e).__name__}: {e} (OLLAMA_BASE_URL={OLLAMA_BASE_URL})"
                ) from e
            _warn_retry(attempt, e)
            time.sleep(2 * attempt)


def get_dimension(model: str, base_url: str) -> int:
    with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
        data = _post(client, _url(base_url, "/api/show"), {"name": model})
    model_info = data.get("model_info", {})
    for key, value in model_info.items():
        if "embedding_length" in key.lower():
            return value

    test_embedding = get_embeddings_sync(["test"], model, base_url)[0]
    return len(test_embedding)


def get_embeddings_sync(texts: list[str], model: str, base_url: str) -> list[list[float]]:
    url = _url(base_url, "/api/embed")
    embeddings: list[list[float]] = []
    with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
        for i in range(0, len(texts), MAX_INPUTS_PER_REQUEST):
            batch = texts[i : i + MAX_INPUTS_PER_REQUEST]
            embeddings += _post(client, url, {"model": model, "input": batch})["embeddings"]
    return embeddings

import asyncio
import sys
import time

import httpx

from uridx.config import normalize_ollama_url

# Read timeout must cover a full 64-chunk batch on a contended server (model
# reload / other models hogging it); connect stays short so a down host fails fast.
DEFAULT_TIMEOUT = httpx.Timeout(300.0, connect=10.0)
ATTEMPTS = 3


def _url(base_url: str, path: str) -> str:
    """Build a native Ollama API URL, tolerating an OpenAI-compat /v1 suffix on base_url."""
    return f"{normalize_ollama_url(base_url)}{path}"


def _warn_retry(attempt: int, exc: Exception) -> None:
    print(f"Warning: {type(exc).__name__} from Ollama, retrying ({attempt}/{ATTEMPTS - 1})...", file=sys.stderr)


def _post(url: str, payload: dict) -> dict:
    """POST with retries on transient network errors (timeouts, dropped connections)."""
    for attempt in range(1, ATTEMPTS + 1):
        try:
            with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.TransportError as e:
            if attempt == ATTEMPTS:
                raise
            _warn_retry(attempt, e)
            time.sleep(2 * attempt)


async def _post_async(url: str, payload: dict) -> dict:
    for attempt in range(1, ATTEMPTS + 1):
        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.TransportError as e:
            if attempt == ATTEMPTS:
                raise
            _warn_retry(attempt, e)
            await asyncio.sleep(2 * attempt)


def get_dimension(model: str, base_url: str) -> int:
    data = _post(_url(base_url, "/api/show"), {"name": model})
    model_info = data.get("model_info", {})
    for key, value in model_info.items():
        if "embedding_length" in key.lower():
            return value

    test_embedding = get_embeddings_sync(["test"], model, base_url)[0]
    return len(test_embedding)


def get_embeddings_sync(texts: list[str], model: str, base_url: str) -> list[list[float]]:
    return _post(_url(base_url, "/api/embed"), {"model": model, "input": texts})["embeddings"]


async def get_embeddings(texts: list[str], model: str, base_url: str) -> list[list[float]]:
    return (await _post_async(_url(base_url, "/api/embed"), {"model": model, "input": texts}))["embeddings"]

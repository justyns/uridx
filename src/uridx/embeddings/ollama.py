import httpx

from uridx.config import normalize_ollama_url

DEFAULT_TIMEOUT = 60.0


def _url(base_url: str, path: str) -> str:
    """Build a native Ollama API URL, tolerating an OpenAI-compat /v1 suffix on base_url."""
    return f"{normalize_ollama_url(base_url)}{path}"


def get_dimension(model: str, base_url: str) -> int:
    with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
        response = client.post(
            _url(base_url, "/api/show"),
            json={"name": model},
        )
        response.raise_for_status()
        data = response.json()

        model_info = data.get("model_info", {})
        for key, value in model_info.items():
            if "embedding_length" in key.lower():
                return value

    test_embedding = get_embeddings_sync(["test"], model, base_url)[0]
    return len(test_embedding)


def get_embeddings_sync(texts: list[str], model: str, base_url: str) -> list[list[float]]:
    with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
        response = client.post(
            _url(base_url, "/api/embed"),
            json={"model": model, "input": texts},
        )
        response.raise_for_status()
        return response.json()["embeddings"]


async def get_embeddings(texts: list[str], model: str, base_url: str) -> list[list[float]]:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        response = await client.post(
            _url(base_url, "/api/embed"),
            json={"model": model, "input": texts},
        )
        response.raise_for_status()
        return response.json()["embeddings"]

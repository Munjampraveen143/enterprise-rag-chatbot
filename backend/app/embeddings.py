import httpx


OLLAMA_EMBED_URL = (
    "http://localhost:11434/api/embeddings"
)

EMBEDDING_MODEL = "nomic-embed-text"


async def generate_embedding(
    text: str,
) -> list[float]:

    if not text or not text.strip():

        raise ValueError(
            "Text cannot be empty"
        )

    payload = {
        "model": EMBEDDING_MODEL,
        "prompt": text,
    }

    async with httpx.AsyncClient(
        timeout=120.0
    ) as client:

        response = await client.post(
            OLLAMA_EMBED_URL,
            json=payload,
        )

    if response.status_code != 200:

        raise RuntimeError(
            "Ollama embedding request failed: "
            f"{response.text}"
        )

    data = response.json()

    embedding = data.get(
        "embedding"
    )

    if not embedding:

        raise RuntimeError(
            "Ollama returned an empty embedding"
        )

    return embedding
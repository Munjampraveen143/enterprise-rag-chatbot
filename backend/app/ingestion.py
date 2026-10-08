from typing import List, Dict, Any

from app.embeddings import generate_embedding
from app.vector_store import VectorStore


class DocumentIngestion:

    def __init__(
        self,
        vector_store: VectorStore | None = None,
    ):

        self.vector_store = (
            vector_store
            if vector_store is not None
            else VectorStore(
                dimension=768
            )
        )

    async def add_chunks(
        self,
        chunks: List[str],
        metadata: List[Dict[str, Any]] | None = None,
    ):

        if not chunks:

            return 0

        embeddings = []

        for chunk in chunks:

            embedding = await generate_embedding(
                chunk
            )

            embeddings.append(
                embedding
            )

        self.vector_store.add(
            embeddings=embeddings,
            texts=chunks,
            metadata=metadata,
        )

        return len(chunks)
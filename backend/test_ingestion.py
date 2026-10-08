import hashlib
from pathlib import Path
from typing import List, Dict, Any

from app.embeddings import generate_embedding
from app.vector_store import VectorStore


class DocumentIngestion:

    SUPPORTED_FILE_TYPES = {
        ".pdf",
        ".docx",
        ".txt",
    }

    MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB

    def __init__(
        self,
        vector_store: VectorStore | None = None
    ):

        self.vector_store = (
            vector_store
            if vector_store is not None
            else VectorStore(
                dimension=768
            )
        )

    # =====================================================
    # FILE VALIDATION
    # =====================================================

    def validate_file(
        self,
        filename: str,
        file_size: int
    ) -> None:

        if not filename:
            raise ValueError(
                "Filename is required"
            )

        extension = (
            Path(filename)
            .suffix
            .lower()
        )

        if extension not in self.SUPPORTED_FILE_TYPES:

            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported types: "
                f"{', '.join(sorted(self.SUPPORTED_FILE_TYPES))}"
            )

        if file_size <= 0:

            raise ValueError(
                "File cannot be empty"
            )

        if file_size > self.MAX_FILE_SIZE:

            raise ValueError(
                "File size exceeds the maximum "
                "allowed size of 25 MB"
            )

    # =====================================================
    # FILE HASH
    # =====================================================

    def calculate_file_hash(
        self,
        file_bytes: bytes
    ) -> str:

        if not file_bytes:

            raise ValueError(
                "Cannot calculate hash "
                "for empty file"
            )

        return hashlib.sha256(
            file_bytes
        ).hexdigest()

    # =====================================================
    # BUILD CHUNK METADATA
    # =====================================================

    def build_chunk_metadata(
        self,
        chunks: List[str],
        base_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:

        metadata_list = []

        for index, chunk in enumerate(
            chunks,
            start=1
        ):

            metadata = dict(
                base_metadata
            )

            metadata[
                "chunk_id"
            ] = index

            metadata[
                "chunk_length"
            ] = len(chunk)

            metadata_list.append(
                metadata
            )

        return metadata_list

    # =====================================================
    # EMBEDDINGS
    # =====================================================

    async def generate_embeddings(
        self,
        chunks: List[str]
    ) -> List[List[float]]:

        if not chunks:

            return []

        embeddings = []

        for index, chunk in enumerate(
            chunks,
            start=1
        ):

            if not chunk or not chunk.strip():

                continue

            embedding = (
                await generate_embedding(
                    chunk
                )
            )

            embeddings.append(
                embedding
            )

            print(
                f"Generated embedding "
                f"{index}/{len(chunks)}"
            )

        return embeddings

    # =====================================================
    # ADD CHUNKS
    # =====================================================

    async def add_chunks(
        self,
        chunks: List[str],
        metadata: List[Dict[str, Any]] | None = None
    ) -> int:

        if not chunks:

            return 0

        if metadata is None:

            metadata = [
                {}
                for _ in chunks
            ]

        if len(metadata) != len(chunks):

            raise ValueError(
                "Number of metadata records "
                "must match number of chunks"
            )

        # Remove empty chunks while keeping
        # metadata aligned.

        valid_chunks = []
        valid_metadata = []

        for chunk, chunk_metadata in zip(
            chunks,
            metadata
        ):

            if not chunk or not chunk.strip():
                continue

            valid_chunks.append(
                chunk.strip()
            )

            valid_metadata.append(
                chunk_metadata
            )

        if not valid_chunks:

            return 0

        # Generate embeddings.

        embeddings = (
            await self.generate_embeddings(
                valid_chunks
            )
        )

        if len(embeddings) != len(
            valid_chunks
        ):

            raise RuntimeError(
                "Embedding count does not "
                "match chunk count"
            )

        # Store vectors and metadata.

        self.vector_store.add(
            embeddings=embeddings,
            texts=valid_chunks,
            metadata=valid_metadata
        )

        return len(valid_chunks)

    # =====================================================
    # INGEST DOCUMENT
    # =====================================================

    async def ingest_document(
        self,
        filename: str,
        file_bytes: bytes,
        chunks: List[str],
        document_id: int | None = None,
        page_number: int | None = None,
        section: str | None = None
    ) -> Dict[str, Any]:

        # -----------------------------------------------
        # Validate file
        # -----------------------------------------------

        file_size = len(
            file_bytes
        )

        self.validate_file(
            filename=filename,
            file_size=file_size
        )

        # -----------------------------------------------
        # Calculate SHA-256
        # -----------------------------------------------

        file_hash = (
            self.calculate_file_hash(
                file_bytes
            )
        )

        # -----------------------------------------------
        # File type
        # -----------------------------------------------

        file_type = (
            Path(filename)
            .suffix
            .lower()
        )

        # -----------------------------------------------
        # Base metadata
        # -----------------------------------------------

        base_metadata = {
            "document_id": document_id,
            "source": filename,
            "file_type": file_type,
            "file_hash": file_hash,
        }

        if page_number is not None:

            base_metadata[
                "page_number"
            ] = page_number

        if section is not None:

            base_metadata[
                "section"
            ] = section

        # -----------------------------------------------
        # Build metadata for chunks
        # -----------------------------------------------

        chunk_metadata = (
            self.build_chunk_metadata(
                chunks=chunks,
                base_metadata=base_metadata
            )
        )

        # -----------------------------------------------
        # Add to vector store
        # -----------------------------------------------

        stored_chunks = (
            await self.add_chunks(
                chunks=chunks,
                metadata=chunk_metadata
            )
        )

        # -----------------------------------------------
        # Return ingestion information
        # -----------------------------------------------

        return {
            "filename": filename,
            "file_hash": file_hash,
            "file_type": file_type,
            "file_size": file_size,
            "document_id": document_id,
            "chunk_count": stored_chunks,
        }
import faiss
import numpy as np

from pathlib import Path
from typing import Any, Dict, List


class VectorStore:

    def __init__(
        self,
        dimension: int = 768,
        storage_dir: str = "vector_data",
    ):
        self.dimension = dimension

        self.storage_dir = Path(
            storage_dir
        )

        self.storage_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.index_path = (
            self.storage_dir / "index.faiss"
        )

        self.text_path = (
            self.storage_dir / "texts.npy"
        )

        self.metadata_path = (
            self.storage_dir / "metadata.npy"
        )

        # -------------------------------------------------
        # LOAD OR CREATE FAISS INDEX
        # -------------------------------------------------

        if self.index_path.exists():

            self.index = faiss.read_index(
                str(self.index_path)
            )

            if (
                self.index.d !=
                self.dimension
            ):
                raise RuntimeError(
                    "FAISS index dimension mismatch. "
                    f"Expected {self.dimension}, "
                    f"got {self.index.d}."
                )

        else:

            self.index = faiss.IndexFlatIP(
                self.dimension
            )

        # -------------------------------------------------
        # LOAD TEXTS
        # -------------------------------------------------

        if self.text_path.exists():

            self.texts = np.load(
                self.text_path,
                allow_pickle=True,
            ).tolist()

        else:

            self.texts = []

        # -------------------------------------------------
        # LOAD METADATA
        # -------------------------------------------------

        if self.metadata_path.exists():

            self.metadata = np.load(
                self.metadata_path,
                allow_pickle=True,
            ).tolist()

        else:

            self.metadata = []

        # -------------------------------------------------
        # CONSISTENCY CHECK
        # -------------------------------------------------

        if self.index.ntotal != len(
            self.texts
        ):
            raise RuntimeError(
                "Vector index and text storage "
                "are out of sync."
            )

        if self.index.ntotal != len(
            self.metadata
        ):
            raise RuntimeError(
                "Vector index and metadata storage "
                "are out of sync."
            )

    # =====================================================
    # ADD
    # =====================================================

    def add(
        self,
        embeddings,
        texts,
        metadata=None,
    ):

        if len(embeddings) != len(
            texts
        ):
            raise ValueError(
                "Number of embeddings must "
                "match number of texts."
            )

        if not embeddings:
            return

        if metadata is None:

            metadata = [
                {}
                for _ in texts
            ]

        if len(metadata) != len(
            texts
        ):
            raise ValueError(
                "Number of metadata records "
                "must match number of texts."
            )

        vectors = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if vectors.ndim != 2:

            raise ValueError(
                "Embeddings must be a 2D array."
            )

        if vectors.shape[1] != self.dimension:

            raise ValueError(
                "Embedding dimension mismatch. "
                f"Expected {self.dimension}, "
                f"got {vectors.shape[1]}."
            )

        faiss.normalize_L2(
            vectors
        )

        self.index.add(
            vectors
        )

        self.texts.extend(
            texts
        )

        self.metadata.extend(
            metadata
        )

        self.save()

    # =====================================================
    # SEARCH
    # =====================================================

    def search(
        self,
        query_embedding,
        top_k: int = 5,
    ):

        if self.index.ntotal == 0:
            return []

        if top_k <= 0:
            return []

        query_vector = np.asarray(
            [query_embedding],
            dtype=np.float32,
        )

        if query_vector.shape[1] != self.dimension:

            raise ValueError(
                "Query embedding dimension mismatch. "
                f"Expected {self.dimension}, "
                f"got {query_vector.shape[1]}."
            )

        faiss.normalize_L2(
            query_vector
        )

        distances, indices = (
            self.index.search(
                query_vector,
                min(
                    top_k,
                    self.index.ntotal,
                ),
            )
        )

        results = []

        for similarity, index in zip(
            distances[0],
            indices[0],
        ):

            if index == -1:
                continue

            if index >= len(
                self.texts
            ):
                continue

            metadata = {}

            if index < len(
                self.metadata
            ):

                metadata = (
                    self.metadata[index]
                )

            results.append(
                (
                    self.texts[index],
                    float(similarity),
                    metadata,
                )
            )

        return results

    # =====================================================
    # COUNT
    # =====================================================

    def count(self) -> int:

        return self.index.ntotal

    # =====================================================
    # DELETE BY DOCUMENT
    # =====================================================

    def delete_by_document_id(
        self,
        document_id: int,
    ) -> int:

        keep_indices = []

        deleted_count = 0

        for index, metadata in enumerate(
            self.metadata
        ):

            if metadata.get(
                "document_id"
            ) == document_id:

                deleted_count += 1

            else:

                keep_indices.append(
                    index
                )

        if deleted_count == 0:
            return 0

        # -------------------------------------------------
        # RECONSTRUCT EXISTING VECTORS
        # -------------------------------------------------

        old_vectors = self.index.reconstruct_n(
            0,
            self.index.ntotal,
        )

        if keep_indices:

            kept_vectors = old_vectors[
                keep_indices
            ]

        else:

            kept_vectors = np.empty(
                (
                    0,
                    self.dimension,
                ),
                dtype=np.float32,
            )

        # -------------------------------------------------
        # CREATE NEW INDEX
        # -------------------------------------------------

        self.index = faiss.IndexFlatIP(
            self.dimension
        )

        if len(
            kept_vectors
        ) > 0:

            faiss.normalize_L2(
                kept_vectors
            )

            self.index.add(
                kept_vectors
            )

        # -------------------------------------------------
        # UPDATE TEXTS
        # -------------------------------------------------

        self.texts = [
            self.texts[index]
            for index in keep_indices
        ]

        # -------------------------------------------------
        # UPDATE METADATA
        # -------------------------------------------------

        self.metadata = [
            self.metadata[index]
            for index in keep_indices
        ]

        self.save()

        return deleted_count

    # =====================================================
    # SAVE
    # =====================================================

    def save(self):

        faiss.write_index(
            self.index,
            str(
                self.index_path
            ),
        )

        np.save(
            self.text_path,
            np.array(
                self.texts,
                dtype=object,
            ),
        )

        np.save(
            self.metadata_path,
            np.array(
                self.metadata,
                dtype=object,
            ),
        )
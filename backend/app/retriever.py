import re
from typing import Any, Dict, List, Tuple

from app.embeddings import generate_embedding
from app.vector_store import VectorStore


class Retriever:

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

    # =====================================================
    # TOKENIZATION
    # =====================================================

    def _tokenize(
        self,
        text: str,
    ) -> List[str]:

        return [
            word.lower().strip(
                ".,!?;:()[]{}\"'"
            )
            for word in text.split()
            if len(
                word.strip(
                    ".,!?;:()[]{}\"'"
                )
            ) > 2
        ]

    # =====================================================
    # KEYWORD SCORE
    # =====================================================

    def _keyword_score(
        self,
        query: str,
        text: str,
    ) -> float:

        query_words = set(
            self._tokenize(query)
        )

        text_words = set(
            self._tokenize(text)
        )

        if not query_words:
            return 0.0

        matches = (
            query_words.intersection(
                text_words
            )
        )

        return (
            len(matches)
            / len(query_words)
        )

    # =====================================================
    # PHRASE SCORE
    # =====================================================

    def _phrase_score(
        self,
        query: str,
        text: str,
    ) -> float:

        query_words = self._tokenize(
            query
        )

        text_words = self._tokenize(
            text
        )

        if not query_words:
            return 0.0

        query_lower = " ".join(
            query_words
        )

        text_lower = " ".join(
            text_words
        )

        if query_lower in text_lower:
            return 1.0

        if len(query_words) < 2:
            return 0.0

        total_phrases = (
            len(query_words) - 1
        )

        phrase_matches = 0

        for index in range(
            total_phrases
        ):

            phrase = (
                query_words[index]
                + " "
                + query_words[index + 1]
            )

            if phrase in text_lower:
                phrase_matches += 1

        return (
            phrase_matches
            / total_phrases
        )

    # =====================================================
    # CONTACT QUERY DETECTION
    # =====================================================

    def _is_contact_query(
        self,
        query: str,
    ) -> bool:

        query_lower = query.lower()

        contact_terms = [
            "email",
            "gmail",
            "mail",
            "phone",
            "mobile",
            "contact",
            "contact number",
            "phone number",
            "mobile number",
            "telephone",
        ]

        return any(
            term in query_lower
            for term in contact_terms
        )

    # =====================================================
    # CONTACT SCORE
    # =====================================================

    def _contact_score(
        self,
        query: str,
        text: str,
    ) -> float:

        if not self._is_contact_query(
            query
        ):
            return 0.0

        text_lower = text.lower()

        score = 0.0

        # Email detection
        email_pattern = (
            r"\b[a-zA-Z0-9._%+-]+"
            r"@[a-zA-Z0-9.-]+\."
            r"[a-zA-Z]{2,}\b"
        )

        emails = re.findall(
            email_pattern,
            text,
        )

        if emails:
            score = max(
                score,
                1.0,
            )

        if "gmail" in text_lower:
            score = max(
                score,
                0.95,
            )

        # Phone detection
        phone_pattern = (
            r"(?<!\d)"
            r"(?:\+?\d[\d\s().-]{7,}\d)"
            r"(?!\d)"
        )

        phones = re.findall(
            phone_pattern,
            text,
        )

        if phones:
            score = max(
                score,
                1.0,
            )

        if (
            "phone" in text_lower
            or "mobile" in text_lower
            or "contact number" in text_lower
            or "telephone" in text_lower
        ):
            score = max(
                score,
                0.9,
            )

        return score

    # =====================================================
    # SEMANTIC SCORE
    # =====================================================

    def _semantic_score(
        self,
        similarity: float,
    ) -> float:

        score = (
            similarity + 1.0
        ) / 2.0

        return max(
            0.0,
            min(
                1.0,
                score,
            ),
        )

    # =====================================================
    # METADATA SCORE
    # =====================================================

    def _metadata_score(
        self,
        metadata: Dict[str, Any],
    ) -> float:

        score = 0.0

        if metadata.get("source"):
            score += 0.25

        if metadata.get("page_number"):
            score += 0.25

        if metadata.get("section"):
            score += 0.25

        if metadata.get("file_type"):
            score += 0.25

        return score

    # =====================================================
    # SOURCE QUALITY
    # =====================================================

    def _source_quality_score(
        self,
        metadata: Dict[str, Any],
    ) -> float:

        score = 0.0

        if metadata.get("document_id"):
            score += 0.30

        if metadata.get("source"):
            score += 0.20

        if metadata.get("page_number"):
            score += 0.20

        if metadata.get("section"):
            score += 0.20

        if metadata.get("file_type"):
            score += 0.10

        return score

    # =====================================================
    # HYBRID SCORE
    # =====================================================

    def _hybrid_score(
        self,
        semantic_score: float,
        keyword_score: float,
        phrase_score: float,
        metadata_score: float,
        contact_score: float,
    ) -> float:

        if contact_score > 0:

            return (
                semantic_score * 0.35
                + keyword_score * 0.15
                + phrase_score * 0.10
                + metadata_score * 0.05
                + contact_score * 0.35
            )

        return (
            semantic_score * 0.60
            + keyword_score * 0.20
            + phrase_score * 0.15
            + metadata_score * 0.05
        )

    # =====================================================
    # SOURCE-AWARE SCORE
    # =====================================================

    def _source_aware_score(
        self,
        hybrid_score: float,
        source_quality_score: float,
    ) -> float:

        return (
            hybrid_score * 0.90
            + source_quality_score * 0.10
        )

    # =====================================================
    # DUPLICATE CHECK
    # =====================================================

    def _is_duplicate(
        self,
        text: str,
        selected_texts: List[str],
    ) -> bool:

        normalized_text = (
            " ".join(
                text.lower().split()
            )
        )

        for selected_text in selected_texts:

            normalized_selected = (
                " ".join(
                    selected_text.lower().split()
                )
            )

            if (
                normalized_text
                == normalized_selected
            ):
                return True

        return False

    # =====================================================
    # SOURCE KEY
    # =====================================================

    def _source_key(
        self,
        metadata: Dict[str, Any],
    ) -> str:

        document_id = metadata.get(
            "document_id"
        )

        source = metadata.get(
            "source"
        )

        return str(
            document_id
            or source
            or "unknown"
        )

    # =====================================================
    # RETRIEVE
    # =====================================================

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[
        Tuple[
            str,
            float,
            Dict[str, Any],
        ]
    ]:

        if not query or not query.strip():
            return []

        if top_k <= 0:
            return []

        query = query.strip()

        query_embedding = (
            await generate_embedding(
                query
            )
        )

        semantic_results = (
            self.vector_store.search(
                query_embedding=query_embedding,
                top_k=max(
                    top_k * 5,
                    25,
                ),
            )
        )

        if not semantic_results:
            return []

        scored_results = []

        for (
            text,
            similarity,
            metadata,
        ) in semantic_results:

            if not text or not text.strip():
                continue

            if metadata is None:
                metadata = {}

            semantic_score = (
                self._semantic_score(
                    similarity
                )
            )

            keyword_score = (
                self._keyword_score(
                    query,
                    text,
                )
            )

            phrase_score = (
                self._phrase_score(
                    query,
                    text,
                )
            )

            contact_score = (
                self._contact_score(
                    query,
                    text,
                )
            )

            metadata_score = (
                self._metadata_score(
                    metadata
                )
            )

            hybrid_score = (
                self._hybrid_score(
                    semantic_score,
                    keyword_score,
                    phrase_score,
                    metadata_score,
                    contact_score,
                )
            )

            source_quality_score = (
                self._source_quality_score(
                    metadata
                )
            )

            source_aware_score = (
                self._source_aware_score(
                    hybrid_score,
                    source_quality_score,
                )
            )

            enriched_metadata = dict(
                metadata
            )

            enriched_metadata[
                "semantic_score"
            ] = round(
                semantic_score,
                6,
            )

            enriched_metadata[
                "keyword_score"
            ] = round(
                keyword_score,
                6,
            )

            enriched_metadata[
                "phrase_score"
            ] = round(
                phrase_score,
                6,
            )

            enriched_metadata[
                "contact_score"
            ] = round(
                contact_score,
                6,
            )

            enriched_metadata[
                "metadata_score"
            ] = round(
                metadata_score,
                6,
            )

            enriched_metadata[
                "hybrid_score"
            ] = round(
                hybrid_score,
                6,
            )

            enriched_metadata[
                "source_quality_score"
            ] = round(
                source_quality_score,
                6,
            )

            enriched_metadata[
                "source_aware_score"
            ] = round(
                source_aware_score,
                6,
            )

            scored_results.append(
                (
                    text,
                    float(similarity),
                    enriched_metadata,
                    source_aware_score,
                )
            )

        scored_results.sort(
            key=lambda item: item[3],
            reverse=True,
        )

        final_results = []

        selected_texts = []

        source_counts = {}

        # =================================================
        # DIVERSITY PASS
        # =================================================

        for (
            text,
            similarity,
            metadata,
            source_aware_score,
        ) in scored_results:

            if self._is_duplicate(
                text,
                selected_texts,
            ):
                continue

            source_key = (
                self._source_key(
                    metadata
                )
            )

            current_count = (
                source_counts.get(
                    source_key,
                    0,
                )
            )

            if current_count >= 3:
                continue

            final_results.append(
                (
                    text,
                    similarity,
                    metadata,
                )
            )

            selected_texts.append(
                text
            )

            source_counts[
                source_key
            ] = current_count + 1

            if len(
                final_results
            ) >= top_k:
                break

        # =================================================
        # FILL REMAINING RESULTS
        # =================================================

        if len(
            final_results
        ) < top_k:

            for (
                text,
                similarity,
                metadata,
                source_aware_score,
            ) in scored_results:

                if self._is_duplicate(
                    text,
                    selected_texts,
                ):
                    continue

                final_results.append(
                    (
                        text,
                        similarity,
                        metadata,
                    )
                )

                selected_texts.append(
                    text
                )

                if len(
                    final_results
                ) >= top_k:
                    break

        return final_results
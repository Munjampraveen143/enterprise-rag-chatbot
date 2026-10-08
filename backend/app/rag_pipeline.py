from app.generator import (
    generate_answer,
    rewrite_query,
)
from app.retriever import Retriever


class RAGPipeline:

    def __init__(
        self,
        retriever: Retriever | None = None,
    ):

        self.retriever = (
            retriever
            if retriever is not None
            else Retriever(
                vector_store=None
            )
        )

    # =====================================================
    # PROJECT LIST QUESTION DETECTION
    # =====================================================

    def _is_project_list_question(
        self,
        question: str,
    ) -> bool:

        question_lower = (
            question.lower().strip()
        )

        patterns = [
            "what projects",
            "which projects",
            "list projects",
            "list all projects",
            "all projects",
            "project names",
            "projects has",
            "projects have",
            "projects did",
            "projects worked",
            "projects work",
            "worked on projects",
            "projects of",
            "projects by",
        ]

        return any(
            pattern in question_lower
            for pattern in patterns
        )

    # =====================================================
    # ALGORITHM LIST QUESTION DETECTION
    # =====================================================

    def _is_algorithm_list_question(
        self,
        question: str,
    ) -> bool:

        question_lower = (
            question.lower().strip()
        )

        patterns = [
            "what algorithms",
            "which algorithms",
            "list algorithms",
            "machine learning algorithms",
            "ml algorithms",
            "algorithms does",
            "algorithms know",
            "algorithms used",
            "algorithms learned",
        ]

        return any(
            pattern in question_lower
            for pattern in patterns
        )

    # =====================================================
    # SIMPLE FACT QUESTION DETECTION
    # =====================================================

    def _is_simple_fact_question(
        self,
        question: str,
    ) -> bool:

        question_lower = (
            question.lower().strip()
        )

        patterns = [

            # ---------------------------------------------
            # NAME
            # ---------------------------------------------

            "what is the candidate's name",
            "what is the candidates name",
            "what is candidate name",
            "candidate name",
            "name of the candidate",

            # ---------------------------------------------
            # CGPA
            # ---------------------------------------------

            "what is the candidate's cgpa",
            "what is the candidates cgpa",
            "what is candidate cgpa",
            "candidate cgpa",
            "what is the cgpa",
            "cgpa of the candidate",

            # ---------------------------------------------
            # EMAIL
            # ---------------------------------------------

            "what is the candidate's email",
            "what is the candidates email",
            "what is candidate email",
            "candidate email",
            "candidate's email",
            "email of the candidate",

            # ---------------------------------------------
            # PHONE
            # ---------------------------------------------

            "what is the candidate's phone",
            "what is the candidates phone",
            "what is candidate phone",
            "candidate phone",
            "candidate mobile",
            "candidate's mobile",
            "phone number of the candidate",

            # ---------------------------------------------
            # LINKEDIN
            # ---------------------------------------------

            "what is the candidate's linkedin",
            "what is the candidates linkedin",
            "candidate linkedin",
            "candidate's linkedin",
            "linkedin of the candidate",

            # ---------------------------------------------
            # GITHUB
            # ---------------------------------------------

            "what is the candidate's github",
            "what is the candidates github",
            "candidate github",
            "candidate's github",
            "github of the candidate",

            # ---------------------------------------------
            # CURRENT JOB
            # ---------------------------------------------

            "what is the candidate's current job",
            "what is the candidates current job",
            "candidate current job",
            "current job title",
            "candidate's job title",
            "candidate job title",
            "what job does the candidate have",

            # ---------------------------------------------
            # CURRENT COMPANY
            # ---------------------------------------------

            "which company is the candidate currently working at",
            "which company does the candidate work at",
            "where does the candidate currently work",
            "candidate current company",
            "candidate's current company",
            "current company",
            "current employer",
        ]

        return any(
            pattern in question_lower
            for pattern in patterns
        )

    # =====================================================
    # FACT KEYWORD ENRICHMENT
    # =====================================================

    def _build_fact_search_query(
        self,
        question: str,
    ) -> str:

        question_lower = (
            question.lower().strip()
        )

        # ---------------------------------------------
        # CGPA
        # ---------------------------------------------

        if "cgpa" in question_lower:
            return (
                f"{question} "
                "CGPA education degree university "
                "academic qualification"
            )

        # ---------------------------------------------
        # NAME
        # ---------------------------------------------

        if "name" in question_lower:
            return (
                f"{question} "
                "candidate name full name contact"
            )

        # ---------------------------------------------
        # EMAIL
        # ---------------------------------------------

        if "email" in question_lower:
            return (
                f"{question} "
                "email gmail contact information"
            )

        # ---------------------------------------------
        # PHONE
        # ---------------------------------------------

        if (
            "phone" in question_lower
            or "mobile" in question_lower
        ):
            return (
                f"{question} "
                "phone mobile contact number"
            )

        # ---------------------------------------------
        # LINKEDIN
        # ---------------------------------------------

        if "linkedin" in question_lower:
            return (
                f"{question} "
                "LinkedIn profile social profile"
            )

        # ---------------------------------------------
        # GITHUB
        # ---------------------------------------------

        if "github" in question_lower:
            return (
                f"{question} "
                "GitHub profile repository"
            )

        # ---------------------------------------------
        # CURRENT JOB
        # ---------------------------------------------

        if (
            "current job" in question_lower
            or "job title" in question_lower
        ):
            return (
                f"{question} "
                "current job title Data Analyst "
                "professional experience employment"
            )

        # ---------------------------------------------
        # CURRENT COMPANY
        # ---------------------------------------------

        if (
            "current company" in question_lower
            or "currently working" in question_lower
            or "current employer" in question_lower
        ):
            return (
                f"{question} "
                "current company employer "
                "professional experience"
            )

        return question

    # =====================================================
    # EXHAUSTIVE QUESTION DETECTION
    # =====================================================

    def _is_exhaustive_question(
        self,
        question: str,
    ) -> bool:

        return (
            self._is_project_list_question(
                question
            )
            or
            self._is_algorithm_list_question(
                question
            )
        )

    # =====================================================
    # ANSWER
    # =====================================================

    async def answer(
        self,
        question: str,
        top_k: int = 5,
        history: str = "",
    ) -> dict:

        if not question or not question.strip():

            raise ValueError(
                "Question cannot be empty."
            )

        question = question.strip()

        # =================================================
        # QUESTION TYPE
        # =================================================

        is_fact_question = (
            self._is_simple_fact_question(
                question
            )
        )

        is_exhaustive_question = (
            self._is_exhaustive_question(
                question
            )
        )

        print("\n" + "=" * 70)
        print("RAG PIPELINE")
        print("=" * 70)

        print(
            f"Question: {question}"
        )

        print(
            f"Simple fact question: "
            f"{is_fact_question}"
        )

        print(
            f"Exhaustive question: "
            f"{is_exhaustive_question}"
        )

        # =================================================
        # QUERY REWRITE
        # =================================================

        search_query = await rewrite_query(
            question=question,
            history=history,
        )

        # =================================================
        # FACT QUERY ENRICHMENT
        # =================================================

        if is_fact_question:

            search_query = (
                self._build_fact_search_query(
                    question
                )
            )

            print(
                "Fact search query enrichment applied."
            )

        print(
            f"Search query: {search_query}"
        )

        # =================================================
        # RETRIEVAL SIZE
        # =================================================

        retrieval_top_k = top_k

        # -------------------------------------------------
        # Simple factual questions
        # -------------------------------------------------

        if is_fact_question:

            retrieval_top_k = max(
                10,
                top_k,
            )

            print(
                "Simple factual question detected."
            )

            print(
                f"Retrieval top_k: "
                f"{retrieval_top_k}"
            )

        # -------------------------------------------------
        # Project / algorithm list questions
        # -------------------------------------------------

        if is_exhaustive_question:

            retrieval_top_k = max(
                20,
                top_k,
            )

            print(
                "Exhaustive question detected."
            )

            print(
                f"Retrieval top_k: "
                f"{retrieval_top_k}"
            )

        # =================================================
        # RETRIEVE
        # =================================================

        results = await self.retriever.retrieve(
            query=search_query,
            top_k=retrieval_top_k,
        )

        # =================================================
        # RETRIEVAL DEBUG
        # =================================================

        print("\n" + "-" * 70)
        print("RETRIEVAL DEBUG")
        print("-" * 70)

        print(
            f"Retrieved results: "
            f"{len(results)}"
        )

        for index, (
            chunk_text,
            similarity,
            metadata,
        ) in enumerate(
            results,
            start=1,
        ):

            print("\n" + "-" * 50)

            print(
                f"RESULT #{index}"
            )

            print(
                f"Similarity: "
                f"{similarity}"
            )

            print(
                f"Document: "
                f"{metadata.get('source')}"
            )

            print(
                f"Document ID: "
                f"{metadata.get('document_id')}"
            )

            print(
                f"Page: "
                f"{metadata.get('page_number')}"
            )

            print(
                f"Section: "
                f"{metadata.get('section')}"
            )

            print(
                f"Chunk ID: "
                f"{metadata.get('chunk_id')}"
            )

            print("Chunk text:")

            print(
                chunk_text[:1000]
            )

        print("-" * 70)

        # =================================================
        # NO RESULTS
        # =================================================

        if not results:

            return {
                "answer": (
                    "The information is not available "
                    "in the provided documents."
                ),
                "citations": [],
                "search_query": search_query,
            }

        # =================================================
        # BUILD CONTEXT
        # =================================================

        context_parts = []

        citations = []

        seen_citations = set()

        for (
            index,
            (
                chunk_text,
                similarity,
                metadata,
            ),
        ) in enumerate(
            results,
            start=1,
        ):

            # ---------------------------------------------
            # Metadata
            # ---------------------------------------------

            document_id = metadata.get(
                "document_id"
            )

            document = metadata.get(
                "source"
            )

            page = metadata.get(
                "page_number"
            )

            section = metadata.get(
                "section"
            )

            chunk_id = metadata.get(
                "chunk_id"
            )

            # ---------------------------------------------
            # Context
            # ---------------------------------------------

            context_parts.append(
                f"""
[Retrieved Chunk {index}]

Document ID:
{document_id}

Document:
{document}

Page:
{page}

Section:
{section}

Similarity:
{similarity}

Chunk:
{chunk_text}
""".strip()
            )

            # ---------------------------------------------
            # Citation deduplication
            # ---------------------------------------------

            citation_key = (
                document_id,
                document,
                page,
                section,
            )

            if citation_key in seen_citations:

                continue

            seen_citations.add(
                citation_key
            )

            citations.append(
                {
                    "document_id": document_id,
                    "document": document,
                    "page": page,
                    "section": section,
                    "chunk_id": chunk_id,
                }
            )

        # =================================================
        # COMPLETE CONTEXT
        # =================================================

        context = "\n\n".join(
            context_parts
        )

        print(
            f"Context characters: "
            f"{len(context)}"
        )

        # =================================================
        # GENERATE ANSWER
        # =================================================

        answer = await generate_answer(
            question=question,
            context=context,
            history=history,
        )

        # =================================================
        # RETURN
        # =================================================

        print(
            f"Generated answer: "
            f"{answer}"
        )

        print("=" * 70)

        return {
            "answer": answer,
            "citations": citations,
            "search_query": search_query,
        }
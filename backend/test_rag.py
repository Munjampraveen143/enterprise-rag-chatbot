import asyncio
import json

from app.rag_pipeline import RAGPipeline
from app.retriever import Retriever
from app.vector_store import VectorStore


async def main():

    vector_store = VectorStore(
        dimension=768
    )

    retriever = Retriever(
        vector_store=vector_store
    )

    rag_pipeline = RAGPipeline(
        retriever=retriever
    )

    question = (
        "What machine learning algorithms "
        "does Praveen know?"
    )

    result = await rag_pipeline.answer(
        question=question,
        top_k=5,
    )

    print()
    print("=" * 80)
    print("QUESTION:")
    print(question)

    print()
    print("ANSWER:")
    print(result["answer"])

    print()
    print("CITATIONS:")
    print()

    for index, citation in enumerate(
        result["citations"],
        start=1,
    ):

        print(
            f"[S{index}] "
            f"{citation.get('document')}"
        )

        page = citation.get(
            "page"
        )

        if page is not None:

            print(
                f"    Page: {page}"
            )

        section = citation.get(
            "section"
        )

        if section:

            print(
                f"    Section: {section}"
            )

        print(
            f"    Document ID: "
            f"{citation.get('document_id')}"
        )

        print()

    print("=" * 80)

    print()
    print("RAW JSON:")
    print(
        json.dumps(
            result,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":

    asyncio.run(
        main()
    )
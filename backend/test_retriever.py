import asyncio

from app.retriever import Retriever


async def main():

    query = "What machine learning algorithms does Praveen know?"

    print()
    print("QUERY:")
    print(query)

    print()
    print("RETRIEVED RESULTS:")
    print()

    retriever = Retriever()

    results = await retriever.retrieve(
        query=query,
        top_k=3
    )

    if not results:

        print("No results found.")

        return

    for index, (
        text,
        similarity,
        metadata
    ) in enumerate(
        results,
        start=1
    ):

        print("=" * 80)

        print(
            f"RESULT {index}"
        )

        print(
            f"Similarity: {similarity}"
        )

        print(
            f"Metadata: {metadata}"
        )

        print()

        print("Text:")

        print(text)

        print()


if __name__ == "__main__":

    asyncio.run(main())
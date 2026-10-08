import asyncio

from app.embeddings import generate_embedding
from app.vector_store import VectorStore


async def main():

    store = VectorStore(dimension=768)

    documents = [
        "Employees can apply for annual leave through the HR portal.",
        "The company provides health insurance to all full-time employees.",
        "Employees must reset their password every 90 days.",
        "The IT department handles laptop and software support.",
    ]

    print("Generating embeddings...")

    embeddings = []

    for document in documents:
        embedding = await generate_embedding(document)
        embeddings.append(embedding)

    print("Embeddings generated.")

    store.add(
        embeddings=embeddings,
        texts=documents,
    )

    print("Vectors stored:", store.count())

    query = "How do I request vacation?"

    print("\nQuery:", query)

    query_embedding = await generate_embedding(query)

    results = store.search(
        query_embedding=query_embedding,
        top_k=2,
    )

    print("\nSearch results:")

    for text, distance in results:
        print(f"\nDistance: {distance}")
        print(f"Text: {text}")


if __name__ == "__main__":
    asyncio.run(main())
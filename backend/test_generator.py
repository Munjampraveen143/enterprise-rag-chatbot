import asyncio

from app.generator import generate_answer


async def main():

    question = (
        "What machine learning algorithms "
        "does Praveen know?"
    )

    context = """
Praveen's CV lists the following machine
learning algorithms:

Logistic Regression,
Decision Tree,
Random Forest,
XGBoost.

His CV also mentions Neural Networks,
TensorFlow/Keras, and machine learning
classification projects.
"""

    answer = await generate_answer(
        question=question,
        context=context,
    )

    print("\nQUESTION:")
    print(question)

    print("\nANSWER:")
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
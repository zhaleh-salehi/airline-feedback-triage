import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536


def generate_embedding(text: str) -> list[float]:

    if not text or not text.strip():
        raise ValueError("Tweet text cannot be empty")

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
        dimensions=EMBEDDING_DIMENSIONS,
    )

    embedding = response.data[0].embedding

    if len(embedding) != EMBEDDING_DIMENSIONS:
        raise ValueError("Unexpected embedding dimensions")

    return embedding
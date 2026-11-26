# Embeddings, Pinecone index, retrieval, and an additional
# answer-generation helper	

# When knowledge-base functions are called

# Purpose: Connects your local embedding model to Pinecone. 
# It converts text into vectors, stores or searches those vectors, and 
# returns relevant PDF chunks.




# lru_cache: caches initialized objects so they aren't recreated unnecessarily.
# Groq: creates a Groq API client.
# Pinecone: connects to your vector database.
# ServerlessSpec: specifies the cloud and region for a new Pinecone index.
# SentenceTransformer: loads the embedding model.
# settings: supplies the keys and model configuration.

from functools import lru_cache
from groq import Groq
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from app.config import settings


@lru_cache(maxsize=1)
def clients():
    if not settings.groq_api_key:
        raise ValueError("GROQ_API_KEY is missing.")

    if not settings.pinecone_api_key:
        raise ValueError("PINECONE_API_KEY is missing.")

    groq_client = Groq(api_key=settings.groq_api_key)
    pinecone_client = Pinecone(
        api_key=settings.pinecone_api_key
    )

    return groq_client, pinecone_client


@lru_cache(maxsize=1)
def embedding_model():
    # Downloads the model on first use, then runs locally.
    return SentenceTransformer(settings.embed_model)


def embed(texts: list[str]) -> list[list[float]]:
    model = embedding_model()

    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return vectors.tolist()


def ensure_index():
    _, pc = clients()
    index_name = settings.pinecone_index

    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

    index = pc.Index(index_name)

    # Validate the index before attempting vector operations.
    description = pc.describe_index(index_name)
    dimension = getattr(description, "dimension", None)

    if dimension != 384:
        raise ValueError(
            f"Pinecone index '{index_name}' has dimension "
            f"{dimension}; expected 384. Use a compatible index."
        )

    return index


def search_kb(query: str, top_k: int = 5):
    index = ensure_index()
    vector = embed([query])[0]

    result = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True,
    )

    return [
        {
            "page": match.metadata.get("page"),
            "text": match.metadata.get("text", ""),
            "score": round(match.score, 3),
            "source": match.metadata.get(
                "source", "Medical knowledge base"
            ),
        }
        for match in result.matches
        if match.score >= 0.30
    ]


def generate_answer(
    question: str,
    context: str = "",
) -> str:
    groq_client, _ = clients()

    response = groq_client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are MedGuide AI, a medical information "
                    "assistant. Provide clear, cautious, "
                    "evidence-grounded general information. "
                    "Use the supplied context when relevant. "
                    "Do not invent medical facts or citations. "
                    "If context is insufficient, say so. "
                    "Do not claim to diagnose a patient. "
                    "For emergencies, advise immediate medical "
                    "attention."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Medical reference context:\n{context}\n\n"
                    f"User question:\n{question}"
                ),
            },
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content or ""






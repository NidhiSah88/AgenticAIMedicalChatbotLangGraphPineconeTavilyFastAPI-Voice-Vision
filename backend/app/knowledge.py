
# # Import lru_cache to cache the result of a function.
# # This prevents creating new OpenAI and Pinecone clients every time clients() is called.
# from functools import lru_cache

# # Import Pinecone to connect to the vector database.
# # Import ServerlessSpec to configure a serverless Pinecone index.
# from pinecone import Pinecone, ServerlessSpec

# # Import OpenAI to access OpenAI APIs, including text embeddings.
# from openai import OpenAI

# # Import application settings containing API keys, index name, and embedding model.
# from app.config import settings


# # Cache the clients so that the same client instances are reused.
# @lru_cache()
# def clients():

#     # Check that both API keys are configured.
#     # Raise an error early if either key is missing.
#     if not settings.openai_api_key or not settings.pinecone_api_key:
#         raise ValueError(
#             "OpenAI and Pinecone API keys must be set in the environment variables."
#         )

#     # Create and return the OpenAI and Pinecone clients.
#     # The first client communicates with OpenAI.
#     # The second client communicates with Pinecone.
#     return (
#         OpenAI(api_key=settings.openai_api_key),
#         Pinecone(api_key=settings.pinecone_api_key)
#     )


# # Connect to the configured Pinecone index.
# # Create the index if it does not already exist.
# def ensure_index():

#     # Get the cached clients.
#     # '_' means the OpenAI client is not needed in this function.
#     # 'pc' refers to the Pinecone client.
#     _, pc = clients()

#     # Check whether an index with the configured name exists.
#     if not pc.has_index(settings.pinecone_index):

#         # Create a new Pinecone index if it does not exist.
#         pc.create_index(
#             name=settings.pinecone_index,  # Name of the vector index.
#             dimension=1536,                # Number of values in each embedding vector.
#             metric='cosine',               # Measure vector similarity using cosine similarity.
#             spec=ServerlessSpec(
#                 cloud='aws',               # Host the index on AWS.
#                 region='us-east-1'          # AWS region for the index.
#             )
#         )

#     # Return an index handle so other functions can query or update it.
#     return pc.Index(settings.pinecone_index)


# # Convert a list of text strings into embedding vectors.
# def embed(texts):

#     # Get the cached OpenAI client.
#     # '_' means the Pinecone client is not needed here.
#     ai, _ = clients()

#     # Send the text strings to the OpenAI embeddings API.
#     # The selected model converts each text into a numerical vector.
#     # Extract and return the embedding vector for each input text.
#     return [
#         x.embedding
#         for x in ai.embeddings.create(
#             model=settings.openai_embed_model,
#             input=texts
#         ).data
#     ]


# # Search the medical knowledge base for text relevant to a user's question.
# def search_kb(query, top_k=5):

#     # Get a handle to the Pinecone index.
#     index = ensure_index()

#     # Convert the user's question into an embedding vector.
#     # [query] wraps the single question in a list because embed() accepts lists.
#     # [0] extracts the first and only vector from the returned list.
#     vector = embed([query])[0]

#     # Search Pinecone for the top_k most similar vectors.
#     # include_metadata=True also returns the stored source text and page number.
#     result = index.query(
#         vector=vector,
#         top_k=top_k,
#         include_metadata=True
#     )

#     # Convert Pinecone matches into a simpler list of dictionaries.
#     # Keep only results whose similarity score is at least 0.30.
#     return [
#         {
#             'page': m.metadata.get('page'),  # Original PDF page number.
#             'text': m.metadata.get('text', ''),  # Retrieved medical text.
#             'score': round(m.score, 3),  # Similarity score rounded to 3 decimals.
#             'source': 'Gale Encyclopedia (2002)'  # Source attribution.
#         }
#         for m in result.matches
#         if m.score >= 0.30
#     ]




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






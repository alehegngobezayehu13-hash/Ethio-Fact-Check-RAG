from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

# E5 models are multilingual. Swap to "intfloat/multilingual-e5-small" for a
# lighter/faster model, or "intfloat/multilingual-e5-large" for higher quality.
MODEL_NAME = "intfloat/multilingual-e5-base"


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the model once and reuse it (loading is slow)."""
    return SentenceTransformer(MODEL_NAME)


def embed_passages(texts: list[str]) -> np.ndarray:
    """Embed documents for the knowledge base. Returns shape (n, dim)."""
    # E5 models expect a "passage: " prefix on documents.
    prefixed = [f"passage: {t}" for t in texts]
    return get_model().encode(prefixed, normalize_embeddings=True)


def embed_query(text: str) -> np.ndarray:
    """Embed a user claim. Returns shape (dim,)."""
    # E5 models expect a "query: " prefix on searches.
    return get_model().encode(f"query: {text}", normalize_embeddings=True)


if __name__ == "__main__":
    # Quick manual check: similar meanings in different languages should score high.
    claim = embed_query("The government announced a new education policy.")
    docs = {
        "Amharic (same meaning)": "መንግስት አዲስ የትምህርት ፖሊሲ አወጣ።",
        "Tigrinya (same meaning)": "መንግስቲ ሓድሽ ፖሊሲ ትምህርቲ ኣፍሊጡ።",
        "Afaan Oromo (same meaning)": "Mootummaan imaammata barnoota haaraa labse.",
        "English (unrelated)": "I like playing football on weekends.",
    }
    vectors = embed_passages(list(docs.values()))
    for (name, _), vec in zip(docs.items(), vectors):
        # Vectors are normalized, so dot product == cosine similarity.
        print(f"{name}: {float(np.dot(claim, vec)):.3f}")
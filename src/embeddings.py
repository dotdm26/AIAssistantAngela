import asyncio
from typing import List

from sentence_transformers import SentenceTransformer


class LocalNomicEmbeddings:
    """Local embedding adapter with the same interface used by the agent."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

    def embed_query(self, text: str) -> List[float]:
        # Nomic v1.5 expects task prefixing for best retrieval quality.
        encoded = self.model.encode(
            f"search_document: {text}",
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return encoded.tolist()

    async def aembed_query(self, text: str) -> List[float]:
        return await asyncio.to_thread(self.embed_query, text)


def detect_embedding_dimension(embedding_adapter: LocalNomicEmbeddings) -> int:
    """Query the embeddings model once to learn its vector size."""
    sample_text = "Detect embedding dimension."
    if hasattr(embedding_adapter, "embed_query"):
        embedding = embedding_adapter.embed_query(sample_text)
    else:
        embedding = asyncio.run(embedding_adapter.aembed_query(sample_text))
    return len(embedding)

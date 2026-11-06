"""Embedding providers used by conversation memory and tool routing."""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from typing import Any, List


class LocalNomicEmbeddings:
    """Sentence Transformers adapter retained as the default provider."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

    def _encode(self, text: str, prefix: str) -> List[float]:
        encoded = self.model.encode(
            f"{prefix}: {text}",
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return encoded.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._encode(text, "search_query")

    def embed_document(self, text: str) -> List[float]:
        return self._encode(text, "search_document")

    async def aembed_query(self, text: str) -> List[float]:
        return await asyncio.to_thread(self.embed_query, text)


class OnnxEmbeddings:
    """CPU ONNX Runtime provider for an exported sentence-embedding model.

    ``model_path`` points to a directory containing ``model.onnx`` (or
    ``model.int8.onnx``) and a Hugging Face ``tokenizer.json``. Keeping the
    tokenizer and runtime separate from PyTorch makes this provider suitable
    for a compact Windows installer profile.
    """

    def __init__(self, model_path: str | Path):
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"ONNX embedding directory does not exist: {self.model_path}")
        model_file = next(
            (
                self.model_path / candidate
                for candidate in ("model.int8.onnx", "model.onnx", "embed.onnx")
                if (self.model_path / candidate).exists()
            ),
            None,
        )
        tokenizer_file = self.model_path / "tokenizer.json"
        if model_file is None:
            raise FileNotFoundError(f"No ONNX model found in {self.model_path}")
        if not tokenizer_file.exists():
            raise FileNotFoundError(f"tokenizer.json is missing from {self.model_path}")

        self.model_name = str(self.model_path)
        self._np = np
        self.tokenizer = Tokenizer.from_file(str(tokenizer_file))
        manifest_file = self.model_path / "manifest.json"
        manifest: dict[str, Any] = {}
        if manifest_file.exists():
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        model_name = str(manifest.get("model_name", ""))
        legacy_nomic_prefixes = "nomic" in model_name.lower()
        self.query_prefix = str(
            manifest.get("query_prefix", "search_query: " if legacy_nomic_prefixes else "")
        )
        self.document_prefix = str(
            manifest.get("document_prefix", "search_document: " if legacy_nomic_prefixes else "")
        )
        self.max_length = int(manifest.get("max_length", 256))
        if self.max_length <= 0:
            raise ValueError("ONNX embedding max_length must be greater than zero")
        self.tokenizer.enable_truncation(max_length=self.max_length)
        self.session = ort.InferenceSession(str(model_file), providers=["CPUExecutionProvider"])
        self.inputs = {item.name: item for item in self.session.get_inputs()}

    def _encode(self, text: str, prefix: str) -> List[float]:
        encoding = self.tokenizer.encode(f"{prefix}{text}")
        input_values: dict[str, Any] = {}
        for name, metadata in self.inputs.items():
            if name == "input_ids":
                values = encoding.ids
            elif name == "attention_mask":
                values = encoding.attention_mask
            elif name == "token_type_ids":
                values = [0] * len(encoding.ids)
            else:
                continue
            dtype = self._np.int64 if "int64" in metadata.type else self._np.int32
            input_values[name] = self._np.asarray([values], dtype=dtype)

        outputs = self.session.run(None, input_values)
        hidden = self._np.asarray(outputs[0])
        if hidden.ndim == 3:
            mask = self._np.asarray([encoding.attention_mask], dtype=self._np.float32)[..., None]
            hidden = (hidden * mask).sum(axis=1) / self._np.maximum(mask.sum(axis=1), 1e-9)
        if hidden.ndim != 2:
            raise ValueError(f"Unsupported ONNX embedding output shape: {hidden.shape}")

        vector = hidden[0].astype(self._np.float32)
        norm = float(self._np.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._encode(text, self.query_prefix)

    def embed_document(self, text: str) -> List[float]:
        return self._encode(text, self.document_prefix)

    async def aembed_query(self, text: str) -> List[float]:
        return await asyncio.to_thread(self.embed_query, text)


def create_embedding_provider(memory_config) -> LocalNomicEmbeddings | OnnxEmbeddings:
    """Create the configured provider without importing optional runtimes early."""
    if memory_config.backend == "onnx":
        if not memory_config.model_path:
            raise ValueError("ONNX embedding backend requires memory.model_path or EMBEDDING_MODEL_PATH.")
        return OnnxEmbeddings(memory_config.model_path)
    if memory_config.backend != "sentence-transformers":
        raise ValueError(f"Unsupported embedding backend: {memory_config.backend}")
    return LocalNomicEmbeddings(model_name=memory_config.embedding_model)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def detect_embedding_dimension(embedding_adapter) -> int:
    """Query the embedding model once to learn its vector size."""
    if hasattr(embedding_adapter, "embed_query"):
        embedding = embedding_adapter.embed_query("Detect embedding dimension.")
    else:
        embedding = asyncio.run(embedding_adapter.aembed_query("Detect embedding dimension."))
    return len(embedding)

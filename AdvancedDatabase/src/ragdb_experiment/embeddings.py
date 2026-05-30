from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .io import read_jsonl, write_jsonl


class Embedder:
    def __init__(self, model_name: str, dim: int, mock: bool = False) -> None:
        self.model_name = model_name
        self.dim = dim
        self.mock = mock
        self._model = None

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if self.mock:
            return np.vstack([self._mock_embedding(text) for text in texts])

        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)

        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def _mock_embedding(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big", signed=False)
        rng = np.random.default_rng(seed)
        vector = rng.normal(size=self.dim).astype(np.float32)
        norm = np.linalg.norm(vector)
        return vector / norm


def embed_jsonl(
    input_path: Path,
    output_path: Path,
    text_field: str,
    model_name: str,
    dim: int,
    mock: bool,
    batch_size: int,
) -> None:
    rows = list(read_jsonl(input_path))
    embedder = Embedder(model_name=model_name, dim=dim, mock=mock)
    output_rows: list[dict] = []

    for start in tqdm(range(0, len(rows), batch_size), desc="embedding batches"):
        batch = rows[start : start + batch_size]
        vectors = embedder.encode([row[text_field] for row in batch], batch_size=batch_size)
        for row, vector in zip(batch, vectors):
            output = dict(row)
            output["embedding"] = vector.astype(float).tolist()
            output_rows.append(output)

    write_jsonl(output_path, output_rows)

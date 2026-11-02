"""Symmetric BM25 tokenization and normalized dense cosine search with FAISS."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import threading
from typing import Protocol
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from src.models import DocumentChunk, SearchResult


def lexical_tokens(text: str) -> list[str]:
    """One tokenizer for BOTH documents and queries; retain underscore identifiers."""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def unit_rows(values: np.ndarray) -> np.ndarray:
    rows = np.asarray(values, dtype=np.float32)
    if rows.ndim != 2 or not np.isfinite(rows).all():
        raise ValueError("Embeddings must be a finite 2D array")
    lengths = np.linalg.norm(rows, axis=1, keepdims=True)
    if np.any(lengths <= 1e-12):
        raise ValueError("Zero vectors cannot be normalized for cosine similarity")
    return np.ascontiguousarray(rows / lengths, dtype=np.float32)


class Embedder(Protocol):
    signature: str
    def encode(self, texts: list[str]) -> np.ndarray: ...


class LocalEmbedder:
    """MiniLM via CPU ONNX, with full-content windowed mean pooling.

    Large chunks are partitioned into <=224 WordPiece-token windows before model
    inference. Each window is embedded; token-weighted mean pooling then produces
    one unit vector per source chunk. This is not native 1024-token attention.
    """
    def __init__(self, cache_dir: Path):
        from fastembed import TextEmbedding
        from tokenizers import Tokenizer
        self.model_name = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        if self.model_name != "sentence-transformers/all-MiniLM-L6-v2":
            raise ValueError("This audited pipeline supports all-MiniLM-L6-v2; other models need their own input policy")
        self.lock = threading.RLock()
        self.engine = TextEmbedding(model_name=self.model_name, cache_dir=str(cache_dir), threads=2)
        self.engine.model.tokenizer.enable_truncation(max_length=256)
        self.tokenizer = Tokenizer.from_str(self.engine.model.tokenizer.to_str())
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        model_dir = Path(self.engine.model._model_dir)
        model_file = model_dir / self.engine.model.model_description.model_file
        digest = hashlib.sha256()
        with model_file.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        self.model_sha256 = digest.hexdigest()
        self.signature = json.dumps({"model": self.model_name, "model_sha256": self.model_sha256,
                                     "fastembed": importlib.metadata.version("fastembed"),
                                     "policy": "224-wordpiece-windows-weighted-mean-unit-v1"}, sort_keys=True)

    def _windows(self, text: str) -> list[tuple[str, int]]:
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        if not encoded.ids:
            raise ValueError("Cannot embed empty text")
        windows = []
        for start in range(0, len(encoded.ids), 224):
            end = min(start + 224, len(encoded.ids))
            a, b = encoded.offsets[start][0], encoded.offsets[end - 1][1]
            part = text[a:b]
            # Retokenization may differ at punctuation boundaries: never hide a truncation.
            count = len(self.tokenizer.encode(part, add_special_tokens=False).ids)
            if count > 254:
                raise ValueError("Window exceeds the MiniLM input budget")
            windows.append((part, end - start))
        return windows

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        with self.lock:
            groups = [self._windows(text) for text in texts]
            flat = [part for group in groups for part, _ in group]
            vectors = unit_rows(np.asarray(list(self.engine.embed(flat, batch_size=16))))
            output, offset = [], 0
            for group in groups:
                size = len(group)
                weights = np.array([weight for _, weight in group], dtype=np.float32)
                output.append(np.average(vectors[offset:offset + size], axis=0, weights=weights))
                offset += size
            return unit_rows(np.asarray(output))


class SparseIndex:
    method = "BM25"
    def __init__(self, chunks: list[DocumentChunk]):
        if not chunks:
            raise ValueError("Cannot index an empty corpus")
        self.chunks = chunks
        self.index = BM25Okapi([lexical_tokens(chunk.content) for chunk in chunks])

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        if k <= 0:
            raise ValueError("k must be positive")
        tokens = lexical_tokens(query)
        if not tokens:
            return []
        scores = self.index.get_scores(tokens)
        # Stable tie handling; zero-score/no-match chunks are not fabricated hits.
        order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), self.chunks[i].chunk_id))
        selected = [i for i in order if scores[i] > 0][:k]
        return [SearchResult(chunk=self.chunks[i], score=float(scores[i]), rank=r + 1, method=self.method)
                for r, i in enumerate(selected)]


class DenseIndex:
    method = "Dense cosine"
    def __init__(self, chunks: list[DocumentChunk], embedder: Embedder, cache_dir: Path | None = None):
        if not chunks:
            raise ValueError("Cannot index an empty corpus")
        self.chunks, self.embedder, self.cache_hit = chunks, embedder, False
        key = hashlib.sha256((embedder.signature + "\n" + "\n".join(
            hashlib.sha256(c.content.encode()).hexdigest() for c in chunks)).encode()).hexdigest()
        path = cache_dir / f"{key}.npy" if cache_dir else None
        if path and path.exists():
            vectors = unit_rows(np.load(path, allow_pickle=False))
            self.cache_hit = True
        else:
            vectors = unit_rows(embedder.encode([c.content for c in chunks]))
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp.npy")
                np.save(temporary, vectors, allow_pickle=False)
                temporary.replace(path)
        if len(vectors) != len(chunks):
            raise ValueError("Vector/chunk row count mismatch")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.dimensions = vectors.shape[1]
        self.vector_bytes = vectors.nbytes

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        if k <= 0:
            raise ValueError("k must be positive")
        if not query.strip():
            return []
        vector = unit_rows(self.embedder.encode([query]))
        if vector.shape[1] != self.dimensions:
            raise ValueError("Query and document vector dimensions differ")
        # IndexFlatIP is exact; normalized inner product equals cosine similarity.
        # Search all rows so ties at the K boundary have a deterministic ID order.
        scores, rows = self.index.search(vector, self.index.ntotal)
        candidates = [(int(i), float(score)) for i, score in zip(rows[0], scores[0]) if i >= 0]
        candidates.sort(key=lambda pair: (-pair[1], self.chunks[pair[0]].chunk_id))
        return [SearchResult(chunk=self.chunks[i], score=score, rank=r + 1, method=self.method)
                for r, (i, score) in enumerate(candidates[:k])]

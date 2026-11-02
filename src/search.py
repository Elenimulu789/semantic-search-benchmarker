"""Build comparable retrievers; no evaluation file is read by this module."""
from dataclasses import dataclass
from pathlib import Path
import time
from src.chunking import chunk_corpus, load_corpus
from src.indexer import DenseIndex, LocalEmbedder, SparseIndex
from src.models import SearchResult

STRATEGIES = ["fixed-256", "fixed-512", "fixed-1024", "structure"]


@dataclass
class SearchConfiguration:
    name: str
    strategy: str
    method: str
    index: SparseIndex | DenseIndex
    build_ms: float

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        return self.index.search(query, k)


def build_workbench(root: Path) -> tuple[dict[str, str], list[SearchConfiguration], LocalEmbedder]:
    documents = load_corpus(root / "corpus")
    embedder = LocalEmbedder(root / ".cache/models")
    configurations = []
    for strategy in STRATEGIES:
        chunks = chunk_corpus(documents, strategy)
        started = time.perf_counter()
        sparse = SparseIndex(chunks)
        configurations.append(SearchConfiguration(f"BM25 / {strategy}", strategy, "BM25", sparse,
                                                  (time.perf_counter() - started) * 1000))
        started = time.perf_counter()
        dense = DenseIndex(chunks, embedder, root / ".cache/embeddings")
        configurations.append(SearchConfiguration(f"Dense / {strategy}", strategy, "Dense", dense,
                                                  (time.perf_counter() - started) * 1000))
    return documents, configurations, embedder

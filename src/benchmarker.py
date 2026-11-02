"""Evidence-span ground truth, consistent top-K metrics, and measured latency."""
import json
from pathlib import Path
import time
import numpy as np
from src.models import EvalQuery, SearchResult
from src.search import SearchConfiguration


def load_queries(path: Path) -> list[EvalQuery]:
    return [EvalQuery.model_validate(item) for item in json.loads(path.read_text())]


def resolve_evidence(query: EvalQuery, documents: dict[str, str]) -> list[tuple[str, int, int]]:
    output = []
    for label in query.relevant:
        if label.doc not in documents:
            raise ValueError(f"{query.query_id}: missing document {label.doc}")
        text, cursor, found = documents[label.doc], 0, -1
        for _ in range(label.occurrence + 1):
            found = text.find(label.quote, cursor)
            if found < 0:
                raise ValueError(f"{query.query_id}: evidence quote not found in {label.doc}: {label.quote!r}")
            cursor = found + len(label.quote)
        output.append((label.doc, found, found + len(label.quote)))
    if len(set(output)) != len(output):
        raise ValueError(f"{query.query_id}: duplicated evidence labels")
    return output


def result_evidence(result: SearchResult, spans: list[tuple[str, int, int]]) -> set[int]:
    chunk = result.chunk
    return {i for i, (doc, start, end) in enumerate(spans)
            if doc == chunk.parent_doc and chunk.metadata["char_start"] <= start
            and chunk.metadata["char_end"] >= end}


def metrics_at_k(ranked_evidence: list[set[int]], relevant: set[int], k: int) -> dict[str, float]:
    """Recall counts unique evidence units, not duplicate overlapping chunks."""
    if k <= 0 or not relevant:
        raise ValueError("k and the relevant evidence set must be nonempty")
    top = ranked_evidence[:k]
    found = set().union(*top) & relevant if top else set()
    rank = next((r for r, hits in enumerate(top, 1) if hits & relevant), None)
    return {"hit_rate": float(bool(found)), "recall": len(found) / len(relevant),
            "mrr": 1.0 / rank if rank is not None else 0.0}


def benchmark(documents: dict[str, str], configurations: list[SearchConfiguration],
              queries: list[EvalQuery], ks: tuple[int, ...] = (3, 5, 10), repeats: int = 3,
              progress=None) -> dict:
    if not queries or repeats < 1 or not ks or min(ks) < 1:
        raise ValueError("Require queries, positive Ks, and at least one timing repetition")
    ids = [q.query_id for q in queries]
    if len(set(ids)) != len(ids):
        raise ValueError("Evaluation query IDs must be unique")
    labels = {q.query_id: resolve_evidence(q, documents) for q in queries}
    # Fail explicitly if a strategy loses a label; never silently shrink the denominator.
    for config in configurations:
        for query in queries:
            spans = labels[query.query_id]
            covered = set().union(*(result_evidence(
                SearchResult(chunk=c, score=0, rank=1, method=config.method), spans)
                for c in config.index.chunks))
            if covered != set(range(len(spans))):
                raise ValueError(f"{config.name} cannot cover all labels for {query.query_id}; review evidence span length")
    per_query, summary = [], []
    maximum = max(ks)
    for position, config in enumerate(configurations):
        # Warm-up is excluded. Index construction/model loading are measured separately.
        config.search(queries[0].query, maximum)
        rows, timings = [], []
        for query in queries:
            times, rankings = [], []
            for _ in range(repeats):
                started = time.perf_counter()
                results = config.search(query.query, maximum)
                times.append((time.perf_counter() - started) * 1000)
                rankings.append([r.chunk.chunk_id for r in results])
            if any(r != rankings[0] for r in rankings[1:]):
                raise ValueError(f"Nondeterministic rankings for {config.name}: {query.query_id}")
            timings.extend(times)
            evidence = [result_evidence(r, labels[query.query_id]) for r in results]
            for k in ks:
                metric = metrics_at_k(evidence, set(range(len(labels[query.query_id]))), k)
                record = {"configuration": config.name, "method": config.method, "strategy": config.strategy,
                          "query_id": query.query_id, "query": query.query, "kind": query.kind, "k": k,
                          **metric, "latency_mean_ms": float(np.mean(times)), "timing_samples_ms": times,
                          "top_ids": [r.chunk.chunk_id for r in results[:k]],
                          "first_relevant_rank": next((i for i, hits in enumerate(evidence[:k], 1) if hits), None)}
                rows.append(record)
        per_query.extend(rows)
        for k in ks:
            for group in ("all", "exact", "conceptual"):
                chosen = [r for r in rows if r["k"] == k and (group == "all" or r["kind"] == group)]
                if not chosen:
                    continue
                # Group-specific timing samples rather than copying global timing percentiles.
                group_times = [value for r in chosen for value in r["timing_samples_ms"]]
                record = {"configuration": config.name, "method": config.method, "strategy": config.strategy,
                          "query_group": group, "k": k, "query_count": len(chosen),
                          "hit_rate": float(np.mean([r["hit_rate"] for r in chosen])),
                          "recall": float(np.mean([r["recall"] for r in chosen])),
                          "mrr": float(np.mean([r["mrr"] for r in chosen])),
                          "latency_mean_ms": float(np.mean(group_times)),
                          "latency_p50_ms": float(np.percentile(group_times, 50)),
                          "latency_p95_ms": float(np.percentile(group_times, 95)),
                          "latency_percentile_basis": "individual query/repetition requests",
                          "chunks": len(config.index.chunks), "build_ms": config.build_ms,
                          "embedding_cache_hit": getattr(config.index, "cache_hit", False)}
                summary.append(record)
        if progress:
            progress((position + 1) / len(configurations), config.name)
    return {"summary": summary, "per_query": per_query,
            "protocol": {"ks": list(ks), "repeats": repeats,
                         "index": "FAISS IndexFlatIP, exact cosine on normalized vectors",
                         "recall_unit": "unique labeled source evidence span",
                         "latency": "warm, serial; query embedding + index ranking; excludes build/UI/download",
                         "k_latency": "one search at maximum K, reused for all top-K metric slices"}}

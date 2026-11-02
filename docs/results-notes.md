# Measured results and interpretation

The saved run used the bundled 27-page FastAPI corpus, 24 curated development queries, MiniLM CPU embeddings, exact FAISS cosine search, three timing repetitions, and 32-token overlap for fixed windows. Source commit, model hash, machine information, and protocol are retained in `measured-benchmark.json` and the corpus manifest.

## Overall K=5 results from the actual preparation run

| Configuration | Chunks | Hit Rate@5 | Recall@5 | MRR@5 | Mean latency (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 / 256 | 213 | 0.8333 | 0.7917 | 0.6889 | 0.34 |
| Dense / 256 | 213 | 0.7917 | 0.7708 | 0.6333 | 4.67 |
| BM25 / 512 | 109 | 0.8333 | 0.7917 | 0.6632 | 0.23 |
| Dense / 512 | 109 | 0.8750 | 0.8542 | 0.8438 | 4.52 |
| BM25 / 1024 | 59 | 0.8333 | 0.8125 | 0.6875 | 0.15 |
| Dense / 1024 | 59 | 0.8750 | 0.8542 | 0.8264 | 4.35 |
| BM25 / heading sections | 304 | 0.7917 | 0.7708 | 0.7431 | 0.47 |
| Dense / heading sections | 304 | 0.8750 | 0.8542 | 0.7188 | 4.77 |

Latency varies by computer, warm state, and load. These are observations from one preparation run, not promised performance on the student's laptop. Query latency includes model inference for dense search. Native-process app interaction tests ran separately after the benchmark.

## Two useful video examples

Use **structure** and **K=5**:

1. **Exact identifier:** `WebSocketDisconnect receive_text exception`
   - Observed BM25 relevant rank: **1**.
   - Observed dense relevant rank: **3**.
   - The explicit exception name favors literal matching in this example.
2. **Conceptual question:** `How do I stop overly long text from being accepted as an input?`
   - Observed BM25: **no labeled evidence in the first five results**.
   - Observed dense relevant rank: **1**.
   - The intended answer is string-length validation; dense retrieval matched that meaning without the query naming the parameter.

These examples were selected after inspecting the development results to make the demonstration understandable. They are not a held-out evaluation, and the app never edits rankings to force these outcomes. If a new run differs, describe the actual result.

## What I can reasonably conclude

Dense 512-token chunks achieved the highest MRR@5 in this measured set. The lexical baseline was much faster, and it beat dense retrieval on some individual questions. Heading-based chunking preserved code boundaries, but it did not automatically produce the highest overall ranking score.

Longer chunks can include more answer-bearing text in each result, so higher recall at equal K does not necessarily mean a better experience at an equal context-token budget. Also, the long chunks use pooled short-window embeddings rather than native 1024-token transformer attention.

Before a production decision, collect fresh user questions, label them independently, evaluate at a fixed context budget, and measure latency under concurrent load. A hybrid search experiment would be a reasonable next comparison, but hybrid results are not implemented or claimed here.

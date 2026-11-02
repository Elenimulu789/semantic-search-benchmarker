# AI assistance, prompts, and code audit

## Disclosure and scope

AI assistance was used to prepare the source-ingestion script, chunkers, retrieval code, evaluation harness, Streamlit interface, tests, Docker files, and documentation. The generalized prompts below describe the assistance used; they are not claimed to be a verbatim transcript. Review the code and add your own questions, edits, and observations before submitting.

The saved [initial AI review draft](docs/initial_ai_draft.py.txt) is an isolated, incomplete baseline excerpt, not runtime code. Its two flawed functions were actually executed against simple mathematical examples; the recorded outputs are in [draft-audit-results.json](docs/draft-audit-results.json). No claim is made that these bugs reached a deployed service.

## Prompt log

| Purpose | Generalized prompt | Result to review |
| --- | --- | --- |
| Design | Help me structure a Python documentation search workbench with fixed-token and heading-based chunking, BM25, local dense embeddings, and a separate evaluation dataset. | Modular source files and shared result models |
| Baseline and audit | Provide a small initial ranking and MRR draft, then check its output against known vectors and rankings before using it in the full engine. | Saved review excerpt and actual failing examples |
| Stable ground truth | Explain how to label source passages so changing the chunk size does not change the identity of the correct answer. | Source document plus exact quote/occurrence; map by character offsets |
| Input-length review | Check the selected model's real input limit and avoid comparing 1024-token chunks after silent 256-token truncation. | Model-sized windows and weighted pooled vectors |
| Retrieval QA | Help test original-token overlap, code fences, identifier lookup, real-model paraphrasing, normalization, ties, and misses outside top-K. | Deterministic math tests plus real-model integration tests |
| Dashboard | Help build a side-by-side search view and live benchmark table, charts, exports, and observed failure comparisons. | Streamlit application using the same benchmark engine as the CLI |
| Communication | Help explain the measured results in simple language and organize a 5–7 minute screen-and-face demonstration. | Video notes with actual example queries and limits |

## Concrete flaw 1: similarity ranking reversed

**Saved draft:** `rank_dense()` in `docs/initial_ai_draft.py.txt` calculates a dot product, then sorts it in ascending order. For unit vectors, the dot product is cosine similarity: larger values are better. Sorting it like a distance reverses the ranking.

**Executed example:** document vectors `[1,0]` and `[0,1]`, query `[1,0]`. Expected order: `[0,1]`. Actual draft order: `[1,0]`.

**Corrected code:** `unit_rows()` in `src/indexer.py` normalizes document and query vectors; `DenseIndex.search()` uses FAISS `IndexFlatIP` and sorts higher scores first. Equal scores use chunk IDs for a deterministic tie order.

**Regression test:** `test_cosine_normalization_and_descending_ranking` checks known scores and ranks, including an oversized vector whose magnitude must not decide cosine relevance.

## Concrete flaw 2: missed queries awarded MRR credit

**Saved draft:** `mrr_at_k()` assigns a default rank of 1 when no relevant item is found and searches the entire ranking instead of only the first K items.

**Executed examples:** a complete miss incorrectly receives `1.0` instead of `0.0`; a relevant item at rank 3 with K=2 incorrectly receives `0.3333` instead of `0.0`.

**Corrected code:** `metrics_at_k()` in `src/benchmarker.py` slices the ranking to K, locates the first relevant rank if one exists, and returns zero for a miss. It also counts unique evidence units for recall.

**Regression tests:** `test_known_ranking_math`, `test_missed_query_is_zero_and_top_k_is_enforced`, and duplicate-overlap recall tests.

## Additional line-by-line audit checks

| Topic | Review finding / decision | Code and verification |
| --- | --- | --- |
| Ground-truth leakage | Arbitrary chunk indices would change across sizes. Queries and labels must not enter the embedding corpus. | `load_corpus()` reads Markdown only from `corpus/`; `resolve_evidence()` maps separate labels to source offsets; ingestion and all-strategy coverage tests pass. |
| Distance versus similarity | FAISS inner product is a similarity, not a lower-is-better distance. | Unit vectors, descending score order, numeric tests. |
| BM25 symmetry | Query/document normalization must match. | One `lexical_tokens()` function is called for both paths; mixed-case rare identifier test passes. |
| Empty lexical matches | Returning arbitrary zero-score chunks can look like successful retrieval. | BM25 filters non-positive scores and returns an empty list when no terms match. |
| Fixed code-block slicing | Fixed windows can cut code fences; silently adding repair text would alter the baseline and offsets. | Kept as an explicitly tested baseline limitation. `StructureAwareChunker` ignores headings inside code and retains whole fenced blocks. |
| Model truncation | MiniLM only accepts 256 input tokens; a naive 1024-token input discards content. | <=224 WordPiece-token windows; weighted mean pooling; unit normalization. Test proves the tail is included. |
| Unicode token boundaries | A tiktoken boundary can fall inside a Unicode character. | Adjust window ends to valid UTF-8 boundaries while preserving exact original-token overlap; multilingual reconstruction test passes. |
| Recall inflation | Several overlapping chunks may repeat one answer passage. | Recall uses unique source-evidence units instead of counting repeated chunks. |
| Benchmark timing | Precomputed query vectors would make dense latency misleading. | Every timed dense search embeds the query; document vectors may be cached. Warm-up/build/download/UI are excluded and disclosed. |
| Tie handling | Equivalent scores must not create unexplained ranking changes. | Stable chunk-ID tie ordering and deterministic tests. |

## Actual preparation evidence

- 27 FastAPI Markdown source pages, pinned by commit and checksums.
- 24 curated queries, with 26 labeled evidence spans.
- Eight retrieval/chunk configurations benchmarked at K=3,5,10.
- 28 tests passed, including real MiniLM inference and Streamlit search/benchmark interactions.
- Docker execution and the student's video recording remain to be completed on the student's computer.

Detailed evidence: [verification](docs/verification.md), [pytest output](docs/pytest-output.txt), [measured benchmark JSON](docs/measured-benchmark.json).

## My additional review record

- Functions I personally inspected: ____________________
- Commands I ran and actual results: ____________________
- Changes I made after reviewing the AI draft: ____________________
- Any further AI assistance: ____________________

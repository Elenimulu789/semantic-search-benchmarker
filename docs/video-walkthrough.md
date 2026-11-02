# Simple video notes — target 6 minutes 15 seconds

Record your screen and webcam together. Keep your face visible throughout. Do a practice run so the final video stays between **5 and 7 minutes**.

Before recording: start the app, click **Build / load indexes**, and wait until it is ready. Run the tests once so the model is already downloaded. Open the project files and terminal. For the assignment's container option, start with `docker compose up -d --build` and use the Docker test command below during the video.

These notes describe the saved preparation results. Use your own live results if they differ.

## 0:00–1:00 — What the app does

**Show:** the app and project folder.

“My project is called SearchWorkbench. It compares two ways to search technical documentation. BM25 looks for matching words, while semantic search looks for related meaning.”

“I used 27 pages from the FastAPI documentation and created 24 questions with labeled answer passages. I kept the questions separate from the documents being searched.”

**Open:** `src/chunking.py`.

“I compare chunks of 256, 512, and 1024 tokens with a small overlap. I also split by document headings. The heading version keeps code blocks together. Fixed windows can cut those blocks, which is one weakness I wanted to measure.”

## 1:00–2:30 — AI draft and audit

**Open:** `ai_prompt_log.md`, then `docs/initial_ai_draft.py.txt` and `docs/draft-audit-results.json`.

“I used AI assistance to help prepare the code and documentation. I saved a small review draft and tested its calculations before relying on the full search engine.”

“The first problem was the ranking order. The draft treated similarity like distance and put lower values first. My example showed that it returned the less similar document first.”

**Show:** `DenseIndex.search()` in `src/indexer.py`.

“The corrected version uses normalized vectors and puts the highest similarity first.”

**Show:** `metrics_at_k()` in `src/benchmarker.py`.

“The second problem was MRR. The draft gave a missed answer a score of one and counted answers below the chosen cutoff. The corrected version gives misses zero and only checks the first K results.”

“I also checked the model's input limit. It accepts 256 tokens, so larger chunks are divided into smaller windows and combined. This avoids silently ignoring the end of a long chunk.”

## 2:30–4:00 — Live search comparison

**Choose:** structure chunking, K=5.

**Search:**

```text
WebSocketDisconnect receive_text exception
```

“This is an exact identifier query. In the preparation run, BM25 placed the relevant passage first and dense search placed it third. The specific exception name helps keyword search.”

**Search:**

```text
How do I stop overly long text from being accepted as an input?
```

“This question describes the problem without naming the validation setting. In the preparation run, dense search placed the labeled passage first, while BM25 did not find it in the top five.”

“The highlighted words show literal matches. The scores use different scales, so a BM25 score of ten cannot be directly compared with a cosine score below one.”

## 4:00–5:30 — Benchmark and interpretation

**Open:** Benchmark & failure analysis → click **Run Benchmark**. Keep K=5 and query group all.

“This runs the same questions against both methods and all four chunking strategies. Hit rate tells me how often an answer appears in the results. Recall checks how much of the labeled evidence was found. MRR rewards putting the first correct result closer to the top.”

**Show:** table, chart, and latency chart.

“In the preparation run, dense search with 512-token chunks had the highest MRR at five results. BM25 was much faster. That does not mean one method always wins. The failure panel shows individual questions where each method ranked the answer better.”

“Larger chunks also contain more text, so comparing five large chunks with five small chunks is not an equal text budget. These are only 24 curated questions. I would need more independent questions before making a production decision.”

## 5:30–6:15 — Tests and conclusion

**Run in a second terminal, leaving the app running:**

```powershell
docker compose exec search uv run --frozen pytest -v
```

For a native-only practice run, the equivalent command is `uv run pytest -v`. For the submitted containerized demonstration, show the Docker command actually running.

“The tests check overlap, code blocks, exact identifiers, a real-model paraphrase with no matching words, ranking order, and the metric calculations. They also test the dashboard's search and benchmark buttons.”

**Show the actual test summary.** The preparation run passed 28 tests; read the count from your own run.

“My main takeaway is that semantic search is useful, but exact keyword search still matters. Chunking changes what each result contains, and the evaluation code needs to be audited just as carefully as the search code.”

Upload the recording to YouTube as Unlisted or Public. Add the real link and your repository link to `docs/submission.md`.

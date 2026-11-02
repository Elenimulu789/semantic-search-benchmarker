# Actual preparation verification

## Executed

- Locked dependencies installed with uv on Python 3.11.17.
- The real `sentence-transformers/all-MiniLM-L6-v2` model was downloaded through FastEmbed and run on CPU using ONNX.
- Both the numerical ranking inversion and MRR mistakes in the saved review draft were reproduced. Raw expected/actual values are saved in `draft-audit-results.json`.
- All 27 original source pages were ingested and their checksums verified.
- All 24 queries and 26 evidence spans were checked for valid source matches and coverage under every chunk strategy.
- All eight configurations were benchmarked at K=3/5/10 with three timing repetitions per question/configuration. Real per-query ranks, metrics, timing samples, source/model information, and summaries were saved.
- **28 automated tests passed**. This includes a real-model semantic retrieval test with zero shared lexical tokens, long-input coverage, and Streamlit AppTest interactions with both the search form and the live benchmark button.

Evidence files:

- [pytest-output.txt](pytest-output.txt)
- [pytest-results.xml](pytest-results.xml)
- [measured-benchmark.json](measured-benchmark.json)
- [measured-benchmark.csv](measured-benchmark.csv)
- [draft-audit-results.json](draft-audit-results.json)

## Not executed or created here

- Docker image build and Compose run: no Docker engine is available in the preparation environment. The Docker files are supplied for the student's own containerized demonstration.
- Public cloud deployment: not created; containerized local demonstration is the selected delivery option.
- GitHub repository: not created. The available GitHub integration has no repository-creation operation. Follow the README to create and push an empty repository.
- Screen-and-webcam video: must be recorded by the student.
- Independent browser pixel inspection: not performed. Streamlit AppTest verified app execution, widgets, search, and benchmark interactions; it does not replace visually inspecting the app in the student's browser.

The bundled results are actual preparation evidence, not a claim that the student has already run these checks personally. Complete the record below after running the project.

## My final run record

- Docker Desktop / OS version: ____________________
- `docker compose up -d --build` result: ____________________
- App loaded and both searches demonstrated: ____________________
- Actual pytest count/result: ____________________
- Benchmark completed and exports saved: ____________________
- Browser console / layout check: ____________________
- GitHub URL: ____________________
- Video URL and duration: ____________________

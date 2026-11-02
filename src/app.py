"""Streamlit search comparison and evaluation dashboard."""
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
import pandas as pd
import streamlit as st
from src.benchmarker import benchmark, load_queries
from src.indexer import lexical_tokens
from src.search import build_workbench, STRATEGIES

load_dotenv(ROOT / '.env')
st.set_page_config(page_title='SearchWorkbench', page_icon='🔎', layout='wide')
st.markdown('''<style>
.stApp { background: #f5f7fb; } h1 { letter-spacing: -.04em; }
.result-preview { background: white; border: 1px solid #d9e2ef; border-radius: 12px;
padding: 18px; white-space: pre-wrap; line-height: 1.6; font-family: ui-monospace, monospace;
font-size: 13px; max-height: 360px; overflow: auto; }
mark { background: #fff0ae; padding: 0 2px; border-radius: 3px; }
</style>''', unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def resources(corpus_signature: str):
    return build_workbench(ROOT)


def highlighted(text: str, query: str) -> str:
    safe = html.escape(text)
    terms = sorted(set(lexical_tokens(query)), key=len, reverse=True)
    if terms:
        pattern = r'(?<!\w)(' + '|'.join(re.escape(term) for term in terms) + r')(?!\w)'
        safe = re.sub(pattern, r'<mark>\1</mark>', safe, flags=re.IGNORECASE)
    return '<div class="result-preview">' + safe + '</div>'


def show_results(config, query: str, k: int):
    started = time.perf_counter()
    results = config.search(query, k)
    elapsed = (time.perf_counter() - started) * 1000
    st.subheader(config.method + ' search')
    st.caption(f'{len(results)} results · {elapsed:.2f} ms · {config.strategy}')
    if not results:
        st.info('No positive-scoring keyword matches. Zero-score chunks are not treated as hits.')
    for result in results:
        chunk = result.chunk
        title = next((h[-1] for h in chunk.metadata['headers'] if h), chunk.parent_doc)
        with st.expander(f'#{result.rank} · {title}', expanded=result.rank == 1):
            label = 'BM25 score' if config.method == 'BM25' else 'Cosine similarity'
            st.write(f'**{label}:** {result.score:.4f}')
            st.caption(f'{chunk.parent_doc} · lines {chunk.metadata["line_start"]}–{chunk.metadata["line_end"]}')
            st.markdown(highlighted(chunk.content[:3500], query), unsafe_allow_html=True)
            if len(chunk.content) > 3500:
                st.caption('Preview shortened; full source chunk available below.')
            st.download_button('Download full chunk', chunk.content, 'source-chunk.md', 'text/markdown',
                               key=f'{config.name}-{chunk.chunk_id}')
            st.json(chunk.metadata, expanded=False)


st.caption('TECHNICAL DOCUMENTATION / RETRIEVAL LAB')
st.title('SearchWorkbench')
st.write('Compare exact keyword matching with semantic meaning—and measure what changes when you change the chunks.')
manifest = json.loads((ROOT / 'data/corpus_manifest.json').read_text())
queries = load_queries(ROOT / 'data/eval_queries.json')
col1, col2, col3 = st.columns(3)
col1.metric('Source documentation pages', manifest['page_count'])
col2.metric('Labeled evaluation queries', len(queries))
col3.metric('Search configurations', 8)
st.caption('FastAPI documentation · BM25 + FAISS exact cosine · MiniLM, 384 dimensions · CPU, no API key')

with st.sidebar:
    st.header('Workbench setup')
    st.write('The first build downloads a free local model and embeds the documentation. Later builds reuse cached vectors.')
    if st.button('Build / load indexes', type='primary'):
        st.session_state['ready'] = True
    strategy = st.selectbox('Chunking strategy', STRATEGIES, index=3)
    k = st.select_slider('Results / K', options=[3, 5, 10], value=5)
    st.caption('MiniLM has a 256-token input limit. Long chunks use weighted pooling of smaller windows, not silent truncation.')
    st.markdown('[Source documentation](https://fastapi.tiangolo.com/)')

if not st.session_state.get('ready'):
    st.info('Select “Build / load indexes” in the sidebar to begin. No API key or paid account is needed.')
    st.stop()

signature = hashlib.sha256(('\n'.join(p.relative_to(ROOT/'corpus').as_posix() + '\n' +
    p.read_text() for p in sorted((ROOT/'corpus').rglob('*.md')))).encode()).hexdigest()
try:
    with st.spinner('Loading the model and building eight search configurations…'):
        documents, configurations, model = resources(signature)
except Exception as exc:
    st.error(f'Could not build indexes: {exc}')
    st.info('The first model download needs internet access. Check the terminal output and README troubleshooting, then try again.')
    st.stop()

lookup = {config.name: config for config in configurations}
st.success('Indexes ready. Search data contains source documents only; evaluation queries are kept separate.')
tab_search, tab_benchmark, tab_corpus = st.tabs(['Live search comparator', 'Benchmark & failure analysis', 'Corpus & method'])

with tab_search:
    st.subheader('One question. Two retrieval methods.')
    st.caption('Try an identifier such as response_model_exclude_unset, then ask a conceptual question. Neither method is guaranteed to win.')
    with st.form('search-form'):
        query = st.text_input('Technical question', value='How do I keep large uploaded attachments from exhausting RAM?')
        submitted = st.form_submit_button('Compare search', type='primary')
    if submitted:
        st.session_state['last_query'] = query
    if st.session_state.get('last_query', '').strip():
        active = st.session_state['last_query']
        st.caption('Highlighting shows literal query terms only. BM25 scores and cosine similarities have different scales.')
        left, right = st.columns(2)
        with left:
            show_results(lookup[f'BM25 / {strategy}'], active, k)
        with right:
            show_results(lookup[f'Dense / {strategy}'], active, k)

with tab_benchmark:
    st.subheader('Measure retrieval quality')
    st.write('Run the same 24 labeled queries across BM25 and dense search for 256-, 512-, 1024-token, and heading-based chunks.')
    if st.button('Run Benchmark', type='primary'):
        progress = st.progress(0, text='Warming up…')
        try:
            report = benchmark(documents, configurations, queries,
                               progress=lambda value, name: progress.progress(value, text=name))
            report['run'] = {'utc': datetime.now(timezone.utc).isoformat(), 'model': model.model_name,
                             'model_sha256': model.model_sha256}
            st.session_state['report'] = report
        except Exception as exc:
            st.error(f'Benchmark failed: {exc}')
        finally:
            progress.empty()
    report = st.session_state.get('report')
    if report:
        group = st.radio('Query group', ['all', 'exact', 'conceptual'], horizontal=True)
        frame = pd.DataFrame(report['summary'])
        selected = frame[(frame['k'] == k) & (frame['query_group'] == group)]
        display = selected[['configuration','chunks','hit_rate','recall','mrr',
                            'latency_mean_ms','latency_p50_ms','latency_p95_ms']]
        st.dataframe(display, hide_index=True, width='stretch')
        st.bar_chart(selected.set_index('configuration')[['hit_rate','recall','mrr']])
        st.bar_chart(selected.set_index('configuration')[['latency_mean_ms']])
        st.caption('Quality metrics are 0–1. Latency is warm, serial execution and includes query embedding for dense search. '
                   'One maximum-K search is sliced for K=3/5/10; these are not separately timed K-specific runs.')
        st.download_button('Download measured JSON', json.dumps(report, indent=2), 'benchmark.json', 'application/json')
        st.download_button('Download summary CSV', frame.to_csv(index=False), 'benchmark.csv', 'text/csv')
        st.subheader(f'Failure analysis · {strategy} · K={k}')
        rows = [r for r in report['per_query'] if r['strategy'] == strategy and r['k'] == k]
        pairs = {}
        for row in rows:
            pairs.setdefault(row['query_id'], {})[row['method']] = row
        wins = {'BM25': [], 'Dense': []}
        for pair in pairs.values():
            sparse, dense = pair['BM25'], pair['Dense']
            if sparse['mrr'] != dense['mrr']:
                winner = 'BM25' if sparse['mrr'] > dense['mrr'] else 'Dense'
                wins[winner].append((abs(sparse['mrr'] - dense['mrr']), sparse, dense))
        for method, column in zip(('BM25','Dense'), st.columns(2)):
            with column:
                st.markdown(f'**Where {method} ranked the first relevant result better**')
                if not wins[method]:
                    st.info('No strict MRR wins for this selection. Try another chunk strategy or K.')
                for _, sparse, dense in sorted(wins[method], key=lambda x: -x[0])[:5]:
                    with st.expander(sparse['query'], expanded=False):
                        st.write(f"Query type: {sparse['kind']}. BM25 first relevant rank: {sparse['first_relevant_rank']}; "
                                 f"Dense: {dense['first_relevant_rank']}.")
                        st.caption('Possible explanation: BM25 rewards shared terms and identifiers; dense search can match related meaning '
                                   'but may dilute exact details. This comparison is an observation, not a proven causal diagnosis.')
        st.info('This is a small, curated development set, not a held-out estimate of production search quality. Review the source passages and add new labels before generalizing.')
    else:
        st.info('Run the benchmark to generate live metrics, charts, downloads, and failure examples.')

with tab_corpus:
    st.subheader('Corpus provenance and fair comparison')
    st.write(f"{manifest['page_count']} original Markdown pages from FastAPI's MIT-licensed documentation.")
    st.code(manifest['revision'], language=None)
    st.dataframe(pd.DataFrame(manifest['pages'])[['path','bytes','lines']], hide_index=True, width='stretch')
    st.write('Each label identifies a source document and an exact evidence passage. Chunks are relevant when they fully contain that passage. '
             'Recall counts unique evidence passages, so overlapping chunks cannot inflate it. Two queries require two passages.')
    st.write('FAISS IndexFlatIP uses exact inner-product search on unit-length vectors. It is an embedded in-memory index, not an ANN scaling benchmark. '
             'Ties use chunk IDs for a stable order. The same BM25 tokenizer is used for documents and queries.')
    st.caption('FastAPI include directives remain source references rather than rendered code examples. Fixed windows can split code fences; '
               'heading-based chunks preserve complete fenced blocks. Large-section vectors are pooled across model-sized windows.')

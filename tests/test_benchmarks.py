import hashlib
import json
from pathlib import Path
import pytest
from src.benchmarker import load_queries, metrics_at_k, resolve_evidence, result_evidence
from src.chunking import chunk_corpus, load_corpus
from src.models import SearchResult

ROOT=Path(__file__).resolve().parents[1]


def test_known_ranking_math():
    values=metrics_at_k([set(),{0},set()],{0},3)
    assert values=={'hit_rate':1.0,'recall':1.0,'mrr':0.5}


def test_missed_query_is_zero_and_top_k_is_enforced():
    assert metrics_at_k([set(),set(),{0}],{0},2)=={'hit_rate':0.0,'recall':0.0,'mrr':0.0}
    assert metrics_at_k([],{0},5)['mrr']==0


def test_duplicate_overlapping_hits_do_not_inflate_recall():
    values=metrics_at_k([{0},{0},{0}],{0,1},3)
    assert values['recall']==0.5
    assert values['mrr']==1


def test_multiple_evidence_units_are_counted_once():
    assert metrics_at_k([{0},{0,1}],{0,1},2)['recall']==1


@pytest.mark.parametrize('relevant,k',[(set(),3),({0},0)])
def test_invalid_metric_inputs(relevant,k):
    with pytest.raises(ValueError):metrics_at_k([] ,relevant,k)


def test_all_labels_are_valid_and_covered_under_every_chunk_strategy():
    documents=load_corpus(ROOT/'corpus')
    queries=load_queries(ROOT/'data/eval_queries.json')
    assert len(documents)>=25 and len(queries)>=15
    for strategy in ['fixed-256','fixed-512','fixed-1024','structure']:
        chunks=chunk_corpus(documents,strategy)
        for query in queries:
            spans=resolve_evidence(query,documents)
            found=set().union(*(result_evidence(SearchResult(chunk=c,score=0,rank=1,method='test'),spans) for c in chunks))
            assert found==set(range(len(spans))),(strategy,query.query_id)


def test_source_snapshot_matches_recorded_checksums():
    manifest=json.loads((ROOT/'data/corpus_manifest.json').read_text())
    for page in manifest['pages']:
        assert hashlib.sha256((ROOT/'corpus'/page['path']).read_bytes()).hexdigest()==page['sha256']

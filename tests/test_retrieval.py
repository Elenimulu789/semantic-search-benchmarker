import numpy as np
import pytest
from src.chunking import StructureAwareChunker, load_corpus
from src.indexer import DenseIndex, SparseIndex, lexical_tokens, unit_rows


def fixtures(texts):
    return [StructureAwareChunker().split(text,f'{i}.md')[0] for i,text in enumerate(texts)]


def test_exact_rare_identifier_is_rank_one_case_insensitive():
    chunks=fixtures(['Enable CUSTOM_CONFIG_FLAG_XYZ to record diagnostics.',
                     'Configure the database connection pool.', 'Update a route handler.'])
    results=SparseIndex(chunks).search('custom_config_flag_xyz!',3)
    assert results[0].chunk.parent_doc=='0.md'
    assert lexical_tokens('CUSTOM_CONFIG_FLAG_XYZ!')==lexical_tokens('custom_config_flag_xyz')


def test_zero_keyword_overlap_returns_no_bm25_hits():
    index=SparseIndex(fixtures(['Database transactions are atomic.','Routes return JSON.']))
    assert index.search('zzzz_nonexistent_marker')==[]


class NumericFixtureEmbedder:
    """For deterministic score mathematics ONLY; not a semantic model substitute."""
    signature='unit-test-numeric-v1'
    def encode(self,texts):
        values={'horizontal': [100,0], 'diagonal':[1,1], 'vertical-query':[0,5], 'tie':[100,0]}
        return np.array([values[t] for t in texts],dtype=np.float32)


def test_cosine_normalization_and_descending_ranking(tmp_path):
    chunks=fixtures(['horizontal','diagonal'])
    dense=DenseIndex(chunks,NumericFixtureEmbedder(),tmp_path)
    results=dense.search('vertical-query',2)
    assert results[0].chunk.content=='diagonal'
    assert results[0].score==pytest.approx(2**-0.5,abs=1e-6)
    assert results[1].score==pytest.approx(0.0,abs=1e-6)
    assert DenseIndex(chunks,NumericFixtureEmbedder(),tmp_path).cache_hit


def test_dense_ties_are_stably_ordered_by_chunk_id():
    chunks=fixtures(['horizontal','tie'])
    index=DenseIndex(chunks,NumericFixtureEmbedder())
    assert [r.chunk.chunk_id for r in index.search('vertical-query',2)]==sorted(c.chunk_id for c in chunks)


@pytest.mark.parametrize('values', [[[0,0]], [[float('nan'),1]], [1,2]])
def test_invalid_vectors_are_rejected(values):
    with pytest.raises(ValueError):unit_rows(np.array(values))


def test_ingestion_does_not_read_evaluation_data(tmp_path):
    corpus=tmp_path/'corpus';corpus.mkdir()
    (corpus/'source.md').write_text('# Source\nOrdinary technical content.')
    data=tmp_path/'data';data.mkdir()
    (data/'eval_queries.json').write_text('{"secret_answer":"DO_NOT_INDEX"}')
    assert 'DO_NOT_INDEX' not in ''.join(load_corpus(corpus).values())


@pytest.mark.integration
def test_real_dense_paraphrase_with_zero_exact_keyword_overlap(real_model):
    source='Requests must present a valid password before gaining access.'
    query='Authentication'
    assert set(lexical_tokens(source)).isdisjoint(lexical_tokens(query))
    index=DenseIndex(fixtures([source,'A database index accelerates row lookups.',
                               'Uploaded photographs are resized into thumbnails.']),real_model)
    assert index.search(query,1)[0].chunk.parent_doc=='0.md'


@pytest.mark.integration
def test_model_windows_include_the_entire_long_input(real_model):
    text=('Database transaction consistency and recovery. '*150)+'TAIL_IDENTIFIER_KEEP_ME'
    windows=real_model._windows(text)
    assert len(windows)>1
    assert 'TAIL_IDENTIFIER_KEEP_ME' in windows[-1][0]
    assert sum(weight for _,weight in windows)==len(real_model.tokenizer.encode(text,add_special_tokens=False).ids)
    values=real_model.encode([text,'Short query.'])
    assert values.shape==(2,384)
    np.testing.assert_allclose(np.linalg.norm(values,axis=1),[1,1],atol=1e-5)

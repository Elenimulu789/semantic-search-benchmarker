import pytest
from src.chunking import ENCODING, FixedSizeChunker, StructureAwareChunker, sections


@pytest.mark.parametrize('size,overlap', [(0,0),(5,5),(5,-1)])
def test_invalid_token_windows(size,overlap):
    with pytest.raises(ValueError):FixedSizeChunker(size,overlap)


def test_exact_overlap_and_no_original_token_drops():
    text=' '.join(f'configuration_{i}' for i in range(150))
    chunks=FixedSizeChunker(64,16).split(text,'example.md')
    original=ENCODING.encode(text)
    rebuilt=list(chunks[0].metadata['token_ids'])
    for before,after in zip(chunks,chunks[1:]):
        assert before.metadata['token_ids'][-16:]==after.metadata['token_ids'][:16]
        assert after.metadata['token_start']==before.metadata['token_end']-16
        rebuilt.extend(after.metadata['token_ids'][16:])
    assert rebuilt==original
    assert ENCODING.decode(rebuilt)==text


def test_unicode_boundaries_do_not_corrupt_content():
    text=('ሰላም 🌍 café 数据库 configuration ')*50
    chunks=FixedSizeChunker(80,16).split(text,'unicode.md')
    for chunk in chunks:
        assert '\ufffd' not in chunk.content
        assert text[chunk.metadata['char_start']:chunk.metadata['char_end']]==chunk.content
    ids=list(chunks[0].metadata['token_ids'])
    for before,after in zip(chunks,chunks[1:]):
        assert before.metadata['token_ids'][-16:]==after.metadata['token_ids'][:16]
        ids.extend(after.metadata['token_ids'][16:])
    assert ENCODING.decode(ids)==text


def test_structure_preserves_fences_and_ignores_code_headings():
    text='# Parent\nIntroduction.\n## Child\n```python\n# Not a section\nx = 3\n```\nMore.\n## Other\nDone.\n'
    chunks=StructureAwareChunker().split(text,'code.md')
    assert len(chunks)==3
    assert '```python\n# Not a section\nx = 3\n```' in chunks[1].content
    assert chunks[1].metadata['headers']==[['Parent','Child']]
    assert ''.join(c.content for c in chunks)==text


def test_fixed_windows_expose_code_slicing_as_a_baseline_limitation():
    text='# Example\n```python\n'+('print("hello")\n'*100)+'```\n'
    chunks=FixedSizeChunker(64,8).split(text,'code.md')
    assert len(chunks)>1
    assert any(c.content.count('```')%2==1 for c in chunks)
    assert len(StructureAwareChunker().split(text,'code.md'))==1


def test_empty_and_repeated_headers():
    assert FixedSizeChunker().split('','empty.md')==[]
    assert StructureAwareChunker().split('','empty.md')==[]
    values=sections('# A\nfirst\n# A\nsecond','duplicate.md')
    assert len({s.section_id for s in values})==2

from pathlib import Path
import pytest
from src.indexer import LocalEmbedder

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='session')
def real_model():
    return LocalEmbedder(ROOT / '.cache/models')

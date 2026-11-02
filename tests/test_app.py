from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]


def test_dashboard_loads_without_downloading_a_model():
    app=AppTest.from_file(str(ROOT/'src/app.py')).run(timeout=20)
    assert not app.exception
    assert app.title[0].value=='SearchWorkbench'


@pytest.mark.integration
def test_dashboard_build_and_real_search():
    app=AppTest.from_file(str(ROOT/'src/app.py')).run(timeout=20)
    app.sidebar.button[0].click().run(timeout=240)
    assert not app.exception
    assert any('Indexes ready' in message.value for message in app.success)
    app.button(key='FormSubmitter:search-form-Compare search').click().run(timeout=60)
    assert not app.exception
    assert any('BM25 search' in title.value for title in app.subheader)
    assert any('Dense search' in title.value for title in app.subheader)
    next(button for button in app.button if button.label=='Run Benchmark').click().run(timeout=180)
    assert not app.exception
    assert app.session_state['report']['summary']
    assert any('Failure analysis' in title.value for title in app.subheader)

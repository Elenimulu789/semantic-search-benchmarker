"""Refresh the bundled MIT-licensed FastAPI documentation snapshot.
Run deliberately; normal search and evaluation never fetch or change the corpus.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PAGES = [
    'tutorial/first-steps.md', 'tutorial/path-params.md', 'tutorial/query-params.md',
    'tutorial/body.md', 'tutorial/query-params-str-validations.md',
    'tutorial/path-params-numeric-validations.md', 'tutorial/body-multiple-params.md',
    'tutorial/body-fields.md', 'tutorial/body-nested-models.md',
    'tutorial/schema-extra-example.md', 'tutorial/extra-data-types.md',
    'tutorial/cookie-params.md', 'tutorial/header-params.md',
    'tutorial/response-model.md', 'tutorial/extra-models.md',
    'tutorial/response-status-code.md', 'tutorial/request-forms.md',
    'tutorial/request-files.md', 'tutorial/handling-errors.md',
    'tutorial/dependencies/index.md', 'tutorial/dependencies/dependencies-with-yield.md',
    'tutorial/security/first-steps.md', 'tutorial/background-tasks.md',
    'tutorial/cors.md', 'tutorial/middleware.md', 'advanced/websockets.md',
    'advanced/events.md',
]

def download(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'SearchWorkbench-classroom-corpus'})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main():
    metadata=json.loads(download('https://api.github.com/repos/fastapi/fastapi/commits/master'))
    revision=metadata['sha']
    base=f'https://raw.githubusercontent.com/fastapi/fastapi/{revision}/'
    def fetch(path):
        url=base+'docs/en/docs/'+path
        raw=download(url)
        target=ROOT/'corpus'/path
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(raw)
        return {'path':path,'source_url':url,'documentation_url':'https://fastapi.tiangolo.com/'+path.removesuffix('.md').replace('/index','')+'/',
                'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'lines':len(raw.splitlines())}
    with ThreadPoolExecutor(max_workers=5) as executor:
        pages=list(executor.map(fetch,PAGES))
    (ROOT/'corpus/LICENSE-FastAPI.txt').write_bytes(download(base+'LICENSE'))
    manifest={'repository':'https://github.com/fastapi/fastapi','revision':revision,
              'fetched_utc':datetime.now(timezone.utc).isoformat(), 'license':'MIT',
              'page_count':len(pages),'pages':pages,
              'notes':'Original Markdown snapshot. FastAPI-specific include directives are retained as source text, not resolved to external snippets.'}
    (ROOT/'data/corpus_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Downloaded {len(pages)} source pages; {sum(p["bytes"] for p in pages):,} bytes; revision {revision}')

if __name__=='__main__':main()

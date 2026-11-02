"""Run the same benchmark as the dashboard and save measured evidence."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.benchmarker import benchmark, load_queries
from src.search import build_workbench


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', default='artifacts/benchmark.json')
    args = parser.parse_args()
    print('Loading local model and building eight configurations...', flush=True)
    documents, configs, model = build_workbench(ROOT)
    queries = load_queries(ROOT/'data/eval_queries.json')
    print(f'{len(documents)} documents; {len(queries)} queries; model ready', flush=True)
    report = benchmark(documents, configs, queries, repeats=args.repeats,
                       progress=lambda fraction,name: print(f'{fraction:.0%} {name}', flush=True))
    report['run'] = {'utc': datetime.now(timezone.utc).isoformat(), 'python': platform.python_version(),
                     'platform': platform.platform(), 'model': model.model_name,
                     'model_sha256': model.model_sha256, 'model_signature': model.signature}
    output = ROOT/args.output
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    import pandas as pd
    pd.DataFrame(report['summary']).to_csv(output.with_suffix('.csv'),index=False)
    overall=[r for r in report['summary'] if r['query_group']=='all' and r['k']==5]
    print(pd.DataFrame(overall)[['configuration','hit_rate','recall','mrr','latency_mean_ms','chunks']].to_string(index=False))
    print(f'Saved {output}', flush=True)


if __name__=='__main__':main()

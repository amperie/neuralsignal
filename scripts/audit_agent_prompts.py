"""Audit complete model prompts and export label-blind examples before extraction."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from transformers import AutoTokenizer

from neuralsignal.core.modules.prompting import wrap_with_prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/remote/malt_full_normal_bs1.yaml'))
    parser.add_argument('--output', type=Path, default=Path('runs/datasets/prompt-audit'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.config.read_text())
    config = config.get('feature_config', config)
    template = config['prompt']['template']
    tokenizer = AutoTokenizer.from_pretrained(config['model']['model_name'])
    summary = {}
    for path in args.inputs:
        rows, examples = [], set()
        for index, line in enumerate(path.open(encoding='utf-8')):
            row = json.loads(line)
            # This is the exact legacy rendering, including its literal-placeholder behavior.
            prompt = wrap_with_prompt(template, row)
            tokens = tokenizer(prompt, truncation=False)['input_ids']
            rows.append({'id': row['id'], 'tokens': len(tokens), 'characters': len(prompt),
                         'sha256': hashlib.sha256(prompt.encode()).hexdigest()})
            if row['abnormal'] not in examples:
                examples.add(row['abnormal'])
                name = f'{path.stem}-{"hack" if row["abnormal"] else "benign"}'
                (args.output / f'{name}.txt').write_text(prompt, encoding='utf-8')
                (args.output / f'{name}.metadata.json').write_text(json.dumps({k:v for k,v in row.items() if k not in {'input','output'}}, indent=2), encoding='utf-8')
            if (index + 1) % 1000 == 0:
                print(f'{path.stem}: {index+1}', flush=True)
        pd.DataFrame(rows).to_csv(args.output / f'{path.stem}-lengths.csv', index=False)
        summary[path.stem] = {'rows':len(rows), 'token_percentiles':dict(zip(['min','median','p90','p99','max'], np.percentile([r['tokens'] for r in rows], [0,50,90,99,100]).tolist())), 'unique_full_prompts':len({r['sha256'] for r in rows}), 'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        print(json.dumps({path.stem:summary[path.stem]}), flush=True)
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

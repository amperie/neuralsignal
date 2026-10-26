"""Cache identical encoded inputs; preserve all trajectories in final feature runs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from transformers import AutoTokenizer

from neuralsignal.datasets.sources.jsonl import JsonlSource
from neuralsignal.core.modules.prompting import wrap_with_prompt
from neuralsignal.features.runner import collect_features_batched
from neuralsignal.storage.local import LocalFeatureShardWriter
from neuralsignal.storage.manifests import RunManifest

CONFIG = Path('configs/remote/malt_full_normal_bs1.yaml')


def prepare(paths, output):
    if output.exists():
        raise FileExistsError(output)
    config = yaml.safe_load(CONFIG.read_text())['feature_config']
    if config['generation']['batch_size'] != 1 or config['generation']['truncation_length'] != 512:
        raise ValueError('Exact-input cache requires the verified MALT batch-1, 512-token configuration')
    tokenizer = AutoTokenizer.from_pretrained(config['model']['model_name'])
    seen, selected, mapping, audit = set(), [], {}, {}
    for path in paths:
        entries, counts, visible, truncated = [], {}, 0, 0
        for i, line in enumerate(path.open(encoding='utf-8')):
            row = json.loads(line)
            prompt = wrap_with_prompt(config['prompt']['template'], {'input': row['input'], 'output': row['output']})
            encoded = tokenizer(prompt, padding=False, truncation=True, max_length=512)
            digest = hashlib.sha256(json.dumps(dict(encoded), sort_keys=True).encode()).hexdigest()
            counts[digest] = counts.get(digest, 0) + 1
            prefix = prompt.split('assistant:', 1)[0]
            visible += len(tokenizer(prefix, truncation=True, max_length=513)['input_ids']) < 512
            truncated += len(encoded['input_ids']) == 512
            entries.append({'id': row['id'], 'digest': digest})
            # Verify equivalent features across variants, source classes and models.
            key = (digest, path.stem, row['classification'], row['model'])
            if key not in seen:
                seen.add(key)
                selected.append({**row, 'id': f'{path.stem}:{row["id"]}', 'input_digest': digest})
            if (i+1) % 1000 == 0:
                print(f'{path.stem}: encoded {i+1}', flush=True)
        mapping[str(path)] = entries
        audit[path.stem] = {'rows': len(entries), 'unique_encoded_inputs': len(counts),
                            'input_frequencies': counts, 'assistant_begins_before_limit': visible,
                            'rows_at_token_limit': truncated}
    output.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in selected), encoding='utf-8')
    output.with_suffix('.mapping.json').write_text(json.dumps(mapping), encoding='utf-8')
    output.with_suffix('.audit.json').write_text(json.dumps({'datasets': audit, 'rows': len(selected),
        'config_sha256': hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}, indent=2), encoding='utf-8')
    print(json.dumps(audit), 'Extraction representatives:', len(selected), flush=True)


def expand(compact, features):
    config = yaml.safe_load(CONFIG.read_text())['feature_config']
    audit = json.loads(compact.with_suffix('.audit.json').read_text())
    if audit['config_sha256'] != hashlib.sha256(CONFIG.read_bytes()).hexdigest():
        raise ValueError('Extraction configuration changed since token grouping')
    manifest = json.loads((features/'manifest.json').read_text())
    if manifest['state'] != 'completed':
        raise ValueError('Representative extraction incomplete')
    for shard in manifest['shards']:
        if hashlib.sha256((features/shard['path']).read_bytes()).hexdigest() != shard['sha256']:
            raise ValueError('Representative shard checksum mismatch')
    frame = pd.concat([pd.read_parquet(features/s['path']) for s in manifest['shards']])
    if len(frame) != audit['rows']:
        raise ValueError('Missing representative rows')
    columns = [c for c in frame if c.startswith(('zones__', 'layer_distribution__'))]
    if not columns:
        raise ValueError('Expected activation features')
    frame['digest'] = frame.metadata_json.map(lambda value: json.loads(value)['input_digest'])
    cache = {}
    for digest, rows in frame.groupby('digest'):
        values = rows[columns].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.all(values == values[0]):
            raise ValueError('Identical encoded inputs did not produce bit-identical activation features')
        cache[digest] = rows.iloc[0][columns].astype(float).to_dict()
    for path_string, mapping in json.loads(compact.with_suffix('.mapping.json').read_text()).items():
        path = Path(path_string)
        source_audit = json.loads(path.with_suffix('.audit.json').read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != source_audit['sha256']:
            raise ValueError('Normalized source changed after token grouping')
        root = Path('runs/remote')/path.stem
        if root.exists():
            raise FileExistsError(root)
        examples = JsonlSource(path).iter_examples()
        def extract(batch, specs):
            result = []
            for example in batch:
                entry = next(entries)
                if entry['id'] != example.id:
                    raise ValueError('Trajectory ordering changed')
                result.append(cache[entry['digest']])
            return result
        entries = iter(mapping)
        run = RunManifest(path.stem, {'source': 'jsonl', 'path': str(path)}, manifest['features'], state='running')
        writer = LocalFeatureShardWriter(root, run)
        collect_features_batched(examples, config, writer, extract)
        if run.rows['written'] != len(mapping):
            raise ValueError('Expanded row count mismatch')
        run.rows['expected'] = len(mapping)
        run.state = 'completed'
        run.write_json(root/'manifest.json')
        source_audit = json.loads(path.with_suffix('.audit.json').read_text())
        source_audit.update(token_audit=audit['datasets'][path.stem],
                            extraction_reuse='bit-identical activation features for exact token IDs and attention masks',
                            representative_run=str(features), verified_representatives=len(frame))
        (root/'source_audit.json').write_text(json.dumps(source_audit, indent=2))
        print(f'Materialized {run.rows["written"]} trajectories: {root}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'expand'])
    parser.add_argument('--inputs', nargs='+', type=Path)
    parser.add_argument('--compact', type=Path, required=True)
    parser.add_argument('--features', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.inputs, args.compact)
    else:
        expand(args.compact, args.features)

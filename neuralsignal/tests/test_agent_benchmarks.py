import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neuralsignal.datasets.sources.trace import normalize_trace_record
from neuralsignal.datasets.sources.terminal_wrench import TerminalWrenchSource


def test_trace_preserves_tools_and_all_turns_without_labels_in_text():
    conversation = [{'role': 'user', 'content': 'Fix the code'},
                    {'role': 'assistant', 'content': 'Checking', 'tool_calls': [{'name': 'Read'}], 'tool_results': [{'output': 'source'}]},
                    {'role': 'user', 'content': 'More context'},
                    {'role': 'assistant', 'content': 'Done'}]
    example, = normalize_trace_record({'trajectory_id': 'a', 'conversation': json.dumps(conversation), 'label': '1.1.2, 1.2.1'})
    assert example.metadata['abnormal'] == 1
    assert example.metadata['categories'] == ['1.1.2', '1.2.1']
    assert 'tool_calls' in example.input and 'tool_results' in example.input
    assert example.input.index('Checking') < example.input.index('More context')
    assert example.output == 'assistant: Done'
    assert '1.1.2' not in example.input + example.output


@pytest.mark.parametrize('label', ['', None, 'normal', '0, 1.1.1'])
def test_trace_rejects_unknown_labels(label):
    with pytest.raises(ValueError):
        list(normalize_trace_record({'trajectory_id': 'a', 'conversation': '[]', 'label': label}))


def test_trace_new_group_field_requires_review():
    with pytest.raises(ValueError, match='schema changed'):
        list(normalize_trace_record({'trajectory_id': 'a', 'conversation': '[]', 'label': '0', 'task_id': 'task'}))


def test_wrench_removes_elicitation_and_never_renders_judgment(tmp_path, monkeypatch):
    source = TerminalWrenchSource(tmp_path, 'revision', 'stripped')
    trajectory = {'schema_version': 'ATIF-v1.6', 'steps': [
        {'source': 'user', 'message': 'Task. ELICITATION'},
        {'source': 'agent', 'message': '', 'tool_calls': [{'function_name': 'bash_command', 'arguments': {'keystrokes': 'ls'}}],
         'observation': {'results': [{'content': 'files'}]}}]}
    paths = []
    def read(path):
        paths.append(path)
        return json.dumps(trajectory)
    monkeypatch.setattr(source, 'read', read)
    example = source.normalize({'task_id': 'task', 'model': 'model', 'tree_name': 'hack_trajectories',
                                'trajectory_label': 'v5', 'classification': 'rewarded_serious_exploit',
                                'explanation': 'GROUND TRUTH'}, 'ELICITATION')
    assert '/stripped_trajectories/' in paths[0]
    assert example.metadata['task_id'] == 'task'
    assert example.metadata['abnormal'] == 1
    assert 'ELICITATION' not in example.input
    assert 'GROUND TRUTH' not in example.input + example.output
    assert 'bash_command' in example.output and 'tool: files' in example.output


def evaluator():
    path = Path(__file__).resolve().parents[2] / 'notebooks/evaluate_agent_benchmark.py'
    spec = importlib.util.spec_from_file_location('benchmark_eval', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inner_folds_keep_every_task_together():
    module = evaluator()
    groups = pd.Series(np.repeat(np.arange(12), 4))
    y = pd.Series(np.tile([0, 1, 0, 1], 12))
    x = pd.DataFrame({'feature': np.arange(48)})
    folds = module.inner_splits(x, y, groups, 42)
    for train, valid in folds:
        assert not set(groups.iloc[train]) & set(groups.iloc[valid])
    assert sorted(np.concatenate([v for _, v in folds])) == list(range(48))


def test_notebook_reuse_and_training_threshold():
    module = evaluator()
    helpers, _ = module.notebook_helpers()
    assert helpers['XGB_BASE_PARAMS']['n_jobs'] == 4
    scores = np.array([.1, .3, .6, .9])
    threshold = module.select_threshold(np.array([0, 0, 1, 1]), scores, .95)
    assert np.mean(scores[2:] >= threshold) == 1
    result = module.binary_metrics(helpers, np.array([0, 0, 1, 1]), scores)
    assert result['macro_f1'] == 1 and result['abnormal_catch_rate'] == 1

@pytest.mark.parametrize('second_value', [1.25, 1.5])
def test_exact_input_cache_validates_features_and_preserves_rows(tmp_path, monkeypatch, second_value):
    from scripts import cache_agent_inputs as cache
    from neuralsignal.storage.local import LocalFeatureShardWriter
    from neuralsignal.storage.manifests import RunManifest
    import hashlib
    cache_config = cache.CONFIG.resolve()
    monkeypatch.setattr(cache, 'CONFIG', cache_config)
    monkeypatch.chdir(tmp_path)
    source = tmp_path/'terminal-test.jsonl'
    source.write_text('\n'.join(json.dumps({'id': str(i), 'input': 'task', 'output': 'agent',
                      'abnormal': i, 'task_id': str(i)}) for i in range(2)))
    source.with_suffix('.audit.json').write_text(json.dumps({'rows': 2, 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}))
    compact = tmp_path/'compact.jsonl'
    compact.with_suffix('.audit.json').write_text(json.dumps({'rows': 2,
        'config_sha256': hashlib.sha256(cache_config.read_bytes()).hexdigest(),
        'datasets': {'terminal-test': {'rows': 2}}}))
    compact.with_suffix('.mapping.json').write_text(json.dumps({str(source): [
        {'id': str(i), 'digest': 'same'} for i in range(2)]}))
    features = tmp_path/'representatives'
    manifest = RunManifest('representatives', {'source': 'test'}, {'test': True}, state='completed')
    writer = LocalFeatureShardWriter(features, manifest)
    writer.write_shard([{'metadata_json': json.dumps({'input_digest': 'same'}), 'zones__test': v}
                        for v in [1.25, second_value]])
    if second_value != 1.25:
        with pytest.raises(ValueError, match='bit-identical'):
            cache.expand(compact, features)
    else:
        cache.expand(compact, features)
        result = pd.read_parquet(tmp_path/'runs/remote/terminal-test/features/part-00000.parquet')
        assert result.example_id.tolist() == ['0', '1']
        assert result.zones__test.tolist() == [1.25, 1.25]
        assert [json.loads(v)['abnormal'] for v in result.metadata_json] == [0, 1]


def test_trace_duplicates_stay_together_in_outer_and_inner_folds():
    module = evaluator()
    y = pd.Series(np.tile([0, 1], 30))
    hashes = pd.Series([f'row-{i}' for i in range(60)])
    hashes.iloc[59] = hashes.iloc[1]
    x = pd.DataFrame({'feature': np.arange(60)})
    groups, folds = module.outer_splits(x, y, pd.DataFrame({'transcript_sha256': hashes}), 'trace', None)
    for train, test in folds:
        assert not set(groups.iloc[train]) & set(groups.iloc[test])
        for a, b in module.inner_splits(x.iloc[train], y.iloc[train], groups.iloc[train], 42):
            assert not set(groups.iloc[train].iloc[a]) & set(groups.iloc[train].iloc[b])
    assert sorted(np.concatenate([test for _, test in folds])) == list(range(60))


def test_trace_rejects_conflicting_duplicate_labels():
    module = evaluator()
    with pytest.raises(ValueError, match='conflicting labels'):
        module.outer_splits(pd.DataFrame({'feature': [1, 2]}), pd.Series([0, 1]),
                            pd.DataFrame({'transcript_sha256': ['same', 'same']}), 'trace', None)


def test_full_context_config_disables_truncation():
    import yaml
    from unittest.mock import Mock, patch
    from neuralsignal.features.model_extractor import ModelFeatureExtractor
    from neuralsignal.datasets.v2 import DatasetExample
    config = yaml.safe_load(Path('configs/remote/agent_full_context_bs1.yaml').read_text())
    extractor = ModelFeatureExtractor(config)
    extractor.model, extractor.tokenizer = object(), object()
    extractor._featurize = Mock(return_value={'zones__test': 1.0})
    text = 'actual behavior ' * 2000 + 'FINAL_ACTION'
    with patch('neuralsignal.features.model_extractor.generate_from_batch', return_value=[Mock(data={})]) as generate:
        extractor.extract_batch([DatasetExample(id='long', input=text, output='DONE', labels=[], metadata={})], [])
    assert generate.call_args.kwargs['truncation_length'] == 0
    assert text in generate.call_args.args[0][0]
    assert 'DONE' in generate.call_args.args[0][0]


def test_failed_remote_manifest_aborts_bundle_wait(monkeypatch):
    import io
    from unittest.mock import Mock
    from scripts.collect_agent_benchmark import BenchmarkStore, Boto3ObjectStore
    store = object.__new__(BenchmarkStore)
    store.client = Mock()
    store.client.get_object.return_value = {'Body': io.BytesIO(b'{"state":"failed"}')}
    monkeypatch.setattr(Boto3ObjectStore, 'exists', lambda self, bucket, key: key.endswith('/manifest.json'))
    with pytest.raises(RuntimeError, match='Remote feature extraction failed'):
        store.exists('bucket', 'feature-runs/test/bundle.zip')

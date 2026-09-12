import json
from types import SimpleNamespace

import pandas as pd
import pytest

from neuralsignal.remote.sync import sync_feature_run
from neuralsignal.training.s1 import _read_features
from neuralsignal.storage.manifests import sha256_file
from neuralsignal.features.model_extractor import ModelFeatureExtractor
from neuralsignal.features.selection import FeatureSetSpec
from neuralsignal.datasets.v2 import DatasetExample
from neuralsignal.features.malt_runs import aggregate_malt_runs


@pytest.mark.parametrize('state', ['running', 'failed'])
def test_sync_does_not_complete_unfinished_runs(tmp_path, state):
    def download(bucket, key, path):
        from pathlib import Path
        Path(path).write_text(json.dumps({'run_id': 'a', 'state': state, 'shards': []}))
    with pytest.raises(ValueError, match='completed'):
        sync_feature_run(SimpleNamespace(download_file=download), 's3://bucket/run', tmp_path)
    assert not (tmp_path / '.sync-complete').exists()


def test_sync_does_not_skip_failed_shards(tmp_path):
    def download(bucket, key, path):
        from pathlib import Path
        Path(path).write_text(json.dumps({'run_id': 'a', 'state': 'completed', 'shards': [
            {'state': 'failed', 'path': 'features/part-00000.parquet', 'sha256': 'unused'}]}))
    with pytest.raises(ValueError, match='shard'):
        sync_feature_run(SimpleNamespace(download_file=download), 's3://bucket/run', tmp_path)
    assert not (tmp_path / '.sync-complete').exists()


@pytest.mark.parametrize('state', ['running', 'failed'])
def test_training_rejects_unfinished_manifest(tmp_path, state):
    shard = tmp_path / 'part.parquet'
    pd.DataFrame({'zones__x': [1], 'label': [0]}).to_parquet(shard)
    (tmp_path / 'manifest.json').write_text(json.dumps({'state': state, 'shards': [
        {'path': 'part.parquet', 'rows': 1, 'sha256': sha256_file(shard)}]}))
    with pytest.raises(ValueError, match='completed'):
        _read_features(tmp_path)


@pytest.mark.parametrize('result', [([], []), (['zones__a', 'zones__b'], [1.]), (['zones__a', 'zones__a'], [1., 2.])])
def test_model_features_reject_partial_or_malformed_sets(monkeypatch, result):
    import neuralsignal.features.model_extractor as module
    class Processor:
        def __init__(self, feature_set_configs):
            self.name = feature_set_configs[0]['name']
        def featurize_scan(self, scan):
            return result if self.name == 'zones' else (['layer_distribution__x'], [1.])
    monkeypatch.setattr(module, 'FeatureProcessor', Processor)
    with pytest.raises(ValueError, match='zones'):
        ModelFeatureExtractor({})._featurize({}, DatasetExample('a', 'i', 'o'),
            [FeatureSetSpec('zones'), FeatureSetSpec('layer_distribution')])


def test_malt_pooling_does_not_hide_missing_values():
    data = pd.DataFrame([{'zones__x': value, 'metadata_json': json.dumps({
        'source': 'metr-evals/malt-public', 'label_scope': 'transcript', 'group_id': 'run',
        'run_labels': ['sabotage'], 'sample_index': index})} for index, value in enumerate([1., float('nan')])])
    with pytest.raises(ValueError, match='finite'):
        aggregate_malt_runs(data, ['zones__x'], 'sabotage')


def test_remote_job_rejects_unknown_extraction_mode(tmp_path, monkeypatch):
    import sys
    from neuralsignal.remote import job
    source = tmp_path / 'examples.jsonl'
    source.write_text('{"id":"a","input":"a","output":"b"}\n')
    config = {'extraction': {'mode': 'modle'}, 'dataset': {'source': 'jsonl', 'path': str(source)},
              'run': {'s3_output_uri': 's3://bucket/run'}, 'features': {'materialize': [{'name': 'zones'}]}}
    monkeypatch.setattr(job, '_load_config_from_env', lambda: config)
    monkeypatch.setenv('NEURALSIGNAL_RUN_WORKDIR', str(tmp_path / 'runs'))
    monkeypatch.setattr(sys, 'argv', ['job', '--run-id', 'a'])
    monkeypatch.setattr(job, '_s3_store', lambda: object())
    monkeypatch.setattr(job, 'upload_bundle', lambda *args: None)
    with pytest.raises(ValueError, match='extraction.mode'):
        job.main()


@pytest.mark.parametrize('mode', ['placeholder', 'modle'])
def test_local_collection_validates_mode_and_publishes_completion(tmp_path, mode):
    from neuralsignal.cli.main import _collect_local
    source = tmp_path / 'source.jsonl'
    source.write_text('{"id":"a","input":"a","output":"b"}\n')
    config = tmp_path / 'config.yaml'
    config.write_text(json.dumps({'extraction': {'mode': mode}, 'features': {'materialize': [{'name': 'zones'}]}}))
    out = tmp_path / 'run'
    args = SimpleNamespace(config=str(config), input_jsonl=str(source), out=str(out))
    if mode == 'modle':
        with pytest.raises(ValueError, match='extraction.mode'):
            _collect_local(args)
        assert not out.exists()
    else:
        assert _collect_local(args) == 0
        assert json.loads((out / 'manifest.json').read_text())['state'] == 'completed'
        assert len(_read_features(out)) == 1

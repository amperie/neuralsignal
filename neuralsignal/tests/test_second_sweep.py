import json
from types import SimpleNamespace

import pandas as pd
import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

from neuralsignal.storage.s3 import Boto3ObjectStore
from neuralsignal.remote.sync import sync_feature_run
from neuralsignal.storage.manifests import sha256_file
from neuralsignal.training.s1 import _read_features, train_s1


@pytest.mark.parametrize('code', ['403', 'AccessDenied', 'ExpiredToken', '500'])
def test_s3_lookup_propagates_service_errors(code):
    error = ClientError({'Error': {'Code': code}}, 'HeadObject')
    def head(**kwargs):
        raise error
    store = object.__new__(Boto3ObjectStore)
    store.client = SimpleNamespace(head_object=head)
    with pytest.raises(ClientError):
        store.exists('bucket', 'key')


def test_s3_lookup_propagates_connection_errors():
    def head(**kwargs):
        raise EndpointConnectionError(endpoint_url='https://fixture.invalid')
    store = object.__new__(Boto3ObjectStore)
    store.client = SimpleNamespace(head_object=head)
    with pytest.raises(EndpointConnectionError):
        store.exists('bucket', 'key')


@pytest.mark.parametrize('code', ['404', 'NoSuchKey', 'NotFound'])
def test_s3_lookup_missing_object(code):
    def head(**kwargs):
        raise ClientError({'Error': {'Code': code}}, 'HeadObject')
    store = object.__new__(Boto3ObjectStore)
    store.client = SimpleNamespace(head_object=head)
    assert store.exists('bucket', 'key') is False


@pytest.mark.parametrize('shard_path', ['../outside.parquet', '/tmp/outside.parquet'])
def test_sync_rejects_escaping_paths_and_clears_marker(tmp_path, shard_path):
    (tmp_path / '.sync-complete').write_text('old-run')
    calls = []
    def download(bucket, key, path):
        calls.append(key)
        if key.endswith('manifest.json'):
            from pathlib import Path
            Path(path).write_text(json.dumps({'run_id': 'new-run', 'shards': [
                {'path': shard_path, 'sha256': 'unused', 'state': 'written'}]}))
        else:
            raise AssertionError('must not download unsafe shard')
    with pytest.raises(ValueError, match='path'):
        sync_feature_run(SimpleNamespace(download_file=download), 's3://bucket/run', tmp_path)
    assert len(calls) == 1
    assert not (tmp_path / '.sync-complete').exists()


def test_failed_sync_clears_old_completion_marker(tmp_path):
    (tmp_path / '.sync-complete').write_text('old-run')
    def download(*args):
        raise OSError('connection lost')
    with pytest.raises(OSError):
        sync_feature_run(SimpleNamespace(download_file=download), 's3://bucket/run', tmp_path)
    assert not (tmp_path / '.sync-complete').exists()


def make_manifest_run(tmp_path):
    features = tmp_path / 'features'
    features.mkdir()
    shard = features / 'part-00000.parquet'
    pd.DataFrame({'zones__x': [1, 2], 'label': [0, 1]}).to_parquet(shard)
    manifest = {'run_id': 'fixture', 'state': 'completed', 'shards': [
        {'path': 'features/part-00000.parquet', 'sha256': sha256_file(shard), 'rows': 2}]}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    return shard


def test_training_uses_only_manifest_shards(tmp_path):
    make_manifest_run(tmp_path)
    pd.DataFrame({'zones__x': [999], 'label': [1]}).to_parquet(tmp_path / 'features/part-00001.parquet')
    assert _read_features(tmp_path)['zones__x'].tolist() == [1, 2]


def test_training_rejects_corrupt_shards(tmp_path):
    shard = make_manifest_run(tmp_path)
    pd.DataFrame({'zones__x': [999, 999], 'label': [0, 1]}).to_parquet(shard)
    with pytest.raises(ValueError, match='checksum'):
        _read_features(tmp_path)


@pytest.mark.parametrize('labels', [[0.5, 1.5] * 8, [1, 2] * 8])
def test_training_rejects_nonbinary_labels(tmp_path, labels):
    path = tmp_path / 'features.parquet'
    pd.DataFrame({'zones__x': range(len(labels)), 'label': labels}).to_parquet(path)
    with pytest.raises(ValueError, match='0 and 1'):
        train_s1(path, 'label')


@pytest.mark.parametrize('name', ['zones', 'layer_distribution'])
def test_smoke_feature_config_produces_features(name):
    import torch
    from neuralsignal.config import load_config
    from neuralsignal.core.modules.feature_sets.feature_processor import FeatureProcessor
    config = load_config('configs/feature_collection/smoke_runpod_malt.yaml')
    spec = next(item for item in config['features']['materialize'] if item['name'] == name)
    scan = {'layer_order': ['l0'], 'layer_id_to_name': {'l0': 'Dense'}, 'zone_size': 512,
            'outputs': {'l0': torch.tensor([[1., 2.], [3., 4.]])}}
    columns, values = FeatureProcessor(feature_set_configs=[{'name': name, **spec['config']}]).featurize_scan(scan)
    assert columns
    assert len(columns) == len(values)

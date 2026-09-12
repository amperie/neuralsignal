from types import SimpleNamespace

import pytest
import torch

from neuralsignal.core.modules import model_instrumentation as instrumentation
from neuralsignal.core.exceptions.NSAbortLLM import NSAbortLLM
from neuralsignal.datasets.v2 import DatasetExample
from neuralsignal.features.runner import collect_features
from neuralsignal.sdk.client import NeuralSignal
from neuralsignal.storage.local import LocalFeatureShardWriter
from neuralsignal.storage.manifests import RunManifest


class Tokenizer:
    eos_token_id = 0

    def __call__(self, values, **kwargs):
        self.options = kwargs
        return SimpleNamespace(input_ids=torch.ones((len(values), 2), dtype=torch.long))

    def decode(self, value, **kwargs):
        return 'decoded'


class Model:
    device = torch.device('cpu')
    name_or_path = 'fixture'

    def generate(self, input_ids, **kwargs):
        return input_ids


@pytest.mark.parametrize('limit', [0, 64])
def test_batch_respects_token_limit(limit):
    tokenizer = Tokenizer()
    instrumentation.generate_from_batch(['a', 'b'], Model(), tokenizer, truncation_length=limit)
    assert tokenizer.options['truncation'] is (limit > 0)
    assert tokenizer.options.get('max_length') == (limit or None)


def instrument_fixture(monkeypatch):
    removed = []
    monkeypatch.setattr(instrumentation, 'instrument_model', lambda *args: ['hook'])
    monkeypatch.setattr(instrumentation, 'deinstrument_model', lambda handles: removed.extend(handles))
    monkeypatch.setattr(instrumentation, 'Collector', lambda *args: SimpleNamespace(
        finish_and_get_data=lambda: None, get_data_by_batch_index=lambda index: {}))
    return removed


@pytest.mark.parametrize('stage', ['tokenize', 'decode'])
def test_hooks_removed_on_failure(monkeypatch, stage):
    removed = instrument_fixture(monkeypatch)

    class BrokenTokenizer(Tokenizer):
        def __call__(self, *args, **kwargs):
            if stage == 'tokenize':
                raise ValueError('fixture failure')
            return super().__call__(*args, **kwargs)

        def decode(self, *args, **kwargs):
            raise ValueError('fixture failure')

    with pytest.raises(ValueError, match='fixture failure'):
        instrumentation.generate_from_batch(['a'], Model(), BrokenTokenizer(), {})
    assert removed == ['hook']


def test_collector_abort_supports_large_batches(monkeypatch):
    removed = instrument_fixture(monkeypatch)

    class AbortingModel(Model):
        def generate(self, **kwargs):
            raise NSAbortLLM('collected requested layer')

    scans = instrumentation.generate_from_batch(['a'] * 8, AbortingModel(), Tokenizer(), {})
    assert len(scans) == 8
    assert all(scan.data['decoded_output'] == 'ABORT' for scan in scans)
    assert removed == ['hook']


@pytest.mark.parametrize('count', [0, 3])
def test_sdk_rejects_misaligned_results(count):
    client = NeuralSignal(evaluator=lambda *args: [{'scores': {}}] * count)
    with pytest.raises(RuntimeError, match='results'):
        client.evaluate_batch([{'input': 'a', 'output': 'b'}] * 2)


@pytest.mark.parametrize('failure', ['empty_dataset', 'empty_features', 'disabled_features'])
def test_collection_rejects_empty_output(tmp_path, failure):
    manifest = RunManifest('fixture', {'name': 'fixture'}, {'schema_version': 'features.v1'})
    writer = LocalFeatureShardWriter(tmp_path, manifest)
    examples = [] if failure == 'empty_dataset' else [DatasetExample(id='a', input='a', output='b')]
    config = {'features': {'materialize': [{'name': 'zones', 'enabled': failure != 'disabled_features'}]}}
    with pytest.raises((ValueError, RuntimeError), match='[Ee]xamples|[Ff]eature'):
        collect_features(examples, config, writer, lambda *args: {})
    assert manifest.rows['written'] == 0


def test_new_writer_rejects_existing_shards(tmp_path):
    manifest = RunManifest('fixture', {'name': 'fixture'}, {'schema_version': 'features.v1'})
    writer = LocalFeatureShardWriter(tmp_path, manifest)
    writer.write_shard([{'zones__a': 1}])
    original = (tmp_path / 'features' / 'part-00000.parquet').read_bytes()
    with pytest.raises(FileExistsError, match='shards'):
        LocalFeatureShardWriter(tmp_path, RunManifest('fixture', {'name': 'fixture'}, {'schema_version': 'features.v1'}))
    assert (tmp_path / 'features' / 'part-00000.parquet').read_bytes() == original


def test_bad_bundle_preserves_collected_results(tmp_path):
    import zipfile
    from neuralsignal.storage.bundle import unpack_bundle
    target = tmp_path / 'collected'
    target.mkdir()
    (target / 'manifest.json').write_text('previous results')
    bundle = tmp_path / 'broken.zip'
    bundle.write_bytes(b'not a zip')
    with pytest.raises(zipfile.BadZipFile):
        unpack_bundle(bundle, target)
    assert (target / 'manifest.json').read_text() == 'previous results'

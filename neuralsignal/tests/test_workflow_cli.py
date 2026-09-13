import importlib
import io
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from neuralsignal.cli.main import main
from neuralsignal.cli.selection import choose, choose_config, choose_run
from neuralsignal.training.s1 import train_s1


class Terminal(io.StringIO):
    def isatty(self):
        return True


def training_data(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'zones__x': [0., 1.] * 40, 'label': [0, 1] * 40}).to_parquet(path)
    return path


def test_config_picker_lists_nested_yaml_and_colors_eligible_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'configs/nested').mkdir(parents=True)
    (tmp_path / 'configs/a.yaml').write_text('runpod: {}')
    (tmp_path / 'configs/b.yml').write_text('dataset:\n  label_column: label\n')
    (tmp_path / 'configs/nested/c.yaml').write_text('extraction:\n  mode: model\n')
    monkeypatch.setattr('sys.stdin', Terminal())
    stream = Terminal()
    monkeypatch.setattr('sys.stdout', stream)
    monkeypatch.delenv('NO_COLOR', raising=False)
    monkeypatch.setenv('TERM', 'xterm')
    answers = iter(['no', '1', '3'])
    monkeypatch.setattr('builtins.input', lambda _: next(answers))
    assert choose_config('remote') == 'configs/nested/c.yaml'
    text = stream.getvalue()
    assert 'configs/a.yaml' in text and 'configs/b.yml' in text
    assert '\033[32mconfigs/nested/c.yaml' in text
    assert 'not selectable here' in text


def test_run_picker_finds_nested_runs_and_excludes_s1(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'runs/remote/collected/features').mkdir(parents=True)
    (tmp_path / 'runs/s1').mkdir()
    monkeypatch.setattr('sys.stdin', Terminal())
    monkeypatch.setattr('builtins.input', lambda _: '2')
    assert choose_run() == 'runs/remote/collected'
    text = capsys.readouterr().out
    assert 'runs/s1' in text and 'S1 training outputs' in text
    assert '\033[' not in text


@pytest.mark.parametrize('answer', ['q', None])
def test_picker_can_cancel(monkeypatch, answer):
    monkeypatch.setattr('sys.stdin', Terminal())
    def prompt(_):
        if answer is None:
            raise EOFError
        return answer
    monkeypatch.setattr('builtins.input', prompt)
    with pytest.raises(SystemExit, match='cancelled'):
        choose([(Path('config.yaml'), 'config', True)], 'Configs', 'Pass CONFIG')


def test_picker_scripts_get_actionable_error(monkeypatch):
    monkeypatch.setattr('sys.stdin', io.StringIO())
    with pytest.raises(SystemExit, match='Pass CONFIG'):
        choose([(Path('config.yaml'), 'config', True)], 'Configs', 'Pass CONFIG')


def test_train_picks_config_and_run_over_stale_yaml_path(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'configs').mkdir()
    (tmp_path / 'configs/train.yaml').write_text('dataset:\n  path: stale-path\n  label_column: label\nmlflow:\n  enabled: false\nmodel:\n  params:\n    n_estimators: 10\n')
    training_data(tmp_path / 'runs/remote/data/features/part-00000.parquet')
    monkeypatch.setattr('sys.stdin', Terminal())
    answers = iter(['1', '2'])
    monkeypatch.setattr('builtins.input', lambda _: next(answers))
    assert main(['train']) == 0
    text = capsys.readouterr().out
    assert 'Choose a training config' in text and 'Choose a feature run' in text
    output = next((tmp_path / 'runs/s1').iterdir())
    assert json.loads((output / 'training.json').read_text())['dataset_path'].endswith('runs/remote/data')
    assert '"accuracy": 1.0' in text


@pytest.mark.parametrize('stage', ['set_experiment', 'log_metrics', 'model'])
def test_mlflow_failures_keep_model_metrics_and_console_results(tmp_path, monkeypatch, capsys, stage):
    import mlflow
    import mlflow.xgboost
    monkeypatch.chdir(tmp_path)
    dataset = training_data(tmp_path / 'features.parquet')
    config = tmp_path / 'train.yaml'
    config.write_text('dataset:\n  label_column: label\nmodel:\n  params:\n    n_estimators: 10\n')
    tracking = []
    monkeypatch.setattr(mlflow, 'set_tracking_uri', tracking.append)
    monkeypatch.setattr(mlflow, 'set_experiment', lambda *a, **k: None)
    monkeypatch.setattr(mlflow, 'start_run', lambda **k: nullcontext())
    for name in ['log_params', 'log_metrics', 'log_dict', 'log_artifact']:
        monkeypatch.setattr(mlflow, name, lambda *a, **k: None)
    monkeypatch.setattr(mlflow.xgboost, 'log_model', lambda *a, **k: None)
    def fail(*a, **k):
        saved = list(Path('runs/s1').glob('*/model.ubj'))
        assert saved, 'Model must be durable before any MLflow operation'
        raise RuntimeError('reporting unavailable')
    if stage == 'model':
        monkeypatch.setattr(mlflow.xgboost, 'log_model', fail)
    else:
        monkeypatch.setattr(mlflow, stage, fail)
    monkeypatch.setenv('MLFLOW_TRACKING_URI', 'http://tracking.test:5000')
    assert main(['train', str(config), '--run', str(dataset)]) == 0
    captured = capsys.readouterr()
    assert 'MLflow reporting failed' in captured.err
    result = json.loads(captured.out)
    assert result['metrics']['accuracy'] == 1.0
    assert tracking == ['http://tracking.test:5000']
    output = Path(result['out'])
    assert {'model.ubj', 'metrics.json', 'training.json', 'selected_features.json',
            'predictions.csv', 'confusion_matrix.json', 'classification_report.json'} <= {p.name for p in output.iterdir()}
    assert json.loads((output / 'metrics.json').read_text()) == result['metrics']
    restored = XGBClassifier()
    restored.load_model(output / 'model.ubj')
    np.testing.assert_array_equal(restored.predict(pd.DataFrame({'zones__x': [0., 1.]})), [0, 1])


def test_run_downloads_then_trains_with_offline_tracking(tmp_path, monkeypatch, capsys):
    from neuralsignal.remote import lifecycle
    from neuralsignal.storage.bundle import create_bundle, upload_bundle
    from neuralsignal.tests.test_remote_lifecycle_v2 import FakeStore, FakeRunPod
    monkeypatch.chdir(tmp_path)
    source = tmp_path / 'worker-result'
    training_data(source / 'features/part-00000.parquet')
    handoff = FakeStore()
    bundle = create_bundle(source, tmp_path / 'bundle.zip')
    upload_bundle(handoff, bundle, 's3://handoff/feature-runs/workflow-test/bundle.zip')
    feature = tmp_path / 'feature.yaml'
    feature.write_text('run:\n  s3_output_uri: s3://handoff/feature-runs\ndataset:\n  source: jsonl\n')
    manifest = tmp_path / 'pod.yaml'
    manifest.write_text('runpod:\n  image: fixture\n')
    config = tmp_path / 'train.yaml'
    config.write_text('dataset:\n  path: intentionally-wrong\n  label_column: label\nmodel:\n  params:\n    n_estimators: 10\n')
    api = FakeRunPod()
    monkeypatch.setattr(lifecycle, 'DefaultRunPodApi', lambda: api)
    monkeypatch.setattr(lifecycle, 'Boto3ObjectStore', lambda **k: handoff)
    s1 = importlib.import_module('neuralsignal.training.s1')
    def offline(*a, **k):
        raise RuntimeError('offline')
    monkeypatch.setattr(s1, '_log_mlflow', offline)
    assert main(['run', str(feature), '--train-config', str(config), '--manifest', str(manifest),
                 '--run-id', 'workflow-test', '--terraform-dir', '', '--env-file', '']) == 0
    result = json.loads(capsys.readouterr().out)
    assert api.terminated == ['pod-1']
    assert result['training_metrics']['accuracy'] == 1.0
    assert Path(result['training_output_dir'], 'model.ubj').exists()
    assert Path(result['target_dir'], 'features/part-00000.parquet').exists()
    assert json.loads(Path(result['training_output_dir'], 'training.json').read_text())['dataset_path'].endswith('runs/remote/workflow-test')


def test_run_selects_configs_and_generates_id_before_dry_run(tmp_path, monkeypatch):
    cli = importlib.import_module('neuralsignal.cli.main')
    feature = tmp_path / 'feature.yaml'
    feature.write_text('dataset:\n  source: malt\n')
    train = tmp_path / 'train.yaml'
    train.write_text('dataset:\n  label_column: label\n')
    selected = []
    def choose(kind):
        selected.append(kind)
        return str(train if kind == 'training' else feature)
    monkeypatch.setattr(cli, 'choose_config', choose)
    captured = {}
    def collect(*args, **kwargs):
        captured.update(args=args, **kwargs)
        return {}
    monkeypatch.setattr(cli, 'remote_collect_lifecycle', collect)
    assert main(['run', '--dry-run']) == 0
    assert selected == ['remote', 'training']
    assert captured['args'][2].startswith('run-')
    assert captured['train_config_path'] == str(train)
    assert captured['dry_run'] is True

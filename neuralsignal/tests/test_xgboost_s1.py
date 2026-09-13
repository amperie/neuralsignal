import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from neuralsignal.config import load_config
from neuralsignal.training.s1 import train_s1


def training_data(tmp_path):
    path = tmp_path / 'features.parquet'
    pd.DataFrame({'zones__x': [0., 1.] * 40, 'label': [0, 1] * 40}).to_parquet(path)
    return path


def test_s1_trains_xgboost_with_configured_parameters(tmp_path):
    result = train_s1(training_data(tmp_path), 'label', model_config={'params': {'n_estimators': 12, 'max_depth': 2}}, mlflow_config={'enabled': False}, output_root=tmp_path / 's1')
    assert isinstance(result.model, XGBClassifier)
    assert result.model.get_params()['n_estimators'] == 12
    assert result.model.get_params()['tree_method'] == 'hist'
    assert result.metrics['auroc'] == 1.


def test_training_config_targets_remote_mlflow():
    config = load_config('configs/training/sabotage_s1.yaml')
    assert config['model']['type'] == 'xgboost'
    assert config['mlflow']['tracking_uri'] == 'http://z440.lan:5000'
    assert config['features']['include_sets'] == ['zones', 'layer_distribution']


def test_xgboost_mlflow_roundtrip(tmp_path, monkeypatch):
    import mlflow
    import mlflow.xgboost
    from mlflow import MlflowClient
    uri = f'sqlite:///{tmp_path / "mlflow.db"}'
    previous = mlflow.get_tracking_uri()
    previous_registry = mlflow.get_registry_uri()
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    try:
        result = train_s1(training_data(tmp_path), 'label',
            model_config={'params': {'n_estimators': 10}},
            mlflow_config={'tracking_uri': uri, 'experiment_name': 's1-test', 'run_name': 'xgb-roundtrip',
                           'registered_model_name': 's1-test-model'})
        client = MlflowClient(tracking_uri=uri)
        experiment = client.get_experiment_by_name('s1-test')
        runs = client.search_runs([experiment.experiment_id])
        assert len(runs) == 1
        run = runs[0]
        assert run.data.params['model_type'] == 'xgboost'
        assert run.data.params['xgboost.n_estimators'] == '10'
        assert run.data.metrics['auroc'] == result.metrics['auroc']
        artifacts = {item.path for item in client.list_artifacts(run.info.run_id)}
        assert {'selected_features.json', 'metrics.json', 'confusion_matrix.json',
                'confusion_matrix.csv', 'confusion_matrix.png', 'classification_report.json'} <= artifacts
        assert {'auc', 'auroc', 'auprc', 'accuracy', 'precision', 'recall', 'f1',
                'specificity', 'false_positive_rate', 'false_negative_rate', 'mcc',
                'log_loss', 'brier_score', 'tn', 'fp', 'fn', 'tp'} <= run.data.metrics.keys()
        matrix = json.loads(Path(client.download_artifacts(run.info.run_id, 'confusion_matrix.json')).read_text())
        assert matrix['matrix'] == [[10, 0], [0, 10]]
        assert matrix['rows'] == 'actual' and matrix['columns'] == 'predicted'
        report = json.loads(Path(client.download_artifacts(run.info.run_id, 'classification_report.json')).read_text())
        assert report['1']['precision'] == 1.
        assert report['1']['support'] == 10.
        png = Path(client.download_artifacts(run.info.run_id, 'confusion_matrix.png'))
        assert png.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
        restored = mlflow.xgboost.load_model('models:/s1-test-model/1')
        x = pd.DataFrame({'zones__x': [0., 1.]})
        np.testing.assert_allclose(restored.predict_proba(x), result.model.predict_proba(x))
    finally:
        mlflow.set_tracking_uri(previous)
        mlflow.set_registry_uri(previous_registry)


def test_classification_metrics_use_probabilities_and_correct_matrix_orientation():
    from neuralsignal.training.s1 import _classification_metrics
    metrics = _classification_metrics(np.array([0, 0, 0, 1, 1]), np.array([.1, .2, .8, .9, .3]))
    assert [metrics[k] for k in ['tn', 'fp', 'fn', 'tp']] == [2., 1., 1., 1.]
    assert metrics['auc'] == pytest.approx(5 / 6)
    assert metrics['auroc'] == metrics['auc']
    assert metrics['precision'] == metrics['recall'] == metrics['f1'] == .5
    assert metrics['accuracy'] == .6
    assert metrics['specificity'] == pytest.approx(2 / 3)
    assert metrics['false_positive_rate'] == pytest.approx(1 / 3)
    assert metrics['false_negative_rate'] == .5
    assert metrics['brier_score'] == pytest.approx(.238)


def test_single_class_slice_has_explicit_undefined_auc():
    from neuralsignal.training.s1 import _classification_metrics
    metrics = _classification_metrics(np.array([0, 0]), np.array([.1, .8]))
    assert np.isnan(metrics['auc'])
    assert np.isnan(metrics['false_negative_rate'])
    assert metrics['tn'] == metrics['fp'] == 1

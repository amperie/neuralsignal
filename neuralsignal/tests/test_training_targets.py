import io
import json
from pathlib import Path

import pandas as pd
import pytest

from neuralsignal.cli.main import main
from neuralsignal.cli.targets import select_target
from neuralsignal.datasets.sources.jsonl import _example_from_row
from neuralsignal.features.runner import _base_row
from neuralsignal.training.s1 import train_s1
from neuralsignal.training.targets import TargetError, encode_target, prepare_training_data


class Terminal(io.StringIO):
    def isatty(self):
        return True


def data():
    return pd.DataFrame({'zones__x': [0., 1.] * 20, 'target': [0, 1] * 20})


def test_legacy_numeric_column_and_automatic_target():
    for legacy in ('target', None):
        prepared, features, column, definition, summary = prepare_training_data(data(), ['zones__x'], label_column=legacy)
        assert prepared[column].tolist() == data().target.tolist()
        assert summary['positive'] == summary['negative'] == 20
        assert definition == {'source': 'column', 'column': 'target'}


def test_jsonl_preserves_arbitrary_targets_through_feature_collection():
    example = _example_from_row({'id': '1', 'input': 'q', 'output': 'a', 'label': 1, 'outcome': 'success'})
    row = _base_row('run', 0, example)
    assert json.loads(row['metadata_json']) == {'label': 1, 'outcome': 'success'}
    assert encode_target(pd.DataFrame([row]), {'source': 'metadata', 'column': 'label'}) == [1]


@pytest.mark.parametrize('source,column', [('column', 'outcome'), ('metadata', 'outcome')])
def test_category_mapping_and_exclusions(source, column):
    frame = data()
    values = ['success', 'failure', 'unknown', 'failure'] * 10
    frame['outcome'] = values
    frame['metadata_json'] = [json.dumps({'outcome': value}) for value in values]
    prepared, _, _, _, summary = prepare_training_data(frame, ['zones__x'], {
        'source': source, 'column': column, 'mapping': {'success': 1, 'failure': 0}, 'unmatched': 'exclude'})
    assert len(prepared) == 30
    assert summary['excluded'] == 10


def test_example_label_membership_and_conflicts():
    frame = data()
    frame['labels_json'] = [json.dumps(['positive'] if i % 2 else ['normal']) for i in range(len(frame))]
    definition = {'source': 'labels', 'positive_labels': ['positive'], 'negative_labels': ['normal']}
    prepared, _, column, _, _ = prepare_training_data(frame, ['zones__x'], definition)
    assert prepared[column].tolist() == frame.target.tolist()
    frame.loc[0, 'labels_json'] = '["positive", "normal"]'
    with pytest.raises(TargetError, match='both positive and negative'):
        encode_target(frame, definition)


def malt_data():
    rows = []
    for run in range(8):
        for sample in range(2):
            rows.append({'zones__x': float(run % 2), 'existing': run % 2,
                         'metadata_json': json.dumps({'source': 'metr-evals/malt-public', 'group_id': str(run),
                             'label_scope': 'transcript', 'run_labels': ['gives_up'] if run % 2 else ['normal'],
                             'sample_count': 2, 'sample_index': sample, 'completion_count': 1, 'completion_index': 0})})
    return pd.DataFrame(rows)


def test_malt_targets_are_run_scoped_and_existing_numeric_columns_still_work():
    for definition, legacy in [({'source': 'run_labels', 'positive_labels': ['gives_up'], 'unmatched': 'negative'}, None),
                               (None, 'existing')]:
        prepared, _, column, _, summary = prepare_training_data(malt_data(), ['zones__x'], definition, legacy)
        assert len(prepared) == 8 and summary['source_rows'] == 16
        assert summary['unit'] == 'run' and summary['positive'] == 4
        assert prepared[column].tolist() == [0, 1] * 4


def test_incomplete_malt_run_warns_and_prepares_available_samples(caplog):
    frame = malt_data().iloc[1:]
    prepared, _, _, _, summary = prepare_training_data(frame, ['zones__x'],
        {'source': 'run_labels', 'positive_labels': ['gives_up'], 'unmatched': 'negative'})
    assert len(prepared) == 8 and summary['training_units'] == 8
    assert 'continuing with available samples' in caplog.text


def test_generic_grouping_prevents_related_rows_becoming_independent_examples():
    frame = data()
    frame['group'] = [i // 4 for i in range(len(frame))]
    frame['target'] = frame['group'] % 2
    prepared, _, _, _, summary = prepare_training_data(frame, ['zones__x'], {'source': 'column', 'column': 'target', 'group_by': 'group'})
    assert len(prepared) == 10 and summary['unit'] == 'group'
    frame.loc[0, 'target'] = 1
    with pytest.raises(TargetError, match='Conflicting targets'):
        prepare_training_data(frame, ['zones__x'], {'source': 'column', 'column': 'target', 'group_by': 'group'})


def test_target_column_cannot_leak_into_features():
    frame = data().rename(columns={'target': 'zones__label'})
    _, features, _, _, _ = prepare_training_data(frame, ['zones__x', 'zones__label'], {'source': 'column', 'column': 'zones__label'})
    assert features == ['zones__x']


def test_interactive_categorical_target_mapping(tmp_path, monkeypatch, capsys):
    frame = data().drop(columns='target')
    frame['outcome'] = ['success', 'failure'] * 20
    path = tmp_path / 'data.parquet'
    frame.to_parquet(path)
    monkeypatch.setattr('sys.stdin', Terminal())
    answers = iter(['1', '1', '1'])
    monkeypatch.setattr('builtins.input', lambda _: next(answers))
    definition = select_target(path, {})
    assert definition['mapping'] == {'success': 1, 'failure': 0}
    assert 'column.outcome' in capsys.readouterr().out


def test_selected_target_and_splits_saved_with_real_training(tmp_path):
    path = tmp_path / 'data.parquet'
    data().to_parquet(path)
    result = train_s1(path, target_config={'source': 'column', 'column': 'target'},
                      mlflow_config={'enabled': False}, model_config={'params': {'n_estimators': 5}}, output_root=tmp_path / 's1')
    saved = json.loads(Path(result.output_dir, 'training.json').read_text())
    assert saved['target']['column'] == 'target'
    split = json.loads(Path(result.output_dir, 'split.json').read_text())
    assert set(split['train']).isdisjoint(split['test'])
    assert len(split['train']) + len(split['test']) == 40


def test_small_run_returns_actionable_error_without_traceback(tmp_path, capsys):
    frame = malt_data().iloc[:4]
    path = tmp_path / 'small.parquet'
    frame.to_parquet(path)
    config = tmp_path / 'train.yaml'
    config.write_text('model:\n  type: xgboost\n')
    assert main(['train', str(config), '--run', str(path), '--positive-label', 'gives_up']) == 2
    text = capsys.readouterr().err
    assert 'Insufficient data' in text and '2 independent runs' in text
    assert 'Traceback' not in text


def test_numeric_target_used_noninteractively_without_config(tmp_path, monkeypatch):
    path = tmp_path / 'data.parquet'
    data().to_parquet(path)
    monkeypatch.setattr('sys.stdin', io.StringIO())
    assert select_target(path, {}) == {'source': 'column', 'column': 'target'}


def test_explicit_target_overrides_legacy_label():
    from neuralsignal.training.targets import resolve_definition
    assert resolve_definition(data(), {'source': 'column', 'column': 'target'}, 'sabotage')['column'] == 'target'
    with pytest.raises(TargetError, match='binary targets only'):
        resolve_definition(data(), {'type': 'multiclass', 'source': 'column', 'column': 'target'})


def test_default_config_has_no_forced_target():
    from neuralsignal.config import load_config
    config = load_config('configs/training/s1.yaml')
    assert not config.get('target')
    assert not config['dataset'].get('label_column')
    assert 'sabotage' not in str(config)


def test_training_succeeds_with_partial_malt_runs(tmp_path, caplog):
    path = tmp_path / 'partial.parquet'
    malt_data().iloc[1:].to_parquet(path)
    result = train_s1(path, target_config={'source': 'run_labels', 'positive_labels': ['gives_up'], 'unmatched': 'negative'},
                      mlflow_config={'enabled': False}, model_config={'params': {'n_estimators': 5}}, output_root=tmp_path / 's1')
    assert result.metrics['train_rows'] == 6
    assert result.metrics['test_rows'] == 2
    assert Path(result.output_dir, 'model.ubj').exists()
    assert 'Incomplete MALT run' in caplog.text


def test_duplicate_completions_still_fail():
    frame = malt_data()
    frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    with pytest.raises(TargetError, match='Duplicate completions'):
        prepare_training_data(frame, ['zones__x'], {'source': 'column', 'column': 'existing'})

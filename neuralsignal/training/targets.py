"""Resolve binary supervision independently of the feature source adapter."""
from __future__ import annotations

import json
import math

import pandas as pd


class TargetError(ValueError):
    """An actionable target or training-data validation error."""


def metadata_rows(data):
    return [json.loads(value) for value in data['metadata_json']] if 'metadata_json' in data else [{} for _ in range(len(data))]


def is_malt(data):
    return any(row.get('source') == 'metr-evals/malt-public' for row in metadata_rows(data))


def is_fibs(data):
    from neuralsignal.datasets.sources.fibs import FIBS_DATASET
    return any(row.get('source') == FIBS_DATASET for row in metadata_rows(data))


# FIBS labels come from an AI judge (``judge_verdict``), not human annotation.
FIBS_DECEPTIVE_TARGET = {'source': 'fibs_verdict', 'column': 'judge_verdict', 'name': 'deceptive',
                         'unmatched': 'exclude'}


def target_values(data, definition):
    source = definition.get('source', 'column')
    column = definition.get('column')
    if source == 'column':
        if column not in data:
            raise TargetError(f'Missing target column: {column}')
        return data[column].tolist()
    if source == 'metadata':
        return [row.get(column) for row in metadata_rows(data)]
    if source == 'labels':
        if 'labels_json' not in data:
            raise TargetError('This dataset has no labels_json column.')
        return [json.loads(value) for value in data['labels_json']]
    if source == 'run_labels':
        return [row.get('run_labels') for row in metadata_rows(data)]
    if source == 'fibs_verdict':
        from neuralsignal.datasets.sources.fibs import fibs_verdict_label
        return [fibs_verdict_label(row.get(column or 'judge_verdict')) for row in metadata_rows(data)]
    raise TargetError(f'Unsupported target source: {source}. Use column, metadata, labels, run_labels, or fibs_verdict.')


def resolve_definition(data, target=None, label_column=None):
    if target:
        if not isinstance(target, dict):
            raise TargetError('target must be a YAML mapping.')
        if target.get('type', 'binary') != 'binary':
            raise TargetError('S1 currently supports binary targets only.')
        return dict(target)
    if label_column:
        # Existing Parquet targets take precedence over dataset-specific conversion.
        if label_column in data:
            return {'source': 'column', 'column': label_column}
        if label_column == 'deceptive' and is_fibs(data):
            return dict(FIBS_DECEPTIVE_TARGET)
        if is_malt(data):
            return {'source': 'run_labels', 'positive_labels': [label_column], 'unmatched': 'negative'}
        raise TargetError(f'Missing target column: {label_column}. Configure target or choose one interactively.')
    for column in ('target', 'label'):
        if column in data:
            numeric = pd.to_numeric(data[column], errors='coerce')
            if numeric.notna().all() and numeric.isin([0, 1]).all():
                return {'source': 'column', 'column': column}
    raise TargetError('No target defined. Choose one interactively or supply target in the training YAML.')


def encode_target(data, definition):
    values = target_values(data, definition)
    source = definition.get('source', 'column')
    unmatched = definition.get('unmatched', 'error')
    if unmatched not in {'error', 'exclude', 'negative'}:
        raise TargetError('target.unmatched must be error, exclude, or negative.')
    labels = source in {'labels', 'run_labels'}
    mapping = definition.get('mapping')
    positive = definition.get('positive_labels')
    negative = definition.get('negative_labels')
    if labels and not positive:
        raise TargetError('Label membership requires target.positive_labels.')
    if positive and negative and set(positive) & set(negative):
        raise TargetError('Positive and negative labels must not overlap.')
    result = []
    for value in values:
        missing = value is None or (isinstance(value, float) and math.isnan(value))
        encoded = None
        if labels and not missing:
            if not isinstance(value, list):
                raise TargetError('Label membership requires lists of labels, not scalar values.')
            pos = bool(set(value) & set(positive))
            neg = bool(set(value) & set(negative or []))
            if pos and neg:
                raise TargetError('An example matches both positive and negative labels.')
            encoded = 1 if pos else 0 if neg else None
        elif mapping is not None and not missing:
            try:
                encoded = mapping.get(value, mapping.get(str(value)))
            except TypeError:
                raise TargetError('Mapped target values must be scalar.') from None
        elif not missing:
            try:
                encoded = float(value)
            except (ValueError, TypeError):
                raise TargetError('Non-numeric target values require an explicit mapping to 0 and 1.') from None
        if encoded is None:
            if unmatched == 'exclude':
                result.append(None)
                continue
            if unmatched == 'negative' and not missing:
                encoded = 0
            else:
                raise TargetError('Missing or unmapped target value. Define a mapping or set unmatched: exclude.')
        if encoded not in (0, 1):
            raise TargetError('Binary targets must contain only 0 and 1.')
        result.append(int(encoded))
    return result


def prepare_training_data(data, features, target=None, label_column=None):
    definition = resolve_definition(data, target, label_column)
    features = list(features)
    if definition.get('source', 'column') == 'column':
        features = [name for name in features if name != definition.get('column')]
    if definition.get('group_by'):
        features = [name for name in features if name != definition['group_by']]
    if not features:
        raise TargetError('No feature columns remain after excluding the target.')
    values = encode_target(data, definition)
    target_name = '_ns_target'
    if target_name in features:
        raise TargetError('Feature name _ns_target is reserved for target preparation.')
    if is_malt(data):
        from neuralsignal.features.malt_runs import aggregate_malt_runs
        try:
            prepared = aggregate_malt_runs(data, features, target_name, targets=values)
        except ValueError as error:
            raise TargetError(str(error)) from error
        unit = 'run'
    elif is_fibs(data):
        prepared = _prepare_fibs_rows(data, features, target_name, values)
        unit = 'transcript'
    elif definition.get('group_by'):
        group_key = definition['group_by']
        groups = target_values(data, {'source': 'metadata' if group_key.startswith('metadata.') else 'column',
                                      'column': group_key.removeprefix('metadata.')})
        grouped = data[features].copy()
        grouped[target_name] = values
        grouped['_ns_group'] = groups
        if grouped['_ns_group'].isna().any():
            raise TargetError('Missing group IDs; cannot split independent groups.')
        rows = []
        for group, frame in grouped.groupby('_ns_group', sort=False):
            if frame[target_name].nunique(dropna=False) != 1:
                raise TargetError(f'Conflicting targets within group {group}.')
            rows.append({**frame[features].mean().to_dict(), target_name: frame[target_name].iloc[0], 'group_id': group})
        prepared = pd.DataFrame(rows, columns=[*features, target_name, 'group_id'])
        unit = 'group'
    else:
        if definition.get('source') == 'run_labels':
            raise TargetError('Run labels require target.group_by for non-MALT datasets.')
        prepared = data.copy()
        prepared[target_name] = values
        unit = 'example'
    excluded = int(prepared[target_name].isna().sum())
    prepared = prepared[prepared[target_name].notna()].reset_index(drop=True)
    counts = prepared[target_name].value_counts().to_dict()
    summary = {'source_rows': len(data), 'training_units': len(prepared), 'unit': unit,
               'negative': int(counts.get(0, 0)), 'positive': int(counts.get(1, 0)), 'excluded': excluded}
    n_test = math.ceil(len(prepared) * 0.25)
    if min(summary['negative'], summary['positive']) < 2 or n_test < 2 or len(prepared) - n_test < 2:
        raise TargetError(f"Insufficient data: {summary['training_units']} independent {unit}s "
                          f"({summary['negative']} negative, {summary['positive']} positive; {excluded} excluded). "
                          'Need both classes with at least two units each and enough units for the 75/25 split. '
                          'Collect more complete labeled data.')
    return prepared, features, target_name, definition, summary


def _prepare_fibs_rows(data, features, target_name, values):
    """One unit per FIBS transcript: mean-pool chunk rows, keep episode/split metadata."""
    from neuralsignal.training.fibs_eval import BASELINE_COLUMNS, episode_groups
    metadata = metadata_rows(data)
    frame = data[features].astype(float).copy()
    frame[target_name] = values
    frame['_ns_row'] = [row.get('fibs_row_id') or row.get('group_id') for row in metadata]
    if frame['_ns_row'].isna().any():
        raise TargetError('FIBS row is missing fibs_row_id; recollect with the fibs dataset source.')
    rows = []
    for row_id, chunk in frame.groupby('_ns_row', sort=False):
        first = metadata[chunk.index[0]]
        if chunk[target_name].nunique(dropna=False) > 1:
            raise TargetError(f'Conflicting targets across chunks of FIBS row {row_id}.')
        row = {**chunk[features].mean().to_dict(), target_name: chunk[target_name].iloc[0],
               'fibs_split': first.get('fibs_split'), 'fibs_row_id': row_id,
               '_ns_group_keys': first.get('group_keys') or [], '_ns_fallback_group': first.get('group_id') or row_id}
        for key, column in BASELINE_COLUMNS.items():
            row[column] = float(first.get(key) or 0)
        rows.append(row)
    prepared = pd.DataFrame(rows)
    prepared['group_id'] = episode_groups(prepared['_ns_group_keys'].tolist(), prepared['_ns_fallback_group'].tolist())
    return prepared.drop(columns=['_ns_group_keys', '_ns_fallback_group'])

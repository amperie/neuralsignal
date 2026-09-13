"""Discover supervision from the selected feature dataset, not a behavior preset."""
import sys

import pandas as pd

from neuralsignal.cli.selection import choose
from neuralsignal.training.targets import TargetError, metadata_rows, is_malt, target_values, resolve_definition


_EXCLUDED = {'input', 'output', 'example_id', 'run_id', 'row_index', 'group_id', 'source',
             'sample_index', 'completion_index', 'sample_count', 'completion_count',
             'label_scope', 'run_source', 'labels_json', 'metadata_json'}


def candidates(data):
    found = []
    for column in data:
        if column not in _EXCLUDED and ('__' not in column or column in {'label', 'target'}):
            found.append({'source': 'column', 'column': column})
    metadata = metadata_rows(data)
    for column in sorted({key for row in metadata for key in row} - _EXCLUDED - {'run_labels'}):
        found.append({'source': 'metadata', 'column': column})
    if 'labels_json' in data and not is_malt(data):
        found.append({'source': 'labels'})
    if any('run_labels' in row for row in metadata):
        found.append({'source': 'run_labels'})
    usable = []
    for item in found:
        values = target_values(data, item)
        if item['source'] in {'labels', 'run_labels'}:
            if not all(value is None or isinstance(value, list) for value in values):
                continue
            distinct = sorted({label for value in values if value is not None for label in value}, key=str)
        else:
            if any(isinstance(value, (list, dict)) for value in values):
                continue
            distinct = list(pd.Series(values).dropna().unique())
        if distinct and len(distinct) <= 30:
            usable.append((item, distinct))
    return usable


def _pick(options, title):
    entries = [(text, '', True) for text, _ in options]
    selected = choose(entries, title, 'Set target in the training YAML or pass --target-column.')
    return next(value for text, value in options if text == selected)


def select_target(dataset_path, config, target_column=None, positive_label=None):
    from neuralsignal.training.s1 import _read_features
    if target_column and positive_label:
        raise TargetError('Use either --target-column or --positive-label, not both.')
    if target_column:
        return {'source': 'metadata' if target_column.startswith('metadata.') else 'column',
                'column': target_column.removeprefix('metadata.')}
    if config.get('target') and not positive_label:
        return config['target']
    legacy = (config.get('dataset') or {}).get('label_column')
    if legacy and not positive_label:
        return None  # Preserve existing numeric columns and legacy MALT label configs.
    data = _read_features(dataset_path)
    if positive_label:
        return {'source': 'run_labels' if is_malt(data) else 'labels',
                'positive_labels': [positive_label], 'unmatched': 'negative'}
    if not sys.stdin.isatty():
        # Existing conventional numeric columns remain usable by unattended callers.
        return resolve_definition(data)
    available = candidates(data)
    if not available:
        raise TargetError('No usable target columns or label lists found. Preserve labels or a target field during collection.')
    options = []
    for definition, distinct in available:
        preview = ', '.join(str(value) for value in distinct[:8])
        name = definition['source'] + ('.' + definition['column'] if 'column' in definition else '')
        options.append((f'{name}: {preview}', (definition, distinct)))
    definition, distinct = _pick(options, 'Choose the target source (available columns and label lists)')
    if definition['source'] in {'column', 'metadata'}:
        numeric = pd.to_numeric(pd.Series(distinct), errors='coerce')
        if numeric.notna().all() and numeric.isin([0, 1]).all():
            return definition
    positive = _pick([(str(value), value) for value in distinct], 'Choose the positive class (target = 1)')
    negative = _pick([('All other observed values/labels are negative', None)] +
                     [(f'Only {value}; exclude other values', value) for value in distinct if value != positive],
                     'Choose the negative class (target = 0)')
    if definition['source'] in {'labels', 'run_labels'}:
        definition['positive_labels'] = [positive]
        if negative is not None:
            definition['negative_labels'] = [negative]
        definition['unmatched'] = 'negative' if negative is None else 'exclude'
    else:
        # Convert NumPy scalar values to serializable native Python values.
        native = lambda value: value.item() if hasattr(value, 'item') else value
        definition['mapping'] = {native(value): int(value == positive) for value in distinct
                                 if negative is None or value in (positive, negative)}
        definition['unmatched'] = 'error' if negative is None else 'exclude'
    if definition['source'] == 'run_labels' and not is_malt(data):
        groups = []
        for key in ('group_id', 'run_id'):
            if key in data:
                groups.append((key, key))
            if any(key in row for row in metadata_rows(data)):
                groups.append(('metadata.' + key, 'metadata.' + key))
        if not groups:
            raise TargetError('Run labels require group IDs. Set target.group_by to a column or metadata field in YAML.')
        definition['group_by'] = _pick(groups, 'Choose the run/group identifier (features are averaged within each group)')
    return definition

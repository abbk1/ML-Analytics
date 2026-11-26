from io import BytesIO
import numpy as np
import pandas as pd
import pytest
from core.config import ExperimentConfig
from core.data import load_dataset, audit_dataset, dataset_fingerprint, prepare_dataset, stratified_sample


@pytest.mark.parametrize('extension,separator', [('csv', ','), ('tsv', '\t')])
def test_text_upload(numeric_data, extension, separator):
    frame = load_dataset(numeric_data.to_csv(index=False, sep=separator).encode(), f'data.{extension}')
    assert frame.shape == numeric_data.shape


def test_excel_upload(numeric_data):
    output = BytesIO()
    numeric_data.to_excel(output, index=False)
    assert load_dataset(output.getvalue(), 'data.xlsx').shape == numeric_data.shape


@pytest.mark.parametrize('content,name', [(b'', 'a.csv'), (b'hello', 'a.py'), (b'a\n', 'a.csv')])
def test_bad_upload(content, name):
    with pytest.raises(ValueError):
        load_dataset(content, name)


def test_upload_size_limit():
    with pytest.raises(ValueError, match='no larger'):
        load_dataset(b'x' * 1025, 'data.csv', max_upload_mb=0.0001)


def test_audit_and_missing_targets(mixed_data):
    mixed_data.loc[0, 'target'] = None
    mixed_data['id'] = range(len(mixed_data))
    mixed_data['constant'] = 1
    mixed_data['leak'] = mixed_data.target
    audit = audit_dataset(mixed_data, 'target')
    assert audit['missing'] == 4
    assert len(audit['warnings']) >= 4
    config = ExperimentConfig('target', ['a', 'category'])
    with pytest.raises(ValueError, match='missing labels'):
        prepare_dataset(mixed_data, config)
    config.preprocessing.drop_missing_target = True
    X, y, messages = prepare_dataset(mixed_data, config)
    assert 0 not in X.index and not y.isna().any() and messages


def test_explicit_cleaning_and_conflicts():
    frame = pd.DataFrame({'a': [1, 1, 1, 2, 3, np.nan, 4, 5], 'target': ['a', 'a', 'b', 'b', 'a', 'b', 'a', 'b']})
    config = ExperimentConfig('target', ['a'])
    config.preprocessing.drop_duplicates = True
    config.preprocessing.drop_missing_features = True
    X, _, messages = prepare_dataset(frame, config)
    assert len(X) == 6
    assert any('conflicting' in m for m in messages)
    assert any('duplicate' in m for m in messages)


def test_fingerprint_and_sampling(numeric_data):
    assert dataset_fingerprint(numeric_data) == dataset_fingerprint(numeric_data.copy())
    changed = numeric_data.copy()
    changed.loc[0, 'a'] += 1
    assert dataset_fingerprint(changed) != dataset_fingerprint(numeric_data)
    first = stratified_sample(numeric_data, 'target', 60)
    second = stratified_sample(numeric_data, 'target', 60)
    pd.testing.assert_frame_equal(first, second)
    assert first.target.nunique() == 2

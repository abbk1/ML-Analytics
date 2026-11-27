import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from default_studio import data
from default_studio.data import FEATURES, TARGET, validate_record


def test_complete_dataset_and_provenance(dataset):
    assert dataset.shape == (30000, 25)
    assert len(FEATURES) == 23
    assert set(dataset.ID) == set(range(1, 30001))
    assert dataset[TARGET].value_counts().to_dict() == {0: 23364, 1: 6636}
    assert dataset.isna().sum().sum() == 0
    info = data.metadata()
    assert info['compressed_bytes'] < 1024**2
    assert info['source_csv_sha256'] == '4edce6da569b2c30788a85fd98a9bafb1a983e8ca1a7aac875dd7ff853cbd0bb'
    assert info['bundled_sha256'] == hashlib.sha256((data.DATA_DIR / 'credit_card_default.csv.gz').read_bytes()).hexdigest()


def test_no_original_category_values_removed(dataset):
    assert {0, 5, 6}.issubset(set(dataset.EDUCATION))
    assert 0 in set(dataset.MARRIAGE)
    assert (dataset[data.BILLS] < 0).any().any()
    assert list(dataset.ID.iloc[:3]) == [1, 2, 3]


def test_corrupt_bundle_rejected(tmp_path, monkeypatch):
    (tmp_path / 'metadata.json').write_text(json.dumps(data.metadata()))
    (tmp_path / 'credit_card_default.csv.gz').write_bytes(b'corrupt')
    monkeypatch.setattr(data, 'DATA_DIR', tmp_path)
    with pytest.raises(ValueError, match='checksum'):
        data.load_dataset()


@pytest.mark.parametrize('field,value,message', [
    ('AGE', 12, 'Age'), ('LIMIT_BAL', -100, 'credit limit'),
    ('PAY_AMT1', -1, 'payment amounts'), ('PAY_0', 100, 'supported code'),
    ('SEX', 3, 'supported code'), ('BILL_AMT1', np.inf, 'finite'),
])
def test_invalid_prediction_values(dataset, field, value, message):
    record = dataset.iloc[[0]].copy().astype(float)
    record.loc[0, field] = value
    with pytest.raises(ValueError, match=message):
        validate_record(record)


def test_prediction_schema_and_negative_bill(dataset):
    record = dataset.iloc[[0]].copy()
    record.loc[0, 'BILL_AMT1'] = -100
    values = validate_record(record)
    assert list(values.columns) == FEATURES
    assert 'ID' not in values and TARGET not in values
    assert values.BILL_AMT1.iloc[0] == -100
    with pytest.raises(ValueError, match='Missing required'):
        validate_record(record.drop(columns='AGE'))

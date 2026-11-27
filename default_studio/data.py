"""The complete, checksum-verified UCI public dataset and its fixed schema."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
TARGET = 'default_payment_next_month'
PAYMENT_STATUS = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
BILLS = [f'BILL_AMT{i}' for i in range(1, 7)]
PAYMENTS = [f'PAY_AMT{i}' for i in range(1, 7)]
FEATURES = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', *PAYMENT_STATUS, *BILLS, *PAYMENTS]
CATEGORICAL = ['SEX', 'EDUCATION', 'MARRIAGE', *PAYMENT_STATUS]
NUMERICAL = [column for column in FEATURES if column not in CATEGORICAL]
MONTHS = ['September', 'August', 'July', 'June', 'May', 'April']
CLASS_LABELS = {0: 'No default', 1: 'Default'}


def metadata():
    return json.loads((DATA_DIR / 'metadata.json').read_text())


def validate_dataset(frame):
    if frame.shape != (30000, 25) or set(frame.columns) != {'ID', TARGET, *FEATURES}:
        raise ValueError('The bundled dataset must contain all 30,000 customers and the expected 23 predictors.')
    if frame.isna().any().any() or not all(pd.api.types.is_numeric_dtype(frame[c]) for c in frame):
        raise ValueError('The public dataset must contain its original nonmissing numeric values.')
    if frame.ID.nunique() != 30000 or not frame.ID.between(1, 30000).all():
        raise ValueError('Every original customer identifier must occur exactly once.')
    if frame[TARGET].value_counts().to_dict() != {0: 23364, 1: 6636}:
        raise ValueError('The full original class distribution must be preserved.')


def load_dataset():
    path = DATA_DIR / 'credit_card_default.csv.gz'
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata()['bundled_sha256']:
        raise ValueError('Bundled dataset checksum failed. Restore the verified data file before continuing.')
    frame = pd.read_csv(path, compression='gzip', dtype=np.int32)
    validate_dataset(frame)
    return frame


def validate_record(frame):
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing required fields: {sorted(missing)}')
    values = frame.loc[:, FEATURES].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError('Enter a finite numeric value for every field.')
    if not values.AGE.between(18, 100).all() or (values.LIMIT_BAL < 0).any():
        raise ValueError('Age must be 18–100 and the credit limit must be nonnegative.')
    if (values[PAYMENTS] < 0).any().any():
        raise ValueError('Previous payment amounts must be nonnegative. Bill balances may be negative.')
    allowed = {'SEX': {1, 2}, 'EDUCATION': set(range(7)), 'MARRIAGE': {0, 1, 2, 3},
               **{column: set(range(-2, 10)) for column in PAYMENT_STATUS}}
    for column, choices in allowed.items():
        if not values[column].isin(choices).all():
            raise ValueError(f'Choose a supported code for {column}.')
    return values.astype(np.float32)

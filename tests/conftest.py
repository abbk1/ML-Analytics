import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ml-analytics-matplotlib')
os.environ.setdefault('OMP_NUM_THREADS', '1')


@pytest.fixture
def numeric_data():
    X, y = make_classification(n_samples=150, n_features=5, n_informative=3, n_redundant=0,
                               weights=[0.8, 0.2], random_state=4)
    frame = pd.DataFrame(X, columns=list('abcde'))
    frame['target'] = y.astype(str)
    return frame


@pytest.fixture
def mixed_data(numeric_data):
    result = numeric_data.copy()
    result['category'] = np.where(result.a > 0, 'red', 'blue')
    result.loc[3, 'category'] = None
    result.loc[7, 'a'] = np.nan
    return result


@pytest.fixture
def text_data():
    return pd.DataFrame({'text': [
        f'{"happy good wonderful" if i % 3 else "bad awful sad"} review number {i}' for i in range(90)],
        'target': ['positive' if i % 3 else 'negative' for i in range(90)]})

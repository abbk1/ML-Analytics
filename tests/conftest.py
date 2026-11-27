from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope='session')
def dataset():
    from default_studio.data import load_dataset
    return load_dataset()


@pytest.fixture(scope='session')
def benchmark(dataset):
    from default_studio.modeling import train_models
    return train_models(dataset)

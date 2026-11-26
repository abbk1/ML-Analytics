import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from core.config import ExperimentConfig
from core.preprocessing import build_pipeline, TextCleaner, BoundedDense
from core.prediction import predict_batch
from core.resampling import RecordingSampler
from core.models import make_model


@pytest.mark.parametrize('method', ['none', 'standard', 'minmax', 'robust', 'maxabs'])
def test_scaling_and_imputation_training_only(method):
    X = pd.DataFrame({'x': [0., 2., np.nan, 4.]})
    y = pd.Series(['a', 'a', 'b', 'b'])
    config = ExperimentConfig('target', ['x'])
    config.preprocessing.scaling = method
    pipeline = build_pipeline(X, config, 'Dummy baseline').fit(X, y)
    numerical = pipeline.named_steps['preprocess'].named_transformers_['numeric']
    assert numerical.named_steps['impute'].statistics_[0] == 2
    if method == 'standard':
        assert numerical.named_steps['scale'].mean_[0] == 2
    # An extreme test observation must not affect any fitted statistics.
    pipeline.predict(pd.DataFrame({'x': [100000.]}))
    assert numerical.named_steps['impute'].statistics_[0] == 2


@pytest.mark.parametrize('imputation', ['mean', 'median', 'most_frequent', 'constant'])
def test_missing_strategies(mixed_data, imputation):
    config = ExperimentConfig('target', ['a', 'category'])
    config.preprocessing.imputation = imputation
    pipeline = build_pipeline(mixed_data[config.features], config, 'Logistic Regression')
    pipeline.fit(mixed_data[config.features], mixed_data.target)
    assert len(pipeline.predict(pd.DataFrame({'a': [None], 'category': ['never seen']}))) == 1


@pytest.mark.parametrize('encoding', ['onehot', 'ordinal'])
def test_unknown_categories(mixed_data, encoding):
    config = ExperimentConfig('target', ['a', 'category'])
    config.preprocessing.encoding = encoding
    pipeline = build_pipeline(mixed_data[config.features], config, 'Random Forest', {'n_estimators': 5})
    pipeline.fit(mixed_data[config.features], mixed_data.target)
    assert pipeline.predict(pd.DataFrame({'a': [1.], 'category': ['unknown']})).shape == (1,)


def test_tfidf_sparse_and_no_vocabulary_leakage(text_data):
    config = ExperimentConfig('target', ['text'], text_column='text')
    config.preprocessing.task = 'Text'
    pipeline = build_pipeline(text_data[['text']], config, 'Multinomial NB')
    pipeline.fit(text_data[['text']], text_data.target)
    vectorizer = pipeline.named_steps['preprocess'].named_transformers_['text'].named_steps['tfidf']
    assert 'testonlytoken' not in vectorizer.vocabulary_
    pipeline.predict(pd.DataFrame({'text': ['testonlytoken never seen']}))
    assert 'testonlytoken' not in vectorizer.vocabulary_
    assert sparse.issparse(pipeline.named_steps['preprocess'].transform(text_data[['text']]))
    cleaned = TextCleaner(remove_stopwords=True).transform(['no not never the good'])[0]
    assert 'no' in cleaned.split() and 'not' in cleaned.split() and 'the' not in cleaned.split()


@pytest.mark.parametrize('method', ['random_over', 'random_under', 'smote', 'smote_tomek'])
def test_numeric_resamplers(numeric_data, method):
    config = ExperimentConfig('target', list('abcde'))
    config.resampling.method = method
    config.resampling.k_neighbors = 2
    pipeline = build_pipeline(numeric_data[config.features], config, 'Logistic Regression')
    pipeline.fit(numeric_data[config.features], numeric_data.target)
    sampler = pipeline.named_steps['resample']
    assert sampler.before_counts_['0'] > sampler.before_counts_['1']
    assert sampler.after_counts_['1'] >= sampler.before_counts_['1'] if method != 'random_under' else True


def test_smotenc_and_incompatible_ordinary_smote(mixed_data):
    config = ExperimentConfig('target', ['a', 'category'])
    config.resampling.method = 'smotenc'
    config.resampling.k_neighbors = 2
    pipeline = build_pipeline(mixed_data[config.features], config, 'Logistic Regression')
    pipeline.fit(mixed_data[config.features], mixed_data.target)
    assert pipeline.named_steps['resample'].after_counts_['0'] == pipeline.named_steps['resample'].after_counts_['1']
    assert len(pipeline.predict(pd.DataFrame({'a': [None], 'category': ['unseen']}))) == 1
    config.resampling.method = 'smote'
    with pytest.raises(ValueError, match='categorical'):
        build_pipeline(mixed_data[config.features], config, 'Logistic Regression')


def test_minority_and_allocation_guards():
    X = np.arange(20).reshape(10, 2)
    y = np.array(['a'] * 8 + ['b'] * 2)
    with pytest.raises(ValueError, match='training-fold'):
        RecordingSampler('smote', k_neighbors=2).fit_resample(X, y)
    with pytest.raises(ValueError, match='limit'):
        RecordingSampler('random_over', max_rows=12).fit_resample(X, y)
    with pytest.raises(ValueError, match='dense input'):
        BoundedDense(max_cells=5).fit(X)


def test_sparse_text_resampling(text_data):
    config = ExperimentConfig('target', ['text'], text_column='text')
    config.preprocessing.task = 'Text'
    config.resampling.method = 'smote'
    config.resampling.k_neighbors = 2
    pipeline = build_pipeline(text_data[['text']], config, 'Linear SVM')
    pipeline.fit(text_data[['text']], text_data.target)
    assert sparse.issparse(pipeline.named_steps['preprocess'].transform(text_data[['text']]))


def test_model_parameter_compatibility():
    with pytest.raises(ValueError, match='solver'):
        make_model('Logistic Regression', {'solver': 'lbfgs', 'penalty': 'l1'})
    with pytest.raises(ValueError, match='n_estimators'):
        make_model('Random Forest', {'n_estimators': 10000})
    with pytest.raises(ValueError, match='hidden layers'):
        make_model('MLP', {'hidden_layer_sizes': [10000]})
    assert make_model('Random Forest').n_jobs == 1

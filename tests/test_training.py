import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from core.config import ExperimentConfig
from core.models import DEFAULTS
from core.training import train_experiments, split_dataset
from core.prediction import predict_batch
from core.evaluation import evaluate_predictions


@pytest.fixture
def config():
    value = ExperimentConfig('target', list('abcde'))
    value.split.folds = 3
    value.resampling.k_neighbors = 2
    return value


def test_shared_holdout_and_no_test_resampling(numeric_data, config):
    results, failures = train_experiments(numeric_data, config, {'Logistic Regression': {}, 'Dummy baseline': {}},
                                          ['none', 'random_over'])
    assert not failures and len(results) == 4
    assert len({r['split_id'] for r in results}) == 1
    assert all(r['test_indices'] == results[0]['test_indices'] for r in results)
    assert all(r['train_indices'] == results[0]['train_indices'] for r in results)
    assert set(results[0]['test_indices']).isdisjoint(results[0]['train_indices'])
    assert all(len(r['cv_scores']) == 3 for r in results)
    for result in results:
        assert result['distribution']['test']['count'].sum() == 30
        pd.testing.assert_frame_equal(result['distribution']['test'], results[0]['distribution']['test'])
        learned = result['pipeline'].named_steps['preprocess'].named_transformers_['numeric'].named_steps['scale']
        expected_mean = numeric_data.loc[result['train_indices'], list('abcde')].mean().to_numpy()
        np.testing.assert_allclose(learned.mean_, expected_mean)
    assert results[2]['resampled_delta'] > 0


def test_each_cv_fold_fits_its_own_imputer_and_scaler(numeric_data, config, monkeypatch):
    from sklearn.impute import SimpleImputer
    observed = []
    original_fit = SimpleImputer.fit
    def capture(self, X, y=None):
        observed.append(np.asarray(X).copy())
        return original_fit(self, X, y)
    monkeypatch.setattr(SimpleImputer, 'fit', capture)
    results, failures = train_experiments(numeric_data, config, {'Logistic Regression': {}})
    assert not failures
    # Three disjoint-validation training folds (80 rows) plus the 120-row final fit.
    assert [len(value) for value in observed] == [80, 80, 80, 120]
    test_rows = numeric_data.loc[results[0]['test_indices'], list('abcde')].to_numpy()
    assert not any(np.any(np.all(rows[:, None, :] == test_rows[None, :, :], axis=2)) for rows in observed)


@pytest.mark.parametrize('name', [name for name in DEFAULTS if name != 'Multinomial NB'])
def test_classifiers(numeric_data, config, name):
    overrides = {'max_iter': 12, 'hidden_layer_sizes': [8]} if name == 'MLP' else {'n_estimators': 5} if name in ('Random Forest', 'Gradient Boosting') else {}
    config.split.folds = 2
    results, failures = train_experiments(numeric_data, config, {name: overrides})
    assert not failures, failures
    assert len(results) == 1 and np.isfinite(results[0]['evaluation']['metrics']['f1_macro'])


def test_multiclass():
    X, y = make_classification(n_samples=180, n_features=5, n_informative=4, n_redundant=0,
                               n_classes=3, weights=[0.6, 0.25, 0.15], random_state=2)
    frame = pd.DataFrame(X, columns=list('abcde'))
    frame['target'] = y.astype(str)
    config = ExperimentConfig('target', list('abcde'))
    config.split.folds = 3
    results, failures = train_experiments(frame, config, {'Random Forest': {'n_estimators': 5}})
    assert not failures
    assert results[0]['evaluation']['confusion_matrix'].shape == (3, 3)
    assert 'roc_auc_ovr_macro' in results[0]['evaluation']['metrics']
    assert results[0]['evaluation']['metrics']['balanced_accuracy'] == pytest.approx(results[0]['evaluation']['metrics']['recall_macro'])


def test_rare_class_and_fold_specific_smote_errors(numeric_data, config):
    config.split.folds = 10
    frame = numeric_data.copy()
    frame['target'] = ['1'] * 8 + ['0'] * (len(frame) - 8)
    with pytest.raises(ValueError, match='fewer observations'):
        train_experiments(frame, config, {'Dummy baseline': {}})
    config.split.folds = 3
    config.resampling.method = 'smote'
    config.resampling.k_neighbors = 5
    results, failures = train_experiments(frame, config, {'Logistic Regression': {}})
    assert not results and 'training-fold' in failures[0]['error']


def test_binary_threshold_uses_only_training_oof(numeric_data, config, monkeypatch):
    import core.training as training
    observed = []
    original = training.threshold_analysis
    def capture(y, scores, positive, kind):
        observed.extend(y.index.tolist())
        return original(y, scores, positive, kind)
    monkeypatch.setattr(training, 'threshold_analysis', capture)
    config.optimize_threshold = True
    config.positive_class = '0'  # Also exercise reversed positive-label score orientation.
    results, failures = train_experiments(numeric_data, config, {'Logistic Regression': {}})
    assert not failures
    result = results[0]
    assert set(observed) == set(result['train_indices'])
    assert set(observed).isdisjoint(result['test_indices'])
    assert result['threshold'] is not None and len(result['threshold_table']) == 101


def test_batch_prediction_reuses_pipeline(mixed_data):
    config = ExperimentConfig('target', ['a', 'category'])
    config.split.folds = 2
    results, failures = train_experiments(mixed_data, config, {'Logistic Regression': {}})
    assert not failures
    result = results[0]
    before = result['pipeline'].named_steps['preprocess'].named_transformers_['numeric'].named_steps['impute'].statistics_.copy()
    predicted = predict_batch(result, pd.DataFrame({'a': [10000., np.nan], 'category': ['unknown', 'red']}))
    assert len(predicted) == 2 and 'prediction' in predicted
    np.testing.assert_array_equal(before, result['pipeline'].named_steps['preprocess'].named_transformers_['numeric'].named_steps['impute'].statistics_)
    with pytest.raises(ValueError, match='Missing required'):
        predict_batch(result, pd.DataFrame({'bad': [1]}))
    with pytest.raises(ValueError, match='numerical'):
        predict_batch(result, pd.DataFrame({'a': ['bad'], 'category': ['red']}))


def test_metrics_and_no_fake_auc():
    actual = ['no', 'no', 'yes', 'yes']
    predicted = ['no', 'yes', 'yes', 'yes']
    values = evaluate_predictions(actual, predicted, ['no', 'yes'], positive='yes')
    assert values['metrics']['accuracy'] == 0.75
    assert values['metrics']['recall_positive'] == 1
    assert values['metrics']['specificity'] == 0.5
    assert 'roc_auc' not in values['metrics'] and 'roc_auc' in values['unavailable']
    assert values['confusion_matrix'].tolist() == [[1, 1], [0, 2]]


def test_reserved_class_label_names():
    result = evaluate_predictions(['accuracy', 'macro avg'], ['accuracy', 'accuracy'], ['accuracy', 'macro avg'])
    assert result['per_class'].shape[0] == 2


def test_highly_imbalanced_repeated_cv_and_class_weights():
    X, y = make_classification(n_samples=400, n_features=5, n_informative=3, n_redundant=0,
                               weights=[0.975, 0.025], flip_y=0, random_state=42)
    frame = pd.DataFrame(X, columns=list('abcde'))
    frame['target'] = y.astype(str)
    config = ExperimentConfig('target', list('abcde'))
    config.split.folds = 2
    config.split.repeats = 2
    config.resampling.k_neighbors = 1
    results, failures = train_experiments(frame, config,
        {'Logistic Regression': {'class_weight': 'balanced'}}, ['none', 'smote'])
    assert not failures, failures
    assert len(results) == 2 and all(len(r['cv_scores']) == 4 for r in results)
    assert results[0]['distribution']['dataset']['count'].max() / results[0]['distribution']['dataset']['count'].min() == 39
    pd.testing.assert_frame_equal(results[0]['distribution']['test'], results[1]['distribution']['test'])


def test_text_training_and_batch_prediction(text_data):
    config = ExperimentConfig('target', ['text'], text_column='text')
    config.preprocessing.task = 'Text'
    config.split.folds = 2
    results, failures = train_experiments(text_data, config, {'Multinomial NB': {}, 'Linear SVM': {}}, ['none', 'random_over'])
    assert not failures, failures
    assert len({r['split_id'] for r in results}) == 1
    for result in results:
        output = predict_batch(result, pd.DataFrame({'text': ['a wonderful happy review', None]}))
        assert len(output) == 2 and output.prediction.notna().all()


def test_primary_auc_uses_continuous_scores():
    actual = ['no', 'no', 'yes', 'yes']
    predicted = ['no', 'no', 'no', 'no']
    response = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]])
    result = evaluate_predictions(actual, predicted, ['no', 'yes'], response, 'probability', 'yes')
    assert result['metrics']['accuracy'] == 0.5
    assert result['metrics']['roc_auc'] == 1.0
    assert result['metrics']['average_precision'] == 1.0

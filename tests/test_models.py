import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, roc_auc_score
from default_studio.data import FEATURES, TARGET, NUMERICAL
from default_studio.modeling import comparison_table, predict_record, split_indices


def test_all_records_used_in_fixed_shared_split(dataset, benchmark):
    train, test = benchmark['train_indices'], benchmark['test_indices']
    assert len(train) == 24000 and len(test) == 6000
    assert not set(train) & set(test)
    assert set(train) | set(test) == set(range(30000))
    repeat_train, repeat_test = split_indices(dataset[TARGET])
    np.testing.assert_array_equal(train, repeat_train)
    np.testing.assert_array_equal(test, repeat_test)
    assert set(benchmark['models']) == {'ANN', 'Random Forest'}
    assert all(len(result['predictions']) == 6000 for result in benchmark['models'].values())


def test_ann_validation_disjoint_and_preprocessing_train_only(dataset, benchmark):
    fit, stop, test = benchmark['ann_fit_indices'], benchmark['ann_stop_indices'], benchmark['test_indices']
    assert len(fit) == 21600 and len(stop) == 2400
    assert not set(fit) & set(stop) and not set(fit) & set(test) and not set(stop) & set(test)
    assert set(fit) | set(stop) == set(benchmark['train_indices'])
    numeric = benchmark['models']['ANN']['pipeline'].named_steps['prepare'].named_steps['columns'].named_transformers_['numeric']
    np.testing.assert_allclose(numeric.named_steps['impute'].statistics_, dataset.iloc[fit][NUMERICAL].median())
    np.testing.assert_allclose(numeric.named_steps['scale'].mean_, dataset.iloc[fit][NUMERICAL].mean())
    forest_numeric = benchmark['models']['Random Forest']['pipeline'].named_steps['prepare'].named_steps['columns'].named_transformers_['numeric']
    np.testing.assert_allclose(forest_numeric.named_steps['impute'].statistics_, dataset.iloc[benchmark['train_indices']][NUMERICAL].median())


def test_identifier_and_target_never_enter_model(dataset, benchmark):
    for result in benchmark['models'].values():
        preparation = result['pipeline'].named_steps['prepare']
        assert list(preparation.feature_names_in_) == FEATURES
        assert 'ID' not in preparation.feature_names_in_ and TARGET not in preparation.feature_names_in_
    forest = benchmark['models']['Random Forest']['pipeline'].named_steps['model']
    assert forest.n_estimators == 80 and forest.max_depth == 8 and forest.n_jobs == 1


def test_measured_metrics_and_ann_checkpoint(benchmark):
    table = comparison_table(benchmark)
    assert len(table) == 2 and np.isfinite(table.drop(columns='Model').to_numpy()).all()
    for result in benchmark['models'].values():
        assert result['metrics']['F1 (default)'] == pytest.approx(f1_score(benchmark['test_labels'], result['predictions']))
        assert result['metrics']['ROC-AUC'] == pytest.approx(roc_auc_score(benchmark['test_labels'], result['probabilities']))
        assert result['confusion_matrix'].sum() == 6000
        assert result['fit_seconds'] > 0
    assert 1 <= benchmark['ann_best_epoch'] <= len(benchmark['ann_history']) <= 80
    best = benchmark['ann_history'].set_index('epoch').loc[benchmark['ann_best_epoch'], 'validation_log_loss']
    assert best <= benchmark['ann_history'].validation_log_loss.min() + 0.0001


def test_prediction_reuses_fitted_preprocessing(dataset, benchmark):
    pipeline = benchmark['models']['ANN']['pipeline']
    numeric = pipeline.named_steps['prepare'].named_steps['columns'].named_transformers_['numeric']
    before = numeric.named_steps['scale'].mean_.copy()
    record = dataset.iloc[[0]].copy()
    record.loc[0, 'PAY_0'] = 9  # Safe unknown category from the documented prediction range.
    for name in ('ANN', 'Random Forest'):
        predicted = predict_record(benchmark, name, record)
        assert 0 <= predicted['probability'] <= 1 and predicted['label'] in (0, 1)
    np.testing.assert_array_equal(before, numeric.named_steps['scale'].mean_)
    with pytest.raises(ValueError, match='Choose'):
        predict_record(benchmark, 'SVM', record)

"""Synthetic credit-card tests for full-data Colab training and compact import."""
from dataclasses import replace
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from examples.creditcard_colab import FraudColabConfig, run_fraud_experiment, threshold_on_validation
from core.artifacts import import_results, matrix_evaluation
from core.fraud import creditcard_preset, fraud_outcomes

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fraud_csv(tmp_path):
    rng = np.random.default_rng(42)
    labels = (np.arange(600) % 25 == 0).astype(int)
    frame = pd.DataFrame({'Time': np.arange(600), 'Amount': rng.uniform(0, 200, 600),
                          'V1': labels * 5 + rng.normal(0, 0.5, 600), 'V2': rng.normal(size=600), 'Class': labels})
    frame.loc[4, 'V2'] = np.nan
    path = tmp_path / 'creditcard.csv'
    frame.to_csv(path, index=False)
    return frame, path


def settings(path, tmp_path, **kwargs):
    return FraudColabConfig(csv_path=str(path), output_dir=str(tmp_path / 'run'),
        models=('SGD logistic',), chunk_size=73, min_precision=0.0, rf_estimators=5,
        mlp_iterations=2, **kwargs)


def test_colab_batch_full_data_validation_and_import(fraud_csv, tmp_path, monkeypatch):
    import examples.creditcard_colab as trainer
    frame, path = fraud_csv
    observed = []
    original = trainer._response
    def capture(pipeline, X):
        observed.append(X[:, 0].copy())
        return original(pipeline, X)
    monkeypatch.setattr(trainer, '_response', capture)
    config = settings(path, tmp_path)
    result = run_fraud_experiment(config)
    assert len(observed) == 2  # Validation for selection; test once after threshold is frozen.
    assert min(observed[0]) == 360 and max(observed[0]) == 479
    assert min(observed[1]) == 480 and max(observed[1]) == 599
    bundle = joblib.load(Path(config.output_dir) / 'trusted_fraud_model.joblib')
    imputer = bundle['pipeline'].named_steps['impute']
    np.testing.assert_allclose(imputer.statistics_, frame.drop(columns='Class').iloc[:360].median(), rtol=1e-5)
    assert result['colab_metadata']['total_rows'] == 600
    assert result['colab_metadata']['training_rows_used'] == 360
    content = (Path(config.output_dir) / 'fraud_results.json').read_bytes()
    assert len(content) < 100000
    imported = import_results(content, 'fraud_results.json')[0]
    assert imported['positive_class'] == '1' and imported['pipeline'] is None
    assert imported['validation_comparison'][0]['model'] == 'SGD logistic'
    assert imported['evaluation']['metrics']['average_precision'] == result['metrics']['average_precision']
    assert 'average_precision' not in imported['evaluation']['unavailable']
    assert imported['evaluation']['curves']['1']['fpr'].shape[0] <= 400
    assert sum(fraud_outcomes(imported).values()) == 120
    with pytest.raises(FileExistsError):
        run_fraud_experiment(config)


@pytest.mark.parametrize('chunk_size', [17, 100])
def test_incremental_uses_every_training_row_and_frozen_scaler(fraud_csv, tmp_path, chunk_size):
    frame, path = fraud_csv
    config = replace(settings(path, tmp_path, mode='incremental', epochs=2), chunk_size=chunk_size)
    result = run_fraud_experiment(config)
    bundle = joblib.load(Path(config.output_dir) / 'trusted_fraud_model.joblib')
    scaler = bundle['pipeline'].named_steps['scale']
    assert scaler.n_samples_seen_ == 360
    np.testing.assert_allclose(scaler.mean_, frame.drop(columns='Class').iloc[:360].fillna(0).mean(), rtol=1e-5)
    assert result['colab_metadata']['training_rows_used'] == 360
    assert result['colab_metadata']['epochs'] == 2
    assert result['model_name'] == 'SGD logistic (incremental)'
    assert sum(result['colab_metadata']['partition_counts']['test']) == 120
    assert import_results((Path(config.output_dir) / 'fraud_results.json').read_bytes(), 'result.json')


def test_batch_ann_random_forest_stratified(fraud_csv, tmp_path):
    _, path = fraud_csv
    config = replace(settings(path, tmp_path), split_method='stratified',
                     models=('MLP', 'Random Forest', 'Dummy baseline'))
    result = run_fraud_experiment(config)
    assert {row['model'] for row in result['validation_comparison']} == {'MLP', 'Random Forest', 'Dummy baseline'}
    assert result['colab_metadata']['total_rows'] == 600
    assert len(result['confusion_matrix']) == 2


def test_incremental_rejects_unsorted_time(fraud_csv, tmp_path):
    frame, path = fraud_csv
    frame.loc[100, 'Time'] = -1
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='sorted'):
        run_fraud_experiment(settings(path, tmp_path, mode='incremental'))


def test_equal_timestamps_not_shared_across_partitions(fraud_csv, tmp_path):
    frame, path = fraud_csv
    frame['Time'] = np.arange(len(frame)) // 7
    frame.to_csv(path, index=False)
    result = run_fraud_experiment(settings(path, tmp_path, mode='incremental'))
    train, validation, test = [sum(result['colab_metadata']['partition_counts'][p]) for p in ('training', 'validation', 'test')]
    assert train % 7 == 0 and validation % 7 == 0 and train + validation + test == len(frame)


def test_batch_memory_guard_and_invalid_target(fraud_csv, tmp_path):
    frame, path = fraud_csv
    large = pd.concat([frame] * 40, ignore_index=True)
    large.to_csv(path, index=False)
    with pytest.raises(MemoryError, match='incremental'):
        run_fraud_experiment(settings(path, tmp_path, memory_budget_mb=1))
    frame['Class'] = 3
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='only 0'):
        run_fraud_experiment(settings(path, tmp_path))


def test_threshold_precision_constraint_and_known_fraud_metrics():
    y = np.array([0, 0, 1, 1])
    threshold, selected = threshold_on_validation(y, np.array([0.1, 0.2, 0.8, 0.9]), min_precision=0.9)
    assert threshold == 0.8 and selected['recall'] == 1.0
    with pytest.raises(ValueError, match='No validation threshold'):
        threshold_on_validation(y, np.full(4, 0.5), min_precision=0.9)
    evaluated = matrix_evaluation([[900000000, 1000000], [20, 80]], ['0', '1'], positive='1')
    assert evaluated['metrics']['recall_positive'] == 0.8
    assert evaluated['confusion_matrix'].sum() == 901000100


def test_notebook_embeds_tested_module():
    notebook = json.loads((ROOT / 'examples/credit_card_fraud_colab.ipynb').read_text())
    sources = [''.join(cell['source']) for cell in notebook['cells']]
    assert (ROOT / 'examples/creditcard_colab.py').read_text() in sources
    for cell in notebook['cells']:
        source = ''.join(cell['source'])
        if cell['cell_type'] == 'code' and not source.startswith('%pip'):
            compile(source, '<notebook-cell>', 'exec')


def test_fraud_preset_and_ui(fraud_csv):
    frame, _ = fraud_csv
    preset = creditcard_preset(frame)
    assert preset['positive_class'] == '1'
    assert preset['primary_metric'] == 'average_precision'
    assert 'Class' not in preset['feature_selection']
    app = AppTest.from_file(ROOT / 'app.py').run(timeout=30)
    app.radio[0].set_value('Credit Card Fraud & Colab').run()
    assert not app.exception and len(app.download_button) == 2
    app.radio[0].set_value('Dataset Management').run()
    app.file_uploader[0].set_value(('creditcard.csv', frame.to_csv(index=False).encode(), 'text/csv')).run()
    next(b for b in app.button if b.label == 'Load this dataset').click().run()
    next(b for b in app.button if b.label == 'Apply credit-card fraud preset').click().run()
    assert not app.exception
    assert app.session_state['positive_class'] == '1'
    assert app.session_state['target'] == 'Class'
    assert app.session_state['primary_metric'] == 'average_precision'


def test_fraud_result_ui_without_full_dataset(fraud_csv, tmp_path):
    _, path = fraud_csv
    config = settings(path, tmp_path, export_model=False)
    run_fraud_experiment(config)
    app = AppTest.from_file(ROOT / 'app.py').run()
    app.radio[0].set_value('Import Colab Results').run()
    app.file_uploader[0].set_value(('fraud_results.json', (Path(config.output_dir) / 'fraud_results.json').read_bytes(), 'application/json')).run()
    next(b for b in app.button if b.label == 'Validate and import').click().run()
    assert not app.exception and not app.error
    assert app.session_state['dataset'] is None
    for page in ['Dashboard', 'Credit Card Fraud & Colab', 'Performance Evaluation', 'Export and Reports']:
        app.radio[0].set_value(page).run(timeout=30)
        assert not app.exception, (page, app.exception)
        if page == 'Credit Card Fraud & Colab':
            assert any(metric.label == 'Fraud missed' for metric in app.metric)

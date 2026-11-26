from io import BytesIO
import json
import pandas as pd
import pytest
from core.artifacts import (import_results, export_experiments, csv_bytes, excel_bytes, pipeline_bytes)
from core.config import ExperimentConfig
from core.training import train_experiments
from core.training import comparison_table
from utils.reports import scientific_report


def test_export_import_roundtrip(numeric_data):
    config = ExperimentConfig('target', list('abcde'))
    config.split.folds = 2
    results, failures = train_experiments(numeric_data, config, {'Dummy baseline': {}})
    assert not failures
    imported = import_results(export_experiments(results).encode(), 'results.json')
    assert len(imported) == 1 and imported[0]['pipeline'] is None
    assert imported[0]['evaluation']['metrics']['accuracy'] == results[0]['evaluation']['metrics']['accuracy']
    assert imported[0]['split_id'] != results[0]['split_id']
    assert pipeline_bytes(results[0])
    report = scientific_report(results[0], comparison_table(results))
    assert '## Model parameters' in report and '## Model comparison' in report
    assert 'Dummy baseline' in report and 'non-equivalent' in report


def test_csv_metric_and_prediction_import():
    assert len(import_results(b'model,accuracy,f1_macro\nForest,0.8,0.5\n', 'metrics.csv')) == 1
    result = import_results(b'actual,predicted\nno,no\nyes,no\n', 'predictions.csv', 'Predictions')[0]
    assert result['evaluation']['metrics']['accuracy'] == 0.5
    assert 'roc_auc' not in result['evaluation']['metrics']


@pytest.mark.parametrize('content,name', [
    (b'model,accuracy\nForest,80', 'metrics.csv'),
    (b'{}', 'bad.json'), (b'[]', 'bad.json'),
    (b'{"classes":["a","b"],"confusion_matrix":[[1,-1],[0,2]]}', 'bad.json'),
    (b'{"classes":["a","b"],"confusion_matrix":[[1,0],[0,2]],"metrics":{"accuracy":0.5}}', 'bad.json'),
    (b'malicious', 'model.joblib'), (b'malicious', 'notebook.ipynb'),
])
def test_unsafe_or_invalid_imports(content, name):
    with pytest.raises(ValueError):
        import_results(content, name)


def test_excel_and_csv_formula_safety():
    frame = pd.DataFrame({'text': ['=CMD()', '@SUM(1)', 'ordinary'], 'value': [1, 2, 3]})
    output = pd.read_excel(BytesIO(excel_bytes({'sheet': frame})))
    assert output.text.iloc[0] == "'=CMD()"
    assert b"'=CMD()" in csv_bytes(frame)


def test_import_rejects_malformed_configuration():
    with pytest.raises(ValueError, match='config.preprocessing'):
        import_results(b'{"model_name":"bad","metrics":{"accuracy":0.5},"config":{"preprocessing":"invalid"}}', 'bad.json')

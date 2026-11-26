import pandas as pd
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
from core.config import ExperimentConfig, PreprocessingConfig
from core.data import dataset_fingerprint
from core.training import train_experiments

PAGES = ['Dashboard', 'Dataset Management', 'Exploratory Data Analysis', 'Data Preprocessing',
         'Class Imbalance Analysis', 'Model Configuration', 'Model Training', 'Performance Evaluation',
         'Model Comparison', 'Error Analysis', 'Prediction', 'Experiment History', 'Import Colab Results',
         'Export and Reports', 'Help and Documentation']
APP = Path(__file__).resolve().parents[1] / 'app.py'


def test_empty_app_and_navigation():
    app = AppTest.from_file(APP).run(timeout=30)
    assert not app.exception
    assert app.header[0].value == 'ML Analytics Studio'
    for page in PAGES:
        app.radio[0].set_value(page).run(timeout=30)
        assert not app.exception, (page, app.exception)


def test_all_pages_with_trained_result(numeric_data):
    config = ExperimentConfig('target', list('abcde'))
    config.split.folds = 2
    results, failures = train_experiments(numeric_data, config, {'Logistic Regression': {}})
    assert not failures
    app = AppTest.from_file(APP)
    app.session_state['dataset'] = numeric_data
    app.session_state['dataset_name'] = 'synthetic.csv'
    app.session_state['dataset_id'] = dataset_fingerprint(numeric_data)
    app.session_state['target'] = 'target'
    app.session_state['feature_selection'] = list('abcde')
    app.session_state['history'] = results
    app.session_state['active_ids'] = [results[0]['id']]
    app.run(timeout=30)
    for page in PAGES:
        app.radio[0].set_value(page).run(timeout=30)
        assert not app.exception, (page, app.exception)


def test_dataset_and_configuration_invalidate_active_results(numeric_data):
    from utils.session import replace_dataset, invalidate_config
    # Run within AppTest to exercise real Streamlit session state across reruns.
    app = AppTest.from_string('''
import streamlit as st
import pandas as pd
from utils.session import initialize, replace_dataset, invalidate_config
initialize()
st.session_state.active_ids = ['old']
st.session_state.current_config_id = 'first'
invalidate_config('second')
assert not st.session_state.active_ids
st.session_state.history = [{'id':'historical'}]
replace_dataset(pd.DataFrame({'x':[1,2], 'target':['a','b']}), 'replacement.csv')
assert not st.session_state.active_ids
assert st.session_state.target is None
assert len(st.session_state.history) == 1
st.success('Invalidation passed')
''').run()
    assert not app.exception and app.success[0].value == 'Invalidation passed'


def widget(elements, label):
    return next(element for element in elements if element.label == label)


def test_upload_train_predict_export_journey(numeric_data):
    app = AppTest.from_file(APP).run(timeout=30)
    app.radio[0].set_value('Dataset Management').run()
    app.file_uploader[0].set_value(('synthetic.csv', numeric_data.to_csv(index=False).encode(), 'text/csv')).run()
    widget(app.button, 'Load this dataset').click().run()
    assert not app.exception
    widget(app.button, 'Apply dataset selection').click().run()
    assert app.session_state['target'] == 'target'
    assert not app.session_state['history']
    app.radio[0].set_value('Model Configuration').run()
    widget(app.multiselect, 'Classifiers').set_value(['Dummy baseline', 'Logistic Regression'])
    widget(app.button, 'Save model configuration').click().run()
    app.radio[0].set_value('Class Imbalance Analysis').run()
    widget(app.selectbox, 'Training-only resampling').set_value('random_over')
    widget(app.button, 'Save imbalance handling').click().run()
    app.radio[0].set_value('Model Training').run()
    widget(app.selectbox, 'Experiment mode').set_value('C · Compare original and resampled')
    widget(app.number_input, 'Stratified CV folds').set_value(2)
    widget(app.button, 'Train selected models').click().run(timeout=60)
    assert not app.exception and not app.error, app.error
    assert len(app.session_state['history']) == 4
    assert len(app.session_state['active_ids']) == 4
    for destination in ['Model Comparison', 'Performance Evaluation', 'Error Analysis', 'Export and Reports']:
        app.radio[0].set_value(destination).run(timeout=30)
        assert not app.exception, (destination, app.exception)
    assert app.download_button
    app.radio[0].set_value('Prediction').run()
    widget(app.button, 'Predict record').click().run()
    assert not app.exception and not app.error
    app.file_uploader[0].set_value(('batch.csv', numeric_data.head(3).to_csv(index=False).encode(), 'text/csv')).run()
    widget(app.button, 'Predict batch').click().run()
    assert not app.exception
    assert len(app.session_state['last_predictions']) == 3
    app.radio[0].set_value('Data Preprocessing').run()
    widget(app.selectbox, 'Numerical scaling').set_value('robust')
    widget(app.button, 'Save preprocessing').click().run()
    assert not app.session_state['active_ids']
    assert len(app.session_state['history']) == 4


def test_imported_results_pages():
    app = AppTest.from_file(APP).run()
    app.radio[0].set_value('Import Colab Results').run()
    app.file_uploader[0].set_value(('metrics.csv', b'model,accuracy,f1_macro\nColab,0.8,0.7\n', 'text/csv')).run()
    widget(app.button, 'Validate and import').click().run()
    assert not app.exception and len(app.session_state['history']) == 1
    for destination in ['Performance Evaluation', 'Model Comparison', 'Error Analysis', 'Export and Reports', 'Experiment History']:
        app.radio[0].set_value(destination).run(timeout=30)
        assert not app.exception, (destination, app.exception)


def test_text_pages_and_multiclass_smotenc_pages(text_data, mixed_data):
    for frame, task, features, sampler, models in [
        (text_data, 'Text', ['text'], 'none', {'Multinomial NB': {}}),
        (mixed_data, 'Mixed', ['a', 'category'], 'smotenc', {'Random Forest': {'n_estimators': 5}}),
    ]:
        config = ExperimentConfig('target', features, text_column='text' if task == 'Text' else None)
        config.preprocessing.task = task
        config.resampling.method = sampler
        config.resampling.k_neighbors = 2
        config.split.folds = 2
        results, failures = train_experiments(frame, config, models)
        assert not failures
        app = AppTest.from_file(APP)
        for key, value in {'dataset': frame, 'dataset_name': 'test.csv', 'dataset_id': dataset_fingerprint(frame),
                           'target': 'target', 'feature_selection': features, 'history': results,
                           'prep_config': config.preprocessing, 'resample_config': config.resampling,
                           'text_column': config.text_column, 'active_ids': [results[0]['id']]}.items():
            app.session_state[key] = value
        app.run(timeout=30)
        for page in ['Exploratory Data Analysis', 'Data Preprocessing', 'Class Imbalance Analysis',
                     'Performance Evaluation', 'Prediction', 'Export and Reports']:
            app.radio[0].set_value(page).run(timeout=30)
            assert not app.exception, (task, page, app.exception)

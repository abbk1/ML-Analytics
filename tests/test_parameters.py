from dataclasses import replace
import json
import numpy as np
import pytest
from sklearn.metrics import log_loss
from streamlit.testing.v1 import AppTest
from default_studio.data import metadata, FEATURES, TARGET
from default_studio.parameters import TrainingParameters
from default_studio.modeling import train_models
from tests_ui_helpers import APP, widget, configured_app


@pytest.mark.parametrize('changes', [
    {'forest_trees': 500}, {'forest_depth': 20}, {'forest_min_leaf': 0},
    {'forest_min_split': 1}, {'ann_neurons': 200}, {'ann_second_neurons': 4},
    {'ann_epochs': 1000}, {'ann_patience': 0}, {'ann_learning_rate': float('nan')},
    {'ann_batch_size': 10000}, {'ann_alpha': -1}, {'ann_activation': 'unsupported'},
    {'forest_max_features': 'all'}, {'balance_classes': 'yes'}, {'forest_trees': True},
    {'ann_batch_size': 256.0},
])
def test_backend_rejects_unsafe_or_unsupported_parameters(changes):
    with pytest.raises(ValueError):
        replace(TrainingParameters(), **changes).validate()


def test_custom_parameters_reach_both_fitted_models(dataset):
    parameters = TrainingParameters(forest_trees=20, forest_depth=3, forest_min_leaf=2,
        forest_min_split=4, forest_max_features='log2', ann_neurons=16, ann_second_neurons=8,
        ann_activation='tanh', ann_learning_rate=0.002, ann_batch_size=64,
        ann_epochs=10, ann_patience=3, ann_alpha=0.05, balance_classes=False)
    events = []
    frame = dataset.iloc[:1500]
    benchmark = train_models(frame, lambda name, text: events.append((name, text)), parameters)
    assert benchmark['parameters'] == parameters.to_dict()
    assert benchmark['parameter_fingerprint'] == parameters.fingerprint()
    snapshot = json.loads(json.dumps({'requested': benchmark['parameters'], 'effective': benchmark['effective_parameters']}))
    assert snapshot['requested']['forest_trees'] == 20
    assert snapshot['effective']['ANN']['hidden_layer_sizes'] == [16, 8]
    forest = benchmark['models']['Random Forest']['pipeline'].named_steps['model']
    assert forest.n_estimators == 20 and forest.max_depth == 3
    assert forest.min_samples_leaf == 2 and forest.min_samples_split == 4
    assert forest.max_features == 'log2' and forest.class_weight is None
    assert forest.n_jobs == 1 and forest.max_leaf_nodes == 512
    ann = benchmark['models']['ANN']['pipeline'].named_steps['model']
    assert ann.hidden_layer_sizes == (16, 8) and ann.activation == 'tanh'
    assert ann.learning_rate_init == 0.002 and ann.alpha == 0.05 and ann.batch_size == 64
    assert len(benchmark['ann_history']) <= 10
    assert '20 trees' in events[0][1]
    for name in ('ANN', 'Random Forest'):
        assert benchmark['effective_parameters'][name] == benchmark['models'][name]['pipeline'].named_steps['model'].get_params()
    validation = frame.iloc[benchmark['ann_stop_indices']]
    predictions = benchmark['models']['ANN']['pipeline'].predict_proba(validation[FEATURES])
    loss = log_loss(validation[TARGET], predictions, labels=[0, 1])
    stored = benchmark['ann_history'].set_index('epoch').loc[benchmark['ann_best_epoch'], 'validation_log_loss']
    assert loss == pytest.approx(stored)
    repeat = train_models(frame, parameters=replace(parameters, forest_depth=4))
    np.testing.assert_array_equal(benchmark['test_indices'], repeat['test_indices'])


def test_saving_parameters_invalidates_results_without_training(monkeypatch, benchmark):
    from default_studio import ui
    def forbidden(*args, **kwargs):
        raise AssertionError('Saving controls must not train a model.')
    monkeypatch.setattr(ui, 'train_models', forbidden)
    app = configured_app(benchmark)
    app.session_state['prediction'] = {'results': {'old': True}}
    app.radio[0].set_value('Train').run()
    widget(app.button, 'Save training parameters').click().run(timeout=30)
    assert app.session_state['benchmark'] is not None
    assert 'prediction' in app.session_state
    widget(app.number_input, 'Number of trees').set_value(40)
    widget(app.number_input, 'Maximum epochs').set_value(30)
    widget(app.button, 'Save training parameters').click().run(timeout=30)
    assert not app.exception
    assert app.session_state['benchmark'] is None
    assert 'prediction' not in app.session_state
    assert app.session_state['training_parameters']['forest_trees'] == 40
    assert app.session_state['training_parameters']['ann_epochs'] == 30
    app.radio[0].set_value('Compare').run()
    assert not app.exception
    assert not app.download_button
    app.radio[0].set_value('Train').run()
    assert widget(app.number_input, 'Number of trees').value == 40
    assert widget(app.number_input, 'Maximum epochs').value == 30
    widget(app.button, 'Reset parameters to defaults').click().run()
    assert app.session_state['training_parameters'] == TrainingParameters().to_dict()
    assert widget(app.number_input, 'Number of trees').value == 80


def test_train_button_uses_unsaved_form_settings(monkeypatch, benchmark):
    from default_studio import ui
    calls = []
    def capture(frame, progress, parameters):
        calls.append(parameters)
        # Only UI wiring is stubbed; the separate engine test fits non-default models.
        return {**benchmark, 'parameters': parameters.to_dict(),
                'parameter_fingerprint': parameters.fingerprint()}
    monkeypatch.setattr(ui, 'train_models', capture)
    app = AppTest.from_file(APP).run(timeout=30)
    app.radio[0].set_value('Train').run()
    widget(app.number_input, 'Maximum tree depth').set_value(5)
    widget(app.number_input, 'Second hidden layer neurons (0 = disabled)').set_value(16)
    widget(app.selectbox, 'Activation').set_value('tanh')
    widget(app.checkbox, 'Balance default and no-default classes').uncheck()
    widget(app.button, 'Train ANN and Random Forest').click().run(timeout=30)
    assert not app.exception and len(calls) == 1
    assert calls[0].forest_depth == 5 and calls[0].ann_second_neurons == 16
    assert calls[0].ann_activation == 'tanh' and calls[0].balance_classes is False
    app.radio[0].set_value('Predict').run()
    assert not app.exception and len(calls) == 1

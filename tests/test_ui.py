from streamlit.testing.v1 import AppTest

from tests_ui_helpers import APP, widget, configured_app

PAGES = ['Dataset', 'Train', 'Compare', 'Predict']


def test_four_pages_and_no_upload_or_training_on_start():
    app = AppTest.from_file(APP).run(timeout=30)
    assert not app.exception
    assert list(app.radio[0].options) == PAGES
    assert app.session_state['benchmark'] is None
    assert not app.file_uploader
    assert any(item.value == '30,000' for item in app.metric)
    for page in PAGES:
        app.radio[0].set_value(page).run(timeout=30)
        assert not app.exception
        assert app.session_state['benchmark'] is None


def test_ready_models_render_every_page(benchmark):
    app = configured_app(benchmark)
    for page in PAGES:
        app.radio[0].set_value(page).run(timeout=30)
        assert not app.exception, (page, app.exception)
    widget(app.button, 'Predict with both models').click().run(timeout=30)
    assert not app.exception and not app.error
    assert set(app.session_state['prediction']['results']) == {'ANN', 'Random Forest'}
    assert len(app.metric) == 2
    app.radio[0].set_value('Compare').run(timeout=30)
    assert len(app.download_button) == 2


def test_training_explicit_and_session_results_reused(monkeypatch, benchmark):
    from default_studio import ui
    calls = []
    def train(frame, progress, parameters):
        calls.append(len(frame))
        assert parameters.to_dict() == benchmark['parameters']
        progress('ANN', 'Test callback')
        return benchmark
    monkeypatch.setattr(ui, 'train_models', train)
    app = AppTest.from_file(APP).run(timeout=30)
    app.radio[0].set_value('Train').run()
    assert not calls
    widget(app.button, 'Train ANN and Random Forest').click().run(timeout=30)
    assert not app.exception
    assert calls == [30000]
    for page in ['Compare', 'Predict', 'Dataset', 'Train']:
        app.radio[0].set_value(page).run(timeout=30)
        assert not app.exception
    assert calls == [30000]
    widget(app.button, 'Clear this session').click().run()
    assert app.session_state['benchmark'] is None
    assert 'prediction' not in app.session_state

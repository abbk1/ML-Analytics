from pathlib import Path
from streamlit.testing.v1 import AppTest
from default_studio.data import metadata

APP = Path(__file__).resolve().parents[1] / 'app.py'


def widget(elements, label):
    return next(item for item in elements if item.label == label)


def configured_app(benchmark):
    app = AppTest.from_file(APP)
    app.session_state['studio_dataset_version'] = metadata()['bundled_sha256']
    app.session_state['benchmark'] = benchmark
    app.session_state['studio_page'] = 'Dataset'
    return app.run(timeout=30)

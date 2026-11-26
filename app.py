"""ML Analytics Studio — Streamlit Community Cloud entry point."""
import hashlib
import json
import streamlit as st
from core.config import ExperimentConfig, PreprocessingConfig, ResamplingConfig, SplitConfig
from core.models import DEFAULTS
from utils.session import initialize, invalidate_config, reset_session
from views.data_pages import dataset_page, eda_page, preprocessing_page, imbalance_page
from views.model_pages import (dashboard, model_configuration, training_page, evaluation_page,
                               comparison_page, error_page, prediction_page)
from views.exchange_pages import history_page, import_page, export_page, help_page
from views.fraud_page import fraud_page

PAGES = ['Dashboard', 'Credit Card Fraud & Colab', 'Dataset Management', 'Exploratory Data Analysis', 'Data Preprocessing',
         'Class Imbalance Analysis', 'Model Configuration', 'Model Training', 'Performance Evaluation',
         'Model Comparison', 'Error Analysis', 'Prediction', 'Experiment History', 'Import Colab Results',
         'Export and Reports', 'Help and Documentation']


def current_config():
    state = st.session_state
    if state.dataset is None or not state.target or not state.feature_selection:
        return None
    return ExperimentConfig(target=state.target, features=state.feature_selection,
        text_column=state.text_column, name=state.experiment_name, preprocessing=state.prep_config,
        split=state.split_config, resampling=state.resample_config, primary_metric=state.primary_metric,
        positive_class=state.positive_class, optimize_threshold=state.optimize_threshold)


def main():
    st.set_page_config(page_title='Credit Card Fraud · ML Analytics Studio', page_icon='📊', layout='wide', initial_sidebar_state='expanded')
    initialize()
    defaults = {'prep_config': PreprocessingConfig(), 'resample_config': ResamplingConfig(),
                'split_config': SplitConfig(), 'text_column': None, 'primary_metric': 'f1_macro',
                'positive_class': None, 'optimize_threshold': False, 'experiment_name': 'Experiment',
                'model_settings': {name: dict(DEFAULTS[name]) for name in ['Dummy baseline', 'Random Forest', 'Logistic Regression']}}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    with st.sidebar:
        st.title('Credit Card Fraud Studio')
        st.caption('FRAUD DETECTION & COLAB WORKSPACE')
        st.selectbox('Configuration mode', ['Beginner', 'Advanced'], key='ui_mode')
        page = st.radio('Navigation', PAGES, key='navigation', label_visibility='collapsed')
        st.divider()
        st.caption(st.session_state.dataset_name or 'No dataset loaded')
        st.caption(f"{len(st.session_state.history)} session experiments · CPU only")
        if st.button('Clear entire session'):
            reset_session()
            st.rerun()
    config = current_config()
    if config:
        fingerprint = hashlib.sha256((config.fingerprint() + json.dumps(st.session_state.model_settings, sort_keys=True)).encode()).hexdigest()
        invalidate_config(fingerprint)
    routes = {
        'Dashboard': lambda: dashboard(config), 'Credit Card Fraud & Colab': fraud_page, 'Dataset Management': dataset_page,
        'Exploratory Data Analysis': eda_page, 'Data Preprocessing': lambda: preprocessing_page(config),
        'Class Imbalance Analysis': imbalance_page, 'Model Configuration': model_configuration,
        'Model Training': lambda: training_page(config), 'Performance Evaluation': evaluation_page,
        'Model Comparison': comparison_page, 'Error Analysis': error_page, 'Prediction': prediction_page,
        'Experiment History': history_page, 'Import Colab Results': import_page,
        'Export and Reports': export_page, 'Help and Documentation': help_page,
    }
    routes[page]()
    st.divider()
    st.caption('ML Analytics Studio · Training-only preprocessing · Untouched holdout evaluation · Reproducible session experiments')


if __name__ == '__main__':
    main()

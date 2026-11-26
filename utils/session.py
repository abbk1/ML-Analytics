"""Explicit invalidation while keeping previous experiments as immutable history."""
import streamlit as st
from core.data import dataset_fingerprint


def initialize():
    defaults = {'dataset': None, 'dataset_name': None, 'dataset_id': None, 'history': [],
                'active_ids': [], 'current_config_id': None, 'target': None, 'feature_selection': [],
                'navigation': 'Dashboard', 'trained_config_id': None, 'dataset_generation': 0}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def replace_dataset(frame, name):
    st.session_state.update(dataset=frame, dataset_name=name, dataset_id=dataset_fingerprint(frame),
                            active_ids=[], current_config_id=None, trained_config_id=None,
                            target=None, feature_selection=[], dataset_generation=st.session_state.dataset_generation + 1)


def invalidate_config(config_id):
    previous = st.session_state.current_config_id
    if previous is not None and previous != config_id:
        st.session_state.active_ids = []
        st.session_state.trained_config_id = None
    st.session_state.current_config_id = config_id


def add_results(results):
    if len(st.session_state.history) + len(results) > 30:
        raise ValueError('Session limit: 30 trained/imported results. Remove old experiments before training or importing more.')
    st.session_state.history.extend(results)
    st.session_state.active_ids = [r['id'] for r in results]
    st.session_state.trained_config_id = st.session_state.current_config_id


def reset_session():
    for key in list(st.session_state):
        del st.session_state[key]

"""Four small pages for one public dataset and two models."""
from datetime import datetime, timezone
import json
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from default_studio.data import (load_dataset, metadata, FEATURES, TARGET, MONTHS,
                                  PAYMENT_STATUS, BILLS, PAYMENTS, CLASS_LABELS)
from default_studio.modeling import (train_models, comparison_table, predict_record,
    FOREST_PARAMETERS, ANN_PARAMETERS, ANN_MAX_EPOCHS, ANN_PATIENCE, SEED)

PAGES = ('Dataset', 'Train', 'Compare', 'Predict')
COLORS = {'Random Forest': '#2563eb', 'ANN': '#0d9488'}


@st.cache_data(show_spinner=False)
def public_dataset(checksum):
    return load_dataset()


def chart(fig, key=None):
    fig.update_layout(template='plotly_white', font={'family': 'Arial, sans-serif', 'size': 13},
                      margin={'l': 20, 'r': 20, 't': 45, 'b': 25}, legend_title_text='')
    st.plotly_chart(fig, key=key, width='stretch', config={'displaylogo': False})


def go_to(page):
    st.session_state.studio_page = page


def require_models():
    if st.session_state.benchmark is None:
        st.info('Train the two models first. Results and predictions will appear here.')
        st.button('Go to training', on_click=go_to, args=('Train',), type='primary')
        return None
    return st.session_state.benchmark


def dataset_page(frame):
    st.caption('ONE DATASET · TWO MODELS')
    st.title('Understand credit-card default.')
    st.write('Explore 30,000 customers, train an ANN and a Random Forest, and compare how well they identify payment default next month.')
    columns = st.columns(3)
    columns[0].metric('Customers', f'{len(frame):,}')
    columns[1].metric('Input features', len(FEATURES))
    columns[2].metric('Default rate', f'{frame[TARGET].mean():.2%}')
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader('What are we predicting?')
        st.write('**Default = 1:** the customer defaulted on payment the following month. '
                 '**No default = 0:** the customer did not default.')
        st.write('The data records credit limits, customer characteristics, repayment status, '
                 'bill balances and previous payments from Taiwan in 2005. Currency values are New Taiwan dollars (NT$).')
        st.caption('This is payment-default research. The dataset does not label fraudulent transactions.')
    with right:
        distribution = pd.DataFrame({'Outcome': ['No default', 'Default'],
                                     'Customers': [int((frame[TARGET] == 0).sum()), int(frame[TARGET].sum())]})
        chart(px.bar(distribution, x='Outcome', y='Customers', text='Customers',
                     color='Outcome', color_discrete_sequence=['#2563eb', '#f59e0b'],
                     title='Actual class distribution'))
    st.subheader('All the data, one fixed split')
    for column, title, text in zip(st.columns(3), ['30,000 records', '24,000 for training', '6,000 for testing'],
        ['The complete dataset is bundled with the app. No upload or sample is needed.',
         'Both models share the same development partition. ANN early stopping uses an inner validation split.',
         'The same held-out customers evaluate both models. Test data never fits preprocessing or chooses ANN epochs.']):
        with column.container(border=True):
            st.markdown(f'**{title}**')
            st.write(text)
    st.button('Continue to training', on_click=go_to, args=('Train',), type='primary')
    with st.expander('Preview the dataset'):
        st.dataframe(frame.head(100), hide_index=True, width='stretch')
        st.caption('First 100 of 30,000 records. ID identifies a record and is excluded from the 23 model inputs.')
    with st.expander('Feature guide and source'):
        st.dataframe(pd.DataFrame([
            ['Credit limit & age', 'LIMIT_BAL, AGE'],
            ['Customer categories', 'SEX, EDUCATION, MARRIAGE'],
            ['Repayment status', 'PAY_0 (September), PAY_2 through PAY_6 (August–April)'],
            ['Bill balances', 'BILL_AMT1 through BILL_AMT6'],
            ['Previous payments', 'PAY_AMT1 through PAY_AMT6'],
        ], columns=['Group', 'Original columns']), hide_index=True, width='stretch')
        st.write('Original category codes, including unknown/other codes, are retained. '
                 'Categorical values use one-hot encoding. ANN numerical inputs use training-only standard scaling.')
        st.markdown('[UCI dataset and citation](https://doi.org/10.24432/C55S3H) · '
                    '[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · I-Cheng Yeh (2009).')
        st.caption('Historical demographic features and repayment data require bias, calibration and drift assessment before any real-world use.')


def train_page(frame):
    st.title('Train two models.')
    st.write('One click prepares both models on all 24,000 development records. Page changes and predictions reuse the fitted models in your session.')
    left, right = st.columns(2)
    with left.container(border=True):
        st.subheader('Random Forest')
        st.write('80 trees · maximum depth 8 · balanced class weights')
        st.caption('Captures nonlinear patterns without numerical scaling. Runs with one worker.')
    with right.container(border=True):
        st.subheader('ANN')
        st.write('One hidden layer · 32 neurons · up to 80 epochs')
        st.caption('Uses scaled numerical inputs and balanced training weights. Early stopping restores the best validation checkpoint.')
    st.info('Fixed 80/20 stratified split · random state 42 · 23 inputs · decision threshold 0.50')
    existing = st.session_state.benchmark
    if existing is None:
        if st.button('Train ANN and Random Forest', type='primary', width='stretch'):
            current = st.empty()
            started = time.perf_counter()
            def progress(name, detail):
                current.info(f'{name} · {detail} · elapsed {time.perf_counter() - started:.1f}s')
            try:
                with st.spinner('Training the two models. This may take a little time on a shared CPU…'):
                    st.session_state.benchmark = train_models(frame, progress)
                current.empty()
                st.rerun()
            except (ValueError, MemoryError, RuntimeError) as exc:
                current.empty()
                st.error(f'Training did not complete: {exc}')
    else:
        st.success(f"Both models are ready. Original training/evaluation time: {existing['elapsed_seconds']:.1f}s. "
                   f"ANN restored epoch {existing['ann_best_epoch']} from {len(existing['ann_history'])} completed epochs.")
        st.button('Compare the results', on_click=go_to, args=('Compare',), type='primary')
        st.caption('Training is not repeated on navigation or prediction. Clear this session to start a fresh run.')
    with st.expander('How the comparison stays fair'):
        st.write('Both models receive the same 24,000-customer development partition and the same untouched 6,000-customer test set. '
                 'The forest fits all development rows. The ANN divides those rows into 21,600 fit rows and 2,400 validation rows for early stopping; '
                 'its imputer, encoder and scaler fit only the 21,600 rows. Neither model sees the test data during fitting.')
        st.write('No oversampling, undersampling or feature selection changes the dataset. Class weighting is learned from training labels. '
                 'The ANN stops after eight epochs without sufficient validation-loss improvement. '
                 'The app reports held-out results without using them to declare or tune a winning model.')


def compare_page():
    st.title('Compare what matters.')
    benchmark = require_models()
    if benchmark is None:
        return
    st.write('Both models are evaluated on the same 6,000 unseen customers. Inspect default recall and false alarms alongside overall accuracy.')
    table = comparison_table(benchmark)
    st.dataframe(table, hide_index=True, width='stretch', column_config={
        key: st.column_config.NumberColumn(format='%.3f') for key in table.columns if key != 'Model'})
    st.caption(f"Predicting the training-majority class for every test customer scores {benchmark['majority_baseline']:.2%} accuracy. "
               'That is a reference statistic, not an additional trained model.')
    melted = table.melt(id_vars='Model', value_vars=['Balanced accuracy', 'Precision (default)', 'Recall (default)', 'F1 (default)'],
                        var_name='Metric', value_name='Score')
    chart(px.bar(melted, x='Metric', y='Score', color='Model', barmode='group',
                 color_discrete_map=COLORS, range_y=[0, 1.02], title='Default-class performance'))
    st.subheader('Which errors does each model make?')
    for column, (name, result) in zip(st.columns(2), benchmark['models'].items()):
        with column:
            matrix = result['confusion_matrix']
            chart(px.imshow(matrix, x=['No default', 'Default'], y=['No default', 'Default'],
                            text_auto=True, color_continuous_scale='Blues',
                            labels={'x': 'Predicted', 'y': 'Actual', 'color': 'Customers'}, title=name))
            tn, fp, fn, tp = matrix.ravel()
            st.caption(f'Default cases detected: {tp:,} · missed: {fn:,}. Non-default customers flagged: {fp:,} · cleared: {tn:,}.')
    with st.expander('ROC and precision–recall curves'):
        for kind in ['ROC', 'Precision–recall']:
            fig = go.Figure()
            for name, result in benchmark['models'].items():
                curve = result['curves']
                x, y = (curve['fpr'], curve['tpr']) if kind == 'ROC' else (curve['recall'], curve['precision'])
                fig.add_trace(go.Scatter(x=x, y=y, name=name, mode='lines', line={'color': COLORS[name]}))
            fig.update_layout(title=kind, xaxis_title='False positive rate' if kind == 'ROC' else 'Recall',
                              yaxis_title='True positive rate' if kind == 'ROC' else 'Precision')
            fig.update_xaxes(range=[0, 1])
            fig.update_yaxes(range=[0, 1.02])
            chart(fig)
        st.caption('ROC-AUC and average precision use continuous probability scores. Average precision summarizes ranking; '
                   'it differs from precision at the 0.50 decision threshold.')
    with st.expander('ANN learning progress'):
        chart(px.line(benchmark['ann_history'], x='epoch', y=['training_loss', 'validation_log_loss'],
                      labels={'value': 'Weighted loss', 'epoch': 'Epoch'}, title='Training and validation loss'))
        st.caption(f"Checkpoint selected by inner validation loss: epoch {benchmark['ann_best_epoch']}. Test scores never selected this epoch.")
    st.subheader('Keep your results')
    export = {'dataset': metadata(), 'seed': SEED, 'threshold': 0.5,
              'train_customer_ids': public_dataset(metadata()['bundled_sha256']).ID.iloc[benchmark['train_indices']].tolist(),
              'test_customer_ids': benchmark['test_customer_ids'].tolist(),
              'ann_fit_customer_ids': public_dataset(metadata()['bundled_sha256']).ID.iloc[benchmark['ann_fit_indices']].tolist(),
              'ann_validation_customer_ids': public_dataset(metadata()['bundled_sha256']).ID.iloc[benchmark['ann_stop_indices']].tolist(),
              'models': {'Random Forest': FOREST_PARAMETERS, 'ANN': ANN_PARAMETERS},
              'ann_max_epochs': ANN_MAX_EPOCHS, 'ann_patience': ANN_PATIENCE, 'ann_best_epoch': benchmark['ann_best_epoch'],
              'metrics': table.to_dict(orient='records'), 'versions': benchmark['versions'],
              'trained_at_utc': benchmark['trained_at']}
    left, right = st.columns(2)
    left.download_button('Download comparison CSV', table.to_csv(index=False), 'model_comparison.csv', 'text/csv', width='stretch')
    right.download_button('Download experiment JSON', json.dumps(export, indent=2), 'credit_default_experiment.json',
                           'application/json', width='stretch')
    st.caption('This single split is a reproducible research comparison. Repeatedly inspecting and tuning to its test scores '
               'weakens the evaluation; current lending decisions require separate validation and calibration.')


def predict_page(frame):
    st.title('Try a customer profile.')
    benchmark = require_models()
    if benchmark is None:
        return
    st.write('Edit a public example and ask both fitted models for their estimated next-month default probability.')
    examples = [1, 2, 3, 4, 50, 101]
    customer_id = st.selectbox('Start from a dataset example', examples, format_func=lambda value: f'Customer {value}')
    row = frame.loc[frame.ID == customer_id].iloc[0]
    sex = {1: 'Male (1)', 2: 'Female (2)'}
    education = {0: 'Unknown (0)', 1: 'Graduate school (1)', 2: 'University (2)',
                 3: 'High school (3)', 4: 'Other (4)', 5: 'Unknown (5)', 6: 'Unknown (6)'}
    marriage = {0: 'Unknown (0)', 1: 'Married (1)', 2: 'Single (2)', 3: 'Other (3)'}
    with st.form(f'customer_profile_{customer_id}'):
        left, right = st.columns(2)
        limit = left.number_input('Credit limit (NT$)', min_value=0.0, max_value=10000000.0, value=float(row.LIMIT_BAL), step=10000.0)
        age = right.number_input('Age', min_value=18, max_value=100, value=int(row.AGE))
        columns = st.columns(3)
        gender = columns[0].selectbox('Sex (original dataset categories)', list(sex), index=list(sex).index(int(row.SEX)), format_func=sex.get)
        edu = columns[1].selectbox('Education', list(education), index=list(education).index(int(row.EDUCATION)), format_func=education.get)
        married = columns[2].selectbox('Marital status', list(marriage), index=list(marriage).index(int(row.MARRIAGE)), format_func=marriage.get)
        st.subheader('Six months of repayment history')
        history = pd.DataFrame({'Month': MONTHS, 'Repayment code': row[PAYMENT_STATUS].to_numpy(dtype=int),
                                'Bill balance (NT$)': row[BILLS].to_numpy(dtype=int),
                                'Previous payment (NT$)': row[PAYMENTS].to_numpy(dtype=int)})
        edited = st.data_editor(history, hide_index=True, width='stretch', disabled=['Month'],
            column_config={'Repayment code': st.column_config.SelectboxColumn(options=list(range(-2, 10)), required=True),
                           'Bill balance (NT$)': st.column_config.NumberColumn(required=True),
                           'Previous payment (NT$)': st.column_config.NumberColumn(min_value=0, required=True)},
            num_rows='fixed')
        st.caption('Repayment codes: −2 no consumption, −1 paid duly, 0 revolving credit; '
                   '1–8 indicate months of delay, 9 means nine or more. Negative bill balances are permitted.')
        if st.form_submit_button('Predict with both models', type='primary', width='stretch'):
            record = {'LIMIT_BAL': limit, 'AGE': age, 'SEX': gender, 'EDUCATION': edu, 'MARRIAGE': married}
            record.update({column: edited['Repayment code'].iloc[i] for i, column in enumerate(PAYMENT_STATUS)})
            record.update({column: edited['Bill balance (NT$)'].iloc[i] for i, column in enumerate(BILLS)})
            record.update({column: edited['Previous payment (NT$)'].iloc[i] for i, column in enumerate(PAYMENTS)})
            try:
                predictions = {name: predict_record(benchmark, name, pd.DataFrame([record])) for name in benchmark['models']}
                st.session_state.prediction = {'customer_id': customer_id, 'results': predictions}
            except (ValueError, TypeError) as exc:
                st.error(f'Check the customer profile: {exc}')
    prediction = st.session_state.get('prediction')
    if prediction and prediction['customer_id'] == customer_id:
        for column, (name, value) in zip(st.columns(2), prediction['results'].items()):
            with column.container(border=True):
                st.subheader(name)
                st.metric('Estimated default probability', f"{value['probability']:.1%}")
                st.write(f"Prediction at 0.50: **{CLASS_LABELS[value['label']]}**")
        st.caption('These class-weighted model scores are uncalibrated estimates from historical data, '
                   'not a current lending recommendation. Editing the profile requires another prediction click.')


def main():
    st.set_page_config(page_title='Credit Default Studio', page_icon='💳', layout='wide')
    info = metadata()
    if st.session_state.get('studio_dataset_version') != info['bundled_sha256']:
        st.session_state.clear()
        st.session_state.update(studio_dataset_version=info['bundled_sha256'], benchmark=None, studio_page='Dataset')
    frame = public_dataset(info['bundled_sha256'])
    with st.sidebar:
        st.title('Credit Default Studio')
        st.caption('A small workspace for one clear question.')
        st.radio('Your workspace', PAGES, key='studio_page')
        st.divider()
        st.caption('UCI · 30,000 customers · 23 inputs')
        st.caption('Two models ready' if st.session_state.benchmark is not None else 'Models have not been trained')
        if st.button('Clear this session'):
            st.session_state.benchmark = None
            st.session_state.pop('prediction', None)
            st.rerun()
    routes = {'Dataset': lambda: dataset_page(frame), 'Train': lambda: train_page(frame),
              'Compare': compare_page, 'Predict': lambda: predict_page(frame)}
    routes[st.session_state.studio_page]()
    st.divider()
    st.caption('Credit Default Studio · Full public dataset · ANN + Random Forest · Runs on the server')

"""Fraud-focused entry point for local experiments and Colab offloading."""
from pathlib import Path
import pandas as pd
import streamlit as st
from core.fraud import fraud_outcomes
from views.model_pages import choose_result

ROOT = Path(__file__).resolve().parents[1]
COLAB_URL = 'https://colab.research.google.com/github/abbk1/ML-Analytics/blob/main/examples/credit_card_fraud_colab.ipynb'


def colab_downloads():
    st.link_button('Open fraud training notebook in Google Colab', COLAB_URL, type='primary')
    st.link_button('Read the step-by-step Colab guide',
                   'https://github.com/abbk1/ML-Analytics/blob/main/docs/CREDIT_CARD_FRAUD_COLAB.md')
    st.download_button('Download fraud Colab notebook', (ROOT / 'examples/credit_card_fraud_colab.ipynb').read_bytes(),
                       'credit_card_fraud_colab.ipynb', 'application/x-ipynb+json')
    st.download_button('Download standalone fraud trainer', (ROOT / 'examples/creditcard_colab.py').read_bytes(),
                       'creditcard_colab.py', 'text/x-python')


def fraud_scorecard(result):
    outcomes = fraud_outcomes(result)
    if outcomes is None:
        st.info('A binary result with an explicitly selected positive class and confusion matrix is required.')
        return
    st.caption(f"Positive class = {result['positive_class']} (fraud/event) · {result['source']} · held-out observations only")
    for column, (label, count) in zip(st.columns(4), outcomes.items()):
        column.metric(label, f'{count:,}')
    metrics = result['evaluation']['metrics']
    for column, label, key in zip(st.columns(3), ['Average precision', 'Fraud recall', 'False alarm rate'],
                                   ['average_precision', 'recall_positive', 'false_positive_rate']):
        column.metric(label, f'{metrics[key]:.4f}' if key in metrics else 'Not applicable')
    st.caption('Average precision summarizes fraud precision–recall ranking; it is not the same as fraud precision at the selected threshold. '
               'False alarm rate is FP / all legitimate transactions.')


def fraud_page():
    st.header('Credit Card Fraud & Colab')
    st.write('Use Class = 1 for fraud and Class = 0 for legitimate transactions. Prioritize fraud detection, '
             'missed fraud and false alarms. Accuracy alone can reward predicting every transaction as legitimate.')
    st.subheader('Large dataset: train in Colab, analyze results here')
    st.markdown('''
1. Put **creditcard.csv** in your own Google Drive. You do not need to upload the full dataset into this app.
2. Open the notebook below, install its dependencies, mount Drive, and set the CSV path.
3. Choose **batch** to compare ANN and Random Forest on the full dataset when it fits Colab RAM. If it does not fit, choose **incremental** with a Time-sorted CSV; this uses chunked SGD logistic training, a different lightweight model.
4. Train using earlier transactions, choose the candidate and threshold on validation data, and evaluate the selected candidate on later untouched test transactions.
5. Upload only **fraud_results.json** in **Import Colab Results**. View confusion matrices, fraud outcomes, ROC/PR curves and export reports without storing the full CSV here.
''')
    colab_downloads()
    st.info('Streamlit remains limited to 25 MB uploads and 50,000 local training rows. Colab uses a separate runtime with its own finite RAM, CPU and session limits. '
            'A GPU does not accelerate these scikit-learn ANN/Random Forest implementations.')
    with st.expander('Choosing the fraud experiment'):
        st.markdown('''
- **Target / positive label:** `Class` / `1`. Features are `Time`, `Amount`, `V1`–`V28`; do not include `Class` in inputs.
- **Class weighting first:** the notebook avoids expanding rare fraud cases to the size of the legitimate class. Full SMOTE can multiply memory and runtime.
- **Model selection:** compare validation average precision. Only the selected candidate receives a final holdout evaluation.
- **Threshold:** select validation F2 under your chosen precision floor. F2 emphasizes recall but still accounts for false alerts; the floor is a configurable research choice.
- **Time:** the notebook defaults to a 60/20/20 chronological split, with equal timestamps kept together. Fraud must be present in every partition. For stratified independent-record research, explicitly choose batch `split_method="stratified"`.
- **Limits:** batch checks an array-memory budget; incremental mode limits chunk and evaluation-array sizes. Neither mode silently samples. Large/evolving datasets may still require a larger runtime or production infrastructure.
- **Persistence:** save reports/models in Drive; interrupted training is restarted, not automatically resumed. Keep the trusted model in Colab for batch prediction; the app does not deserialize uploaded models.
''')
    if st.session_state.history:
        st.subheader('Fraud outcomes from your experiment')
        result = choose_result()
        if result:
            fraud_scorecard(result)
            if result.get('validation_comparison'):
                st.subheader('Colab candidate comparison · validation only')
                st.dataframe(pd.DataFrame(result['validation_comparison']), hide_index=True, width='stretch')
                st.caption('These are validation scores used to select the candidate, not final test scores.')
            if result.get('validation_metrics'):
                st.write('Validation threshold selection:', result['validation_metrics'])
                st.write('Frozen decision threshold:', result.get('threshold'))
    else:
        st.info('No fraud experiment yet. Start with the Colab notebook for your full dataset, or apply the credit-card preset after uploading a small dataset for local prototyping.')

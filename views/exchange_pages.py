"""Session history, safe Colab exchange, and downloadable research artifacts."""
from pathlib import Path
from io import BytesIO
import pandas as pd
import streamlit as st
from core.artifacts import (import_results, export_experiments, csv_bytes, excel_bytes,
                            dumps, pipeline_bytes, metadata)
from core.training import comparison_table
from core.data import audit_dataset, class_distribution
from utils.session import add_results
from utils.reports import scientific_report
from views.model_pages import choose_result

ROOT = Path(__file__).resolve().parents[1]


def history_page():
    st.header('Experiment History')
    st.write('History is scoped to this session. Dataset/configuration changes clear active results while preserving '
             'previous experiments with their original dataset fingerprints and settings.')
    st.warning('Streamlit Community Cloud sessions and local files are not permanent storage. Download your results and pipelines before leaving.')
    history = st.session_state.history
    if not history:
        st.info('No experiments yet.')
        return
    st.dataframe(comparison_table(history), hide_index=True, width='stretch')
    st.download_button('Download experiment history JSON', export_experiments(history), 'experiment_history.json', 'application/json')
    selected = st.multiselect('Experiments to remove from memory', [r['id'] for r in history])
    if selected and st.button('Delete selected experiments'):
        st.session_state.history = [r for r in history if r['id'] not in selected]
        st.session_state.active_ids = [key for key in st.session_state.active_ids if key not in selected]
        for key in ['prepared_model_download', 'prepared_model_id']:
            st.session_state.pop(key, None)
        st.rerun()
    result = choose_result()
    if result:
        st.subheader('Reproducibility record')
        st.json(metadata(result), expanded=False)


def import_page():
    st.header('Import Colab Results')
    st.subheader('Full credit-card fraud dataset')
    st.write('Train creditcard.csv in Google Colab, then upload fraud_results.json below. The full transaction dataset stays in Drive. '
             'The JSON contains final test metrics and validation-only candidate comparison, with Class=1 as fraud.')
    from views.fraud_page import colab_downloads
    colab_downloads()
    st.write('Import metric CSV tables, prediction CSVs, JSON experiment metadata, or JSON confusion matrices. '
             'Imported evaluations are labeled external and their split equivalence is unverified.')
    st.code('Metrics CSV: model,accuracy,balanced_accuracy,f1_macro\nPredictions CSV: actual,predicted\nConfusion JSON: {"model_name":"Colab model","classes":["no","yes"],"confusion_matrix":[[20,2],[3,5]]}', language='text')
    st.caption('Metrics use fractions, not percentages. JSON experiment bundles use schema_version 1. '
               'Prediction CSV metrics do not infer probabilities from class labels.')
    kind = st.selectbox('CSV interpretation', ['Metrics table', 'Predictions'])
    upload = st.file_uploader('CSV or JSON result file (up to 10 MB)', type=['csv', 'json'], key='result_import')
    if upload is not None and st.button('Validate and import', type='primary'):
        try:
            results = import_results(upload.getvalue(), upload.name, kind)
            add_results(results)
            # External results are history, not active fitted results for this dataset.
            st.session_state.active_ids = []
            st.success(f'Imported {len(results)} external results. View them in Performance Evaluation or Model Comparison.')
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            st.error(f'Import rejected: {exc}')
    st.download_button('Download Colab export example', (ROOT / 'examples' / 'colab_export_example.py').read_bytes(),
                       'colab_export_example.py', 'text/x-python')
    st.info('Uploaded notebooks, scripts, Pickle and Joblib files are never executed or deserialized. '
            'Locally exported pipelines should only be loaded in a trusted Python environment.')
    image = st.file_uploader('Optional reference chart image', type=['png', 'jpg', 'jpeg'], key='reference_image')
    if image is not None:
        try:
            from PIL import Image
            if image.size > 5 * 1024**2:
                raise ValueError('Reference images are limited to 5 MB.')
            with Image.open(BytesIO(image.getvalue())) as chart:
                if chart.width * chart.height > 12000000:
                    raise ValueError('Reference image exceeds 12 million pixels.')
                chart.verify()
            st.image(image.getvalue(), caption='Imported reference image · not independently verified')
        except Exception as exc:
            st.error(f'Image rejected: {exc}')


def export_page():
    st.header('Export and Reports')
    frame = st.session_state.dataset
    if frame is not None:
        st.subheader('Dataset summaries')
        summary = audit_dataset(frame, st.session_state.target)['columns'].reset_index(names='feature')
        st.download_button('Dataset summary CSV', csv_bytes(summary), 'dataset_summary.csv', 'text/csv')
        if st.session_state.target:
            distribution = class_distribution(frame[st.session_state.target])
            st.download_button('Class distribution CSV', csv_bytes(distribution), 'class_distribution.csv', 'text/csv')
    if not st.session_state.history:
        st.info('Train or import an experiment for evaluation, model and report downloads.')
        return
    result = choose_result()
    comparison = comparison_table(st.session_state.history)
    st.download_button('Model comparison CSV', csv_bytes(comparison), 'model_comparison.csv', 'text/csv')
    st.download_button('All experiment metadata JSON', export_experiments(st.session_state.history), 'experiments.json', 'application/json')
    report = result['evaluation']['per_class'].reset_index(names='class')
    st.download_button('Classification report CSV', csv_bytes(report), 'classification_report.csv', 'text/csv')
    st.download_button('Preprocessing configuration JSON', dumps(result.get('config', {}).get('preprocessing', {})),
                       'preprocessing.json', 'application/json')
    matrix = result['evaluation']['confusion_matrix']
    tables = {'comparison': comparison, 'classification_report': report}
    if matrix is not None:
        cm = pd.DataFrame(matrix, columns=result['classes'], index=result['classes']).reset_index(names='actual')
        st.download_button('Confusion matrix CSV', csv_bytes(cm), 'confusion_matrix.csv', 'text/csv')
        tables['confusion_matrix'] = cm
    if result.get('cv_results') is not None:
        tables['cross_validation'] = result['cv_results']
        st.download_button('Cross-validation CSV', csv_bytes(result['cv_results']), 'cross_validation.csv', 'text/csv')
    if result.get('errors') is not None:
        errors = result['errors'][~result['errors']._correct]
        st.download_button('Misclassified observations CSV', csv_bytes(errors), 'misclassified.csv', 'text/csv')
    st.download_button('Excel evaluation workbook', excel_bytes(tables), 'evaluation.xlsx',
                       'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    st.download_button('Scientific experiment report (Markdown)', scientific_report(result, comparison), 'experiment_report.md', 'text/markdown')
    st.caption('Reports describe observed results and limitations. Use Model Comparison to inspect evidence across candidates. '
               'Interactive charts can be downloaded as PNG using each chart’s camera control.')
    if result.get('pipeline') is not None:
        if st.button('Prepare fitted pipeline download'):
            st.session_state.prepared_model_download = pipeline_bytes(result)
            st.session_state.prepared_model_id = result['id']
        if st.session_state.get('prepared_model_id') == result['id']:
            st.download_button('Download fitted pipeline + metadata (Joblib)', st.session_state.prepared_model_download,
                               f"pipeline_{result['id']}.joblib", 'application/octet-stream')
        st.caption('The artifact includes the fitted preprocessing pipeline, classifier, chosen threshold and metadata. '
                   'Use the same package versions when restoring a trusted export. Never load an untrusted Joblib file.')


def help_page():
    st.header('Help and Documentation')
    st.info('For full creditcard.csv training, open Credit Card Fraud & Colab. Use Class=1 as fraud, train in Colab, '
            'and import fraud_results.json here. Small local experiments can use the explicit fraud preset in Dataset Management.')
    st.markdown('''
### A reproducible workflow
1. **Dataset Management**: upload CSV, TSV or .xlsx, audit warnings, and apply a target and feature selection.
2. **Exploratory Data Analysis**: inspect selected distributions and missing values. Charts explicitly sample at most 5,000 rows.
3. **Data Preprocessing**: save imputation, encoding and scaling; Text mode uses fold-local TF-IDF.
4. **Class Imbalance Analysis**: inspect minority support and save a compatible training-only resampler.
5. **Model Configuration**: choose classifiers. Advanced mode exposes JSON hyperparameters; class weights are model parameters.
6. **Model Training**: explicitly launch original, resampled or paired experiments. The same untouched holdout is used within a run.
7. **Performance Evaluation / Model Comparison / Error Analysis**: examine CV, confusion matrices, per-class recall, PR curves and errors.
8. **Prediction / Export and Reports**: reuse fitted pipelines and download reproducibility records before your session ends.

### Scientific interpretation
- **Macro F1** gives each class equal weight. **Balanced accuracy** is macro recall for single-label classification.
- **Weighted F1** weights by observed support. **Accuracy** can hide minority-class failures. The majority baseline is not evidence of useful detection.
- **ROC-AUC / Average Precision** use probabilities or decision scores, never hard predictions. Multiclass curves and metrics are one-vs-rest.
- **Decision scores are not probabilities**. Resampling and class weights can affect probability calibration.
- **Candidate selection uses training CV**. Threshold tuning uses training out-of-fold scores; its displayed optimized F1 is exploratory and not an unbiased validation estimate.
- The holdout is for final descriptive evaluation. Repeated tuning after inspecting it contaminates the scientific selection process; reserve a new independent final dataset.
- This app supports independent-record classification. Temporal, patient, household or other grouped data need a group/time-aware design outside this version.
- Exact duplicate records and conflicting feature-label groups need review. Removing duplicates is explicit; no target leakage detector can identify all domain-specific leakage.

### Compatibility and cloud limits
- StandardScaler is recommended for scale-sensitive numerical models. Tree-based models generally do not need scaling.
- One-hot encoding handles unseen categories. Ordinal encoding imposes an ordering and should be justified.
- Multinomial NB accepts nonnegative TF-IDF; Gaussian NB and Gradient Boosting use bounded dense conversion for small tabular matrices only.
- SMOTE cannot interpolate ordinary categorical encodings. Use SMOTENC for mixed numerical/categorical data. Random sampling works with sparse text.
- Every oversampled class needs more than k training samples in each fold for SMOTE. Failures never silently change the method.
- TF-IDF SMOTE creates synthetic vectors, not documents. Consider class weights or random sampling for text.
- Limits: 25 MB uploads; 150 MB parsed data; 500 columns; 50,000 training rows; 100,000 resampled rows; six classifiers; 120 fits/run; 30 session results.
- SVM/KNN have a 10,000-training-row limit. Dense-only classifiers reject matrices larger than two million cells. Large problems should use Colab.
- Training is sequential with one worker; no GPU, database, API key or external backend is required.

### Privacy and reproducibility
Uploaded data and models remain in session state and are never globally cached. Session state and Community Cloud files are temporary.
Downloaded error tables may contain original data; store them appropriately. Spreadsheet exports escape formula-like text.
JSON/CSV imports are schema validated. Uploaded Python, notebook, Pickle and Joblib artifacts are not executed or loaded.
Package versions, split indices, dataset fingerprints, parameters, warnings and UTC timestamps accompany local experiments.
''')

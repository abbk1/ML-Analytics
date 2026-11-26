"""Model settings, sequential training, evaluation, comparison, and inference pages."""
from dataclasses import replace
import hashlib
import json
import time
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from core.config import SplitConfig, RANKING_METRICS
from core.data import audit_dataset, class_distribution, load_dataset
from core.models import MODEL_CLASSES, DEFAULTS, SCALE_SENSITIVE, make_model
from core.training import train_experiments, comparison_table
from core.prediction import predict_batch
from core.artifacts import csv_bytes
from utils.charts import show, confusion_chart, response_curves, color_for
from utils.session import add_results


def choose_result(trained_only=False):
    history = [r for r in st.session_state.history if not trained_only or r.get('pipeline') is not None]
    if not history:
        st.info('Train models in Model Training or import compatible Colab results first.')
        return None
    lookup = {r['id']: r for r in history}
    selected = st.selectbox('Experiment result', list(lookup), format_func=lambda key:
        f"{lookup[key]['model_name']} · {lookup[key].get('experiment_name', '')} · {lookup[key]['source']} · {key}")
    return lookup[selected]


def dashboard(config):
    st.header('ML Analytics Studio')
    st.write('A reproducible workspace for classification, minority-class analysis, and fair model comparison. '
             'Upload your data, prepare training-only transformations, and evaluate on an untouched holdout.')
    frame = st.session_state.dataset
    if frame is None:
        st.info('Your workspace is empty. Upload a CSV, TSV or Excel classification dataset to start. No example results are fabricated.')
    else:
        audit = audit_dataset(frame, st.session_state.target)
        distribution = class_distribution(frame[st.session_state.target]) if st.session_state.target else None
        st.subheader(st.session_state.dataset_name)
        cols = st.columns(3)
        cols[0].metric('Observations', len(frame))
        cols[1].metric('Selected features', len(st.session_state.feature_selection))
        cols[2].metric('Target classes', len(distribution) if distribution is not None else 'Select target')
        cols = st.columns(3)
        cols[0].metric('Missing values', audit['missing'])
        cols[1].metric('Imbalance ratio', f"{distribution['count'].max() / distribution['count'].min():.2f}" if distribution is not None and len(distribution) else '—')
        local = [r for r in st.session_state.history if r.get('dataset_fingerprint') == st.session_state.dataset_id and r.get('pipeline') is not None]
        cols[2].metric('Trained models on this dataset', len(local))
        eligible = [r for r in local if config and r['config']['primary_metric'] == config.primary_metric]
        if eligible:
            best = max(eligible, key=lambda r: r['cv_mean'])
            st.success(f"Best training-CV candidate ({config.primary_metric}): {best['model_name']} · "
                       f"{best['cv_mean']:.3f} ± {best['cv_std']:.3f}. Holdout metrics are descriptive, not the selection criterion.")
        if st.session_state.history:
            latest = st.session_state.history[-1]
            st.write(f"Latest experiment: **{latest['model_name']}** · {latest.get('experiment_name', '')} · {latest['timestamp']}")
    st.subheader('Workflow')
    for col, label, destination in zip(st.columns(3), ['1. Upload & audit', '2. Configure & train', '3. Evaluate & export'],
                                        ['Dataset Management', 'Model Configuration', 'Performance Evaluation']):
        col.button(label, on_click=lambda page=destination: st.session_state.update(navigation=page), width='stretch')
    st.caption('All computation runs on the server. Uploaded data and models are scoped to your Streamlit session, with no global data cache.')


def model_configuration():
    st.header('Model Configuration')
    st.write('Select up to six classifiers. Conservative defaults run sequentially with one worker. '
             'The Dummy baseline provides a majority/prior reference.')
    advanced = st.session_state.ui_mode == 'Advanced'
    with st.form('models'):
        selected = st.multiselect('Classifiers', list(MODEL_CLASSES), default=list(st.session_state.model_settings), max_selections=6)
        text_values = {}
        for name in MODEL_CLASSES:
            if advanced or name in st.session_state.model_settings:
                with st.expander(f'{name} parameters', expanded=False):
                    st.caption('JSON overrides expose the documented estimator parameters. class_weight: "balanced" is supported by compatible models.')
                    text_values[name] = st.text_area(f'{name} JSON', value=json.dumps(
                        st.session_state.model_settings.get(name, DEFAULTS[name]), indent=2), height=220, key=f'params_{name}')
        if st.form_submit_button('Save model configuration', type='primary'):
            try:
                if not selected:
                    raise ValueError('Select at least one classifier.')
                settings = {name: json.loads(text_values[name]) if name in text_values else dict(DEFAULTS[name]) for name in selected}
                for name, parameters in settings.items():
                    make_model(name, parameters)
                st.session_state.model_settings = settings
                st.session_state.active_ids = []
                st.rerun()
            except (ValueError, TypeError) as exc:
                st.error(f'Invalid model configuration: {exc}')
    if st.session_state.prep_config.scaling == 'none' and set(st.session_state.model_settings) & SCALE_SENSITIVE:
        st.warning('Scaling is disabled for a scale-sensitive model. Consider StandardScaler or RobustScaler.')
    if st.session_state.prep_config.task == 'Text':
        st.info('Linear SVM, logistic regression and Multinomial NB are efficient text baselines. Gaussian NB, '
                'Gradient Boosting and SMOTENC are disabled for sparse text. SVM probability estimation is off; decision scores support ROC/PR analysis.')
    st.caption('Advanced mode exposes every model’s configuration. Model-specific random_state uses the shared experiment seed. '
               'Finite depth, iteration, estimator, layer and worker limits protect free hosting.')


def training_page(config):
    st.header('Model Training')
    if config is None:
        st.info('Load a dataset and apply target/features in Dataset Management first.')
        return
    if st.session_state.dataset[config.target].dropna().nunique() < 2:
        st.error('Choose a target containing at least two nonmissing classes before training.')
        return
    st.write('Candidates share one stratified holdout. Training-only cross-validation ranks models; '
             'the holdout is never resampled or used for hyperparameter or threshold selection.')
    modes = ['A · Original distribution', 'B · Resampled training', 'C · Compare original and resampled', 'Custom · Saved strategy']
    with st.form('training'):
        name = st.text_input('Experiment name', value=st.session_state.experiment_name, max_chars=120)
        mode = st.selectbox('Experiment mode', modes)
        columns = st.columns(3)
        fraction = columns[0].slider('Holdout test fraction', min_value=0.1, max_value=0.4, value=config.split.test_size, step=0.05)
        folds = columns[1].number_input('Stratified CV folds', min_value=2, max_value=10, value=config.split.folds)
        repeats = columns[2].number_input('CV repeats', min_value=1, max_value=3, value=config.split.repeats)
        seed = st.number_input('Random state', min_value=0, max_value=2147483647, value=config.split.random_state)
        names = list(RANKING_METRICS)
        metric_name = st.selectbox('Primary ranking metric (training CV)', names,
                                   index=list(RANKING_METRICS.values()).index(config.primary_metric))
        classes = sorted(st.session_state.dataset[config.target].dropna().astype(str).unique())
        positive = st.selectbox('Positive class (binary tasks; select the event of interest)', classes,
                                index=classes.index(config.positive_class) if config.positive_class in classes else len(classes) - 1)
        threshold = st.checkbox('Select a binary decision threshold by out-of-fold training F1', value=config.optimize_threshold,
                                help='The precision–recall trade-off and chosen threshold use training OOF responses only. Holdout scores never select the threshold.')
        st.caption(f"Models: {', '.join(st.session_state.model_settings)} · Saved resampling: {config.resampling.method} · "
                   f"Scaling: {config.preprocessing.scaling}. Save preprocessing/model/imbalance settings on their pages.")
        submitted = st.form_submit_button('Train selected models', type='primary')
    if submitted:
        current = replace(config, name=name or 'Experiment', split=SplitConfig(fraction, seed, folds, repeats),
                          primary_metric=RANKING_METRICS[metric_name], positive_class=positive, optimize_threshold=threshold)
        st.session_state.split_config = current.split
        st.session_state.primary_metric = current.primary_metric
        st.session_state.positive_class = positive
        st.session_state.optimize_threshold = threshold
        st.session_state.experiment_name = current.name
        strategies = ['none'] if mode.startswith('A') else [current.resampling.method]
        if mode.startswith('C'):
            strategies = list(dict.fromkeys(['none', current.resampling.method]))
        if (mode.startswith('B') or mode.startswith('C')) and current.resampling.method == 'none':
            st.error('Select and save a resampling method in Class Imbalance Analysis for experiment B or C.')
            return
        if len(st.session_state.history) + len(strategies) * len(st.session_state.model_settings) > 30:
            st.error('This run would exceed 30 session results. Delete old experiments in Experiment History first.')
            return
        bar = st.progress(0.0)
        status = st.empty()
        started = time.perf_counter()
        def progress(value, model, stage):
            bar.progress(float(min(value, 1)))
            status.write(f'{model} · {stage} · elapsed {time.perf_counter() - started:.1f}s')
        try:
            with st.spinner('Fitting training folds and final pipelines…'):
                results, failures = train_experiments(st.session_state.dataset, current,
                    st.session_state.model_settings, strategies, progress)
            for result in results:
                result['dataset_name'] = st.session_state.dataset_name
            st.session_state.current_config_id = hashlib.sha256((current.fingerprint() + json.dumps(
                st.session_state.model_settings, sort_keys=True)).encode()).hexdigest()
            add_results(results)
            bar.progress(1.0)
            if results:
                st.success(f'{len(results)} model/strategy candidates completed in {time.perf_counter() - started:.1f}s.')
                st.dataframe(comparison_table(results), hide_index=True, width='stretch')
            for failure in failures:
                st.error(f"{failure['model']} / {failure['strategy']}: {failure['error']}")
            for result in results:
                if result['warnings']:
                    with st.expander(f"Warnings · {result['model_name']} / {result['config']['resampling']['method']}"):
                        for message in result['warnings']:
                            st.warning(message)
            st.caption('Open Performance Evaluation for the full report and Model Comparison for strategy trade-offs. '
                       'Previously held-out results should not guide iterative tuning; use an independent final dataset for confirmatory work.')
        except (ValueError, TypeError) as exc:
            st.error(str(exc))
    active = [r for r in st.session_state.history if r['id'] in st.session_state.active_ids]
    if active and not submitted:
        st.subheader('Current results')
        st.dataframe(comparison_table(active), hide_index=True, width='stretch')


def evaluation_page():
    st.header('Performance Evaluation')
    result = choose_result()
    if result is None:
        return
    evaluated = result['evaluation']
    metrics = evaluated['metrics']
    cols = st.columns(3)
    for column, key in zip(cols, ['accuracy', 'balanced_accuracy', 'f1_macro']):
        column.metric(key.replace('_', ' ').title(), f'{metrics[key]:.3f}' if key in metrics else 'Not applicable')
    st.caption('For single-label classification, balanced accuracy equals macro recall. Micro precision, recall and F1 equal accuracy.')
    st.subheader('Confusion matrix')
    confusion_chart(result)
    if st.checkbox('Show row-normalized confusion matrix'):
        confusion_chart(result, normalized=True)
    st.subheader('Precision, recall and F1')
    selected = {key: value for key, value in metrics.items() if key.startswith(('precision_', 'recall_', 'f1_'))}
    if selected:
        table = pd.DataFrame({'metric': list(selected), 'value': list(selected.values())})
        show(px.bar(table, x='metric', y='value', range_y=[0, 1.02]))
    st.subheader('Per-class performance')
    report = evaluated['per_class']
    st.dataframe(report, width='stretch')
    if not report.empty:
        show(px.imshow(report[['precision', 'recall', 'f1-score']].to_numpy(), x=['Precision', 'Recall', 'F1'],
                       y=report.index, zmin=0, zmax=1, text_auto='.3f', color_continuous_scale='Blues'))
    st.subheader('ROC curves')
    response_curves(result, 'ROC')
    st.subheader('Precision–recall curves')
    response_curves(result, 'Precision–recall')
    st.caption(f"Response type: {evaluated.get('response_kind', 'unavailable')}. Decision scores are not calibrated probabilities. "
               'Multiclass ROC-AUC and AP use macro-averaged one-vs-rest class responses.')
    st.subheader('All metrics')
    st.dataframe(pd.DataFrame({'metric': list(metrics), 'value': list(metrics.values())}), hide_index=True, width='stretch')
    for metric, reason in evaluated.get('unavailable', {}).items():
        st.info(f'{metric}: Not applicable — {reason}')
    st.subheader('Training time and cross-validation')
    if result.get('cv_scores'):
        st.write(f"Final fit: {result['fit_seconds']:.3f}s · Entire CV + fit + evaluation: {result['total_seconds']:.3f}s")
        st.metric('Training-CV primary metric', f"{result['cv_mean']:.3f} ± {result['cv_std']:.3f}")
        st.dataframe(result['cv_results'], hide_index=True, width='stretch')
        show(px.box(pd.DataFrame({'score': result['cv_scores']}), y='score', points='all', title='Training-fold scores'))
    if result.get('threshold_table') is not None:
        st.subheader('Threshold analysis · training OOF only')
        st.write(f"Selected threshold: {result['threshold']:.4f}; optimized positive-class F1. CV ranking above uses default decision thresholds.")
        table = result['threshold_table']
        show(px.line(table, x='threshold', y=['precision', 'recall', 'f1_positive']))
        st.dataframe(table, hide_index=True, width='stretch')
    pipeline = result.get('pipeline')
    if pipeline:
        model = pipeline.named_steps['model']
        if hasattr(model, 'loss_curve_'):
            show(px.line(y=model.loss_curve_, labels={'index': 'Iteration', 'y': 'Training loss'}, title='MLP convergence'))
            st.caption(f"Iterations: {model.n_iter_}; training loss: {model.loss_:.5f}. See warnings for non-convergence.")
        if hasattr(model, 'feature_importances_'):
            try:
                names = pipeline.named_steps.get('post_encode', pipeline.named_steps['preprocess']).get_feature_names_out()
            except (ValueError, AttributeError):
                names = [f'transformed_feature_{i}' for i in range(len(model.feature_importances_))]
            importance = pd.DataFrame({'feature': names, 'importance': model.feature_importances_}).nlargest(25, 'importance')
            show(px.bar(importance, x='importance', y='feature', orientation='h', title='Top feature importances (impurity-based)'))
            st.caption('Impurity importance can favor high-cardinality features and does not establish causation.')
    for message in result.get('warnings', []):
        st.warning(message)


def comparison_page():
    st.header('Model Comparison')
    results = st.session_state.history
    if not results:
        st.info('Train at least one candidate or import a metric table.')
        return
    experiments = list(dict.fromkeys(r.get('experiment_name', '') for r in results))
    chosen = st.multiselect('Experiments', experiments, default=experiments)
    filtered = [r for r in results if r.get('experiment_name', '') in chosen]
    if not filtered:
        st.info('Select at least one experiment.')
        return
    table = comparison_table(filtered)
    if table.split_id.nunique() > 1:
        st.warning('Non-equivalent comparison: these experiments use different or unverified test observations. '
                   'Filter to one local split identity before directly comparing holdout metrics.')
    else:
        st.success(f"Equivalent holdout observations: {table.split_id.iloc[0]}")
    columns = [c for c in table if pd.api.types.is_numeric_dtype(table[c]) and table[c].notna().any()]
    rank = st.selectbox('Sort table by', columns, index=columns.index('cv_mean') if 'cv_mean' in columns else 0)
    if rank != 'cv_mean':
        st.caption('Sorting holdout results is descriptive. Choose candidates using training-CV scores, not this holdout ranking.')
    if table.primary_metric.nunique() > 1:
        st.warning('CV means use different primary metrics and are not directly comparable.')
    st.dataframe(table.sort_values(rank, ascending=rank == 'training_seconds'), hide_index=True, width='stretch')
    table['candidate'] = table.model + ' · ' + table.resampling + ' · ' + table.id
    color_map = {row.candidate: color_for(row.model) for row in table.itertuples()}
    for metric in ['accuracy', 'balanced_accuracy']:
        if metric in table:
            show(px.bar(table, x='candidate', y=metric, color='candidate', color_discrete_map=color_map,
                        range_y=[0, 1.02], title=metric.replace('_', ' ').title()))
    st.subheader('Accuracy and minority-class performance')
    metrics = [m for m in ['accuracy', 'minority_recall', 'minority_f1', 'f1_macro', 'f1_weighted'] if m in table]
    show(px.bar(table.melt(id_vars='candidate', value_vars=metrics, var_name='metric', value_name='score'),
                x='candidate', y='score', color='metric', barmode='group', range_y=[0, 1.02]))
    st.caption('Minority class is identified using each experiment’s training partition. Accuracy alone does not establish superiority.')
    if 'training_seconds' in table:
        show(px.bar(table, x='candidate', y='training_seconds', title='Final training time (seconds)'))
    cv = pd.DataFrame([{'candidate': f"{r['model_name']} · {r['id']}", 'score': score}
                       for r in filtered for score in r.get('cv_scores', [])])
    if not cv.empty:
        show(px.box(cv, x='candidate', y='score', points='all', title='Training cross-validation scores'))
        summary = table.dropna(subset=['cv_mean'])
        show(px.bar(summary, x='candidate', y='cv_mean', error_y='cv_std', title='CV mean ± standard deviation (descriptive)'))


def error_page():
    st.header('Error Analysis')
    result = choose_result()
    if result is None:
        return
    errors = result.get('errors')
    if errors is None:
        st.info('This import contains no row-level predictions.')
        return
    data = errors
    filter_type = st.selectbox('Outcome', ['All observations', 'Misclassified', 'Correct', 'False positives (binary)', 'False negatives (binary)'])
    positive = result.get('positive_class')
    if filter_type == 'Misclassified':
        data = data[~data._correct]
    elif filter_type == 'Correct':
        data = data[data._correct]
    elif 'binary' in filter_type:
        if len(result.get('classes', [])) != 2 or positive is None:
            st.info('Binary false-positive/negative filters require a locally trained binary result with a selected positive class.')
            return
        data = data[(data._actual != positive) & (data._predicted == positive)] if 'positives' in filter_type else \
            data[(data._actual == positive) & (data._predicted != positive)]
    actual = st.multiselect('Actual classes', sorted(errors._actual.unique()))
    predicted = st.multiselect('Predicted classes', sorted(errors._predicted.unique()))
    if actual:
        data = data[data._actual.isin(actual)]
    if predicted:
        data = data[data._predicted.isin(predicted)]
    st.metric('Matching observations', len(data))
    st.dataframe(data, hide_index=True, width='stretch')
    st.download_button('Download filtered observations', csv_bytes(data), 'error_analysis.csv', 'text/csv')
    confused = errors[~errors._correct].groupby(['_actual', '_predicted']).size().reset_index(name='count').sort_values('count', ascending=False)
    st.subheader('Frequently confused classes')
    st.dataframe(confused, hide_index=True, width='stretch')
    rates = errors.groupby('_actual')._correct.agg(['count', 'mean'])
    rates['error_rate'] = 1 - rates.pop('mean')
    st.dataframe(rates, width='stretch')
    if len(result.get('classes', [])) == 2 and positive:
        st.subheader(f'Event detection summary · positive class {positive}')
        st.caption('For fraud detection, select the fraud label as the positive class before training.')
        for label, mask in [('Event detected', (errors._actual == positive) & (errors._predicted == positive)),
                            ('Event missed', (errors._actual == positive) & (errors._predicted != positive)),
                            ('Non-event incorrectly flagged', (errors._actual != positive) & (errors._predicted == positive)),
                            ('Non-event correctly classified', (errors._actual != positive) & (errors._predicted != positive))]:
            st.write(f'{label}: **{mask.sum():,}**')


def prediction_page():
    st.header('Prediction')
    result = choose_result(trained_only=True)
    if result is None:
        return
    st.info('Prediction applies the exact fitted preprocessing and decision threshold. It never refits transformations.')
    features = result['config']['features']
    st.write('Required features:', ', '.join(features))
    with st.form('single_prediction'):
        record = {}
        pipeline = result['pipeline']
        numeric = next((cols for name, _, cols in pipeline.named_steps['preprocess'].transformers_ if name == 'numeric'), [])
        for feature in features:
            if feature in numeric:
                record[feature] = st.number_input(feature, value=0.0, key=f'predict_{feature}')
            else:
                record[feature] = st.text_area(feature, key=f'predict_{feature}')
        if st.form_submit_button('Predict record', type='primary'):
            try:
                st.dataframe(predict_batch(result, pd.DataFrame([record])), hide_index=True, width='stretch')
            except (ValueError, TypeError) as exc:
                st.error(str(exc))
    batch = st.file_uploader('Batch prediction CSV', type=['csv'], key='batch_upload')
    if batch is not None and st.button('Predict batch'):
        try:
            output = predict_batch(result, load_dataset(batch.getvalue(), batch.name))
            st.session_state.last_predictions = output
            st.session_state.last_prediction_model = result['id']
        except (ValueError, TypeError) as exc:
            st.error(str(exc))
    if st.session_state.get('last_prediction_model') == result['id']:
        output = st.session_state.last_predictions
        st.dataframe(output.head(1000), width='stretch')
        st.download_button('Download predictions', csv_bytes(output), 'predictions.csv', 'text/csv')

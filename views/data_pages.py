"""Dataset, exploratory, preprocessing, and imbalance interfaces."""
from dataclasses import asdict
from collections import Counter
import re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from core.config import PreprocessingConfig, ResamplingConfig, SCALERS
from core.data import load_dataset, audit_dataset, class_distribution, stratified_sample
from core.resampling import METHODS
from core.training import split_dataset
from core.preprocessing import build_pipeline
from core.fraud import is_creditcard_dataset, creditcard_preset
from utils.session import replace_dataset
from utils.charts import show, class_chart


def need_dataset():
    if st.session_state.dataset is None:
        st.info('Upload a classification dataset in Dataset Management to begin.')
        return False
    return True


def dataset_page():
    st.header('Dataset Management')
    st.info('Full creditcard.csv too large? Open Credit Card Fraud & Colab to train in Google Colab and import a small results JSON. '
            'You do not need to upload the full CSV here.')
    st.caption('Upload, audit, and select your classification inputs. Cleaning always requires an explicit choice.')
    upload = st.file_uploader('CSV, TSV or Excel (.xlsx)', type=['csv', 'tsv', 'xlsx'], key='dataset_upload')
    if upload is not None and st.button('Load this dataset', type='primary'):
        try:
            frame = load_dataset(upload.getvalue(), upload.name)
            replace_dataset(frame, upload.name)
            st.rerun()
        except Exception as exc:
            st.error(f'Could not load dataset: {exc}')
    if not need_dataset():
        return
    frame = st.session_state.dataset
    if is_creditcard_dataset(frame):
        st.success('Credit-card schema detected: Class is the target; 1 = fraud and 0 = legitimate.')
        if st.button('Apply credit-card fraud preset', type='primary'):
            try:
                st.session_state.update(creditcard_preset(frame))
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        st.caption('Preset: average-precision ranking, positive Class=1, StandardScaler, 3-fold CV, '
                   'class-weighted Random Forest/logistic regression and a Dummy baseline; no resampling or data removal.')
    st.subheader(st.session_state.dataset_name)
    st.dataframe(frame.head(100), width='stretch')
    st.caption(f'Preview shows up to 100 of {len(frame):,} rows. Training uses the selected full dataset unless you explicitly sample below.')
    target_default = st.session_state.target or ('Class' if is_creditcard_dataset(frame) else frame.columns[-1])
    with st.form('dataset_selection'):
        target = st.selectbox('Target column', frame.columns, index=list(frame.columns).index(target_default))
        task = st.selectbox('Classification task', ['Tabular', 'Mixed', 'Text'],
                            index=['Tabular', 'Mixed', 'Text'].index(st.session_state.prep_config.task),
                            help='Tabular and Mixed infer numerical/categorical column types; Text uses one TF-IDF column.')
        default_features = st.session_state.feature_selection or [c for c in frame if c != target_default]
        features = st.multiselect('Feature columns (the target is always excluded)', frame.columns,
                                  default=[c for c in default_features if c in frame])
        texts = list(frame.columns)
        text_column = st.selectbox('Text column (used only for Text classification)', texts,
                                   index=texts.index(st.session_state.text_column) if st.session_state.text_column in texts else 0)
        if st.form_submit_button('Apply dataset selection'):
            selected = [text_column] if task == 'Text' else [c for c in features if c != target]
            if not selected or target in selected:
                st.error('Select at least one feature distinct from the target.')
            else:
                st.session_state.target = target
                st.session_state.feature_selection = selected
                st.session_state.text_column = text_column if task == 'Text' else None
                st.session_state.prep_config.task = task
                st.session_state.active_ids = []
                st.rerun()
    audit = audit_dataset(frame, st.session_state.target)
    cols = st.columns(4)
    for col, label, value in zip(cols, ['Observations', 'Columns', 'Duplicate records', 'Memory (MB)'],
                               [audit['rows'], len(frame.columns), audit['duplicates'], f"{audit['memory_mb']:.2f}"]):
        col.metric(label, value)
    for message in audit['warnings']:
        st.warning(message)
    st.dataframe(audit['columns'], width='stretch')
    with st.expander('Descriptive statistics'):
        st.dataframe(frame.describe(include='all').T, width='stretch')
    with st.expander('Explicit stratified sampling'):
        st.write('Replaces the current training dataset with a reproducible sample. No automatic sampling occurs during training.')
        if st.session_state.target and len(frame) > 5:
            rows = st.number_input('Sample rows', min_value=2, max_value=len(frame) - 1,
                                   value=min(10000, len(frame) - 1), step=1)
            seed = st.number_input('Sampling seed', min_value=0, max_value=2147483647, value=42)
            if st.button('Replace dataset with stratified sample'):
                try:
                    sampled = stratified_sample(frame, st.session_state.target, rows, seed)
                    name = f'{st.session_state.dataset_name} · stratified sample n={rows}, seed={seed}'
                    replace_dataset(sampled, name)
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
        else:
            st.info('Apply a target selection first.')
    if st.button('Remove current dataset'):
        st.session_state.dataset = None
        st.session_state.active_ids = []
        st.rerun()


def eda_page():
    st.header('Exploratory Data Analysis')
    if not need_dataset():
        return
    frame = st.session_state.dataset
    sample = frame.sample(min(len(frame), 5000), random_state=42)
    st.caption(f'Visualizations use a reproducible sample of {len(sample):,}/{len(frame):,} rows (seed 42). Training data is unchanged.')
    target = st.session_state.target
    if target:
        distribution = class_distribution(frame[target])
        class_chart(distribution)
        show(px.bar(distribution, x='class', y='percentage', title='Class percentages', text_auto='.2f'))
    numerical = sample.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in sample if c not in numerical]
    if numerical:
        feature = st.selectbox('Numerical feature', numerical)
        left, right = st.columns(2)
        with left:
            show(px.histogram(sample, x=feature, color=target if target and frame[target].nunique() <= 15 else None,
                              nbins=40, title='Feature distribution'))
        with right:
            show(px.box(sample, y=feature, x=target if target and frame[target].nunique() <= 15 else None,
                        title='Outlier inspection (1.5 × IQR whiskers)'))
        st.dataframe(frame[numerical].describe().T, width='stretch')
        if len(numerical) >= 2:
            pair = st.multiselect('Scatter plot axes (select two)', numerical, default=numerical[:2], max_selections=2)
            if len(pair) == 2:
                show(px.scatter(sample, x=pair[0], y=pair[1], color=target if target else None, opacity=0.6))
            chosen = st.multiselect('Correlation features (up to 20)', numerical, default=numerical[:min(6, len(numerical))], max_selections=20)
            if len(chosen) >= 2:
                show(px.imshow(sample[chosen].corr(), text_auto='.2f', zmin=-1, zmax=1, color_continuous_scale='RdBu_r', title='Pearson correlation'))
    if categorical:
        feature = st.selectbox('Categorical/text feature', categorical)
        frequency = sample[feature].fillna('(missing)').astype(str).value_counts().head(25).rename_axis('value').reset_index(name='count')
        show(px.bar(frequency, x='value', y='count', title='Top 25 observed values'))
    missing = frame.isna().sum().rename_axis('feature').reset_index(name='missing')
    show(px.bar(missing[missing.missing > 0].head(50), x='feature', y='missing', title='Missing-value counts (up to 50 columns)'))
    if st.session_state.prep_config.task == 'Text' and st.session_state.text_column:
        column = st.session_state.text_column
        texts = sample[column].fillna('').astype(str)
        lengths = pd.DataFrame({'characters': texts.str.len(), 'words': texts.str.split().str.len()})
        if target:
            lengths['class'] = sample[target].astype(str).to_numpy()
        show(px.histogram(lengths, x='characters', color='class' if target else None, nbins=40, title='Text lengths by class'))
        show(px.histogram(lengths, x='words', nbins=40, title='Word counts'))
        # Bound text tokenization work separately from uploaded/training data.
        tokens = re.findall(r'\b\w+\b', ' '.join(texts.str.slice(0, 2000)).lower())
        st.metric('Empty texts (complete dataset)', int(frame[column].fillna('').astype(str).str.strip().eq('').sum()))
        st.metric('Observed vocabulary (visualization sample; first 2,000 characters/record)', len(set(tokens)))
        st.dataframe(pd.DataFrame(Counter(tokens).most_common(30), columns=['word', 'count']), width='stretch')


def preprocessing_page(config):
    st.header('Data Preprocessing')
    st.write('Imputers, encoders, scalers, and TF-IDF learn only from training observations, including inside every CV fold.')
    p = st.session_state.prep_config
    with st.form('preprocessing'):
        left, right = st.columns(2)
        with left:
            imputation = st.selectbox('Numerical missing-value strategy', ['median', 'mean', 'most_frequent', 'constant'],
                                      index=['median', 'mean', 'most_frequent', 'constant'].index(p.imputation))
            fill_value = st.number_input('Numerical constant fill value', value=p.fill_value)
            encoding = st.selectbox('Categorical encoding', ['onehot', 'ordinal'], index=['onehot', 'ordinal'].index(p.encoding))
            scaling = st.selectbox('Numerical scaling', list(SCALERS), index=list(SCALERS).index(p.scaling))
        with right:
            drop_missing_target = st.checkbox('Remove rows with missing target labels', value=p.drop_missing_target)
            drop_missing_features = st.checkbox('Remove rows with incomplete selected features', value=p.drop_missing_features)
            drop_duplicates = st.checkbox('Remove duplicate selected records before splitting', value=p.drop_duplicates)
            st.caption('Categorical missing values use the most frequent training category, or “Missing” for constant imputation.')
        st.subheader('Text preprocessing / TF-IDF')
        cols = st.columns(3)
        with cols[0]:
            lowercase = st.checkbox('Lowercase', value=p.lowercase)
            punctuation = st.checkbox('Remove punctuation', value=p.remove_punctuation)
            numbers = st.checkbox('Remove numbers', value=p.remove_numbers)
        with cols[1]:
            whitespace = st.checkbox('Normalize whitespace', value=p.normalize_whitespace)
            stopwords = st.checkbox('Remove English stop words (preserve no/not/nor/never)', value=p.remove_stopwords)
            sublinear = st.checkbox('Sublinear term frequency', value=p.sublinear_tf)
        with cols[2]:
            ngram = st.select_slider('Maximum n-gram size', options=[1, 2, 3], value=p.ngram_max)
            minimum = st.number_input('min_df (document count)', min_value=1, max_value=10000, value=p.min_df)
            maximum = st.number_input('max_df (document fraction)', min_value=0.01, max_value=1.0, value=p.max_df)
        features = st.number_input('Maximum TF-IDF features', min_value=100, max_value=30000, value=p.max_features, step=100)
        norms = ['l2', 'l1', 'none']
        norm = st.selectbox('TF-IDF normalization', norms, index=norms.index(p.norm or 'none'))
        if st.form_submit_button('Save preprocessing', type='primary'):
            st.session_state.prep_config = PreprocessingConfig(
                task=p.task, imputation=imputation, fill_value=fill_value, encoding=encoding, scaling=scaling,
                drop_missing_features=drop_missing_features, drop_missing_target=drop_missing_target,
                drop_duplicates=drop_duplicates, lowercase=lowercase, remove_punctuation=punctuation,
                remove_numbers=numbers, normalize_whitespace=whitespace, remove_stopwords=stopwords,
                ngram_max=ngram, min_df=minimum, max_df=maximum, max_features=features,
                sublinear_tf=sublinear, norm=None if norm == 'none' else norm)
            st.rerun()
    st.info(SCALERS[p.scaling])
    if p.encoding == 'ordinal':
        st.warning('Ordinal encoding introduces artificial ordering for nominal categories. Use only when this ordering is justified; one-hot is the default.')
    if p.task == 'Text':
        st.caption('Text uses TF-IDF normalization rather than the numerical scaling setting. Negation is preserved by the default English stop-word filter.')
    st.dataframe(pd.DataFrame([
        ['MLP / SVM / KNN', 'Strongly recommended'], ['Logistic Regression', 'Recommended'],
        ['Random Forest / Decision Tree / Gradient Boosting', 'Generally unnecessary'],
        ['Gaussian NB', 'Optional'], ['Multinomial NB', 'Nonnegative TF-IDF only'],
    ], columns=['Model', 'Scaling recommendation']), hide_index=True, width='stretch')
    numerical_choices = st.session_state.dataset[config.features].select_dtypes(include=np.number).columns.tolist() \
        if config and p.task != 'Text' else []
    preview_feature = st.selectbox('Scaling preview feature', numerical_choices) if numerical_choices else None
    if config and p.task != 'Text' and st.button('Preview train-only scaling'):
        try:
            from core.data import prepare_dataset
            X, y, _ = prepare_dataset(st.session_state.dataset, config)
            train, _ = split_dataset(X, y, config.split)
            numerical = X.select_dtypes(include=np.number).columns.tolist()
            if not numerical:
                st.info('There are no numerical features to scale.')
            else:
                pipeline = build_pipeline(X.iloc[train], config, 'Dummy baseline')
                # Preview fits only the preprocessing transformer on training rows.
                pipeline.named_steps['preprocess'].fit(X.iloc[train], y.iloc[train])
                transform = pipeline.named_steps['preprocess'].named_transformers_['numeric']
                subset = X.iloc[train[:min(2000, len(train))]][numerical]
                after = transform.transform(subset)
                feature = preview_feature or numerical[0]
                table = pd.DataFrame({'before': subset[feature].to_numpy(), 'after': after[:, numerical.index(feature)]})
                show(px.histogram(table.melt(var_name='stage', value_name='value'), x='value', color='stage',
                                  facet_col='stage', title=f'{feature} · training-only sample'))
                st.dataframe(pd.DataFrame({'feature': numerical, 'mean_before': subset.mean().to_numpy(),
                                          'mean_after': np.nanmean(after, axis=0)}), width='stretch')
        except ValueError as exc:
            st.error(str(exc))


def imbalance_page():
    st.header('Class Imbalance Analysis')
    if not need_dataset() or not st.session_state.target:
        st.info('Apply a target selection in Dataset Management.')
        return
    frame = st.session_state.dataset
    distribution = class_distribution(frame[st.session_state.target])
    if distribution.empty:
        st.warning('No nonmissing target labels.')
        return
    counts = distribution.set_index('class')['count']
    ratio = counts.max() / counts.min()
    cols = st.columns(4)
    cols[0].metric('Observations with labels', int(counts.sum()))
    cols[1].metric('Classes', len(counts))
    cols[2].metric('Majority / minority ratio', f'{ratio:.2f}')
    cols[3].metric('Majority baseline accuracy', f'{counts.max() / counts.sum():.1%}')
    st.write(f'Majority class: **{counts.idxmax()}** · Minority class: **{counts.idxmin()}** ({counts.min():,} observations).')
    moderate = st.number_input('Moderate-imbalance heuristic threshold', min_value=1.1, max_value=100.0, value=3.0)
    severe = st.number_input('Severe-imbalance heuristic threshold', min_value=moderate, max_value=1000.0, value=max(10.0, moderate))
    st.info(f'Heuristic severity: {"severe" if ratio >= severe else "moderate" if ratio >= moderate else "low"}. '
            'These configurable thresholds are practical heuristics, not universal scientific standards.')
    class_chart(distribution)
    show(px.pie(distribution, names='class', values='count', title='Class percentage distribution'))
    st.dataframe(distribution, hide_index=True, width='stretch')
    if counts.min() < 6:
        st.warning('Minority sample availability is very low. Reduce CV folds and SMOTE neighbors, or collect more examples.')
    st.warning('High accuracy can hide missed minority examples. Prioritize macro F1, balanced accuracy, per-class recall and precision–recall analysis.')
    r = st.session_state.resample_config
    with st.form('resampling'):
        method = st.selectbox('Training-only resampling', METHODS, index=METHODS.index(r.method))
        choice = st.selectbox('Sampling strategy', ['auto', 'minority', 'not majority', 'all', 'binary ratio'],
                              index=['auto', 'minority', 'not majority', 'all', 'binary ratio'].index(r.strategy if isinstance(r.strategy, str) else 'binary ratio'))
        ratio_value = st.number_input('Binary minority / majority target ratio', min_value=0.01, max_value=1.0,
                                      value=float(r.strategy) if not isinstance(r.strategy, str) else 1.0)
        neighbors = st.number_input('SMOTE k_neighbors', min_value=1, max_value=20, value=r.k_neighbors)
        if st.form_submit_button('Save imbalance handling', type='primary'):
            st.session_state.resample_config = ResamplingConfig(method, ratio_value if choice == 'binary ratio' else choice, neighbors)
            st.rerun()
    st.caption('Random under/oversampling support sparse and categorical encodings. SMOTENC supports mixed numerical/categorical input. '
               'SMOTE requires more than k minority samples in every training fold. Unsupported strategies fail explicitly.')
    st.caption('For TF-IDF, SMOTE synthesizes feature vectors, not text documents; synthetic high-dimensional points may lack linguistic meaning.')
    active = [r for r in st.session_state.history if r['id'] in st.session_state.active_ids]
    if active:
        selected = st.selectbox('Inspect training distributions', range(len(active)),
                                format_func=lambda i: f"{active[i]['model_name']} · {active[i]['config']['resampling']['method']}")
        result = active[selected]
        for name, table in result['distribution'].items():
            st.subheader(name.replace('_', ' ').title())
            st.dataframe(table, hide_index=True, width='stretch')
            class_chart(table, name.replace('_', ' ').title())
            st.caption(f"Majority / minority ratio: {table['count'].max() / table['count'].min():.2f}")
        st.metric('Net synthetic / removed training observations', result['resampled_delta'])
        st.caption(f"Generated training observations: {result.get('generated_rows', 0):,} · "
                   f"Removed training observations: {result.get('removed_rows', 0):,}")

"""Shared holdout experiments with fold-local fitting and training-only model selection."""
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import time
import uuid
import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from threadpoolctl import threadpool_limits
from core.data import dataset_fingerprint, prepare_dataset, class_distribution
from core.evaluation import model_response, positive_scores, evaluate_predictions, threshold_analysis
from core.preprocessing import build_pipeline


PACKAGES = ['streamlit', 'pandas', 'numpy', 'scikit-learn', 'imbalanced-learn', 'scipy', 'plotly',
            'matplotlib', 'seaborn', 'joblib', 'openpyxl']


def split_dataset(X, y, config):
    if not 0.1 <= config.test_size <= 0.4:
        raise ValueError('Holdout fraction must be between 10% and 40%.')
    train, test = train_test_split(np.arange(len(X)), test_size=config.test_size,
                                  random_state=config.random_state, stratify=y)
    classes = set(y)
    if set(y.iloc[train]) != classes or set(y.iloc[test]) != classes:
        raise ValueError('Every class must appear in both training and test sets. Change the test fraction or collect more data.')
    if not 2 <= config.folds <= 10 or not 1 <= config.repeats <= 3:
        raise ValueError('Use 2–10 folds and 1–3 repeats.')
    if y.iloc[train].value_counts().min() < config.folds:
        raise ValueError('A class in the training partition has fewer observations than CV folds. '
                         'Reduce folds, adjust test size, or collect more examples.')
    return train, test


def train_experiments(frame, config, models, strategies=None, progress=None):
    """Fit sequentially; failed candidates are reported rather than replaced or hidden.

    The holdout split is constructed once for every candidate. CV sees only its training partition.
    """
    X, y, audit_messages = prepare_dataset(frame, config)
    train, test = split_dataset(X, y, config.split)
    strategies = strategies or [config.resampling.method]
    if not models or len(models) > 6 or len(strategies) > 3:
        raise ValueError('Select 1–6 classifiers and at most 3 strategies per run.')
    fit_count = len(models) * len(strategies) * (config.split.folds * config.split.repeats + 1)
    if fit_count > 120:
        raise ValueError(f'This run requires {fit_count} fits; the cloud limit is 120. Reduce models, folds, repeats or strategies.')
    positive = config.positive_class or sorted(y.unique())[-1]
    if positive not in set(y):
        raise ValueError('Selected positive class is absent from the dataset.')
    binary_metrics = {'f1_positive', 'recall_positive', 'average_precision'}
    if y.nunique() != 2 and (config.primary_metric in binary_metrics or config.optimize_threshold):
        raise ValueError('Positive-class metrics and threshold optimization require binary classification.')
    fingerprint = dataset_fingerprint(frame)
    split_id = hashlib.sha256(json.dumps([fingerprint, config.target, X.index[test].tolist()]).encode()).hexdigest()[:16]
    batch_id = uuid.uuid4().hex[:12]
    results, failures = [], []
    completed = 0
    cv = RepeatedStratifiedKFold(n_splits=config.split.folds, n_repeats=config.split.repeats,
                                 random_state=config.split.random_state)
    X_train, y_train = X.iloc[train], y.iloc[train]
    X_test, y_test = X.iloc[test], y.iloc[test]
    for strategy in strategies:
        current = replace(config, resampling=replace(config.resampling, method=strategy))
        for name, parameters in models.items():
            started = time.perf_counter()
            label = f'{name} · {strategy}'
            if progress:
                progress(completed / fit_count, label, 'Validating configuration')
            try:
                if name in ('SVM', 'KNN') and len(X_train) > 10000:
                    raise ValueError(f'{name} is limited to 10,000 training rows. Use Linear SVM, a smaller explicit sample, or Colab.')
                pipeline = build_pipeline(X_train, current, name, parameters)
                classes = sorted(y.unique())
                cv_scores, cv_rows, oof_scores = [], [], np.zeros(len(train), dtype=float)
                oof_counts = np.zeros(len(train), dtype=int)
                captured = []
                with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as log:
                    warnings.simplefilter('always')
                    for fold, (fit_index, validation_index) in enumerate(cv.split(X_train, y_train)):
                        fitted = clone(pipeline)
                        fitted.fit(X_train.iloc[fit_index], y_train.iloc[fit_index])
                        predictions = fitted.predict(X_train.iloc[validation_index])
                        response, kind = model_response(fitted, X_train.iloc[validation_index])
                        evaluated = evaluate_predictions(y_train.iloc[validation_index], predictions,
                                                         classes, response, kind, positive)
                        metric = evaluated['metrics'].get(config.primary_metric)
                        if metric is None or not np.isfinite(metric):
                            raise ValueError(f'Primary metric {config.primary_metric} is not applicable to this classifier.')
                        cv_scores.append(float(metric))
                        cv_rows.append({'fold': fold % config.split.folds + 1,
                                        'repeat': fold // config.split.folds + 1, **evaluated['metrics']})
                        if config.optimize_threshold:
                            scores = positive_scores(response, classes, positive)
                            if scores is None:
                                raise ValueError('Threshold optimization requires probabilities or decision scores.')
                            oof_scores[validation_index] += scores
                            oof_counts[validation_index] += 1
                        completed += 1
                        if progress:
                            progress(completed / fit_count, label, f'CV fold {fold + 1}/{config.split.folds * config.split.repeats}')
                        del fitted
                    selected_threshold, threshold_table = None, None
                    if config.optimize_threshold:
                        selected_threshold, threshold_table = threshold_analysis(
                            y_train, oof_scores / oof_counts, positive, kind)
                    final_started = time.perf_counter()
                    pipeline.fit(X_train, y_train)
                    fit_seconds = time.perf_counter() - final_started
                    predictions = pipeline.predict(X_test)
                    response, kind = model_response(pipeline, X_test)
                    if selected_threshold is not None:
                        negative = next(c for c in classes if c != positive)
                        predictions = np.where(positive_scores(response, classes, positive) >= selected_threshold,
                                               positive, negative)
                    evaluated = evaluate_predictions(y_test, predictions, classes, response, kind, positive)
                    captured = list(dict.fromkeys(str(item.message) for item in log))
                sampler = pipeline.named_steps.get('resample')
                distribution = {
                    'dataset': class_distribution(y), 'training': class_distribution(y_train),
                    'resampled_training': class_distribution(y_train) if sampler is None else pd.DataFrame(
                        {'class': list(sampler.after_counts_), 'count': list(sampler.after_counts_.values())}),
                    'test': class_distribution(y_test),
                }
                minority = y_train.value_counts().idxmin()
                evaluated['metrics']['minority_recall'] = float(evaluated['per_class'].loc[minority, 'recall'])
                evaluated['metrics']['minority_f1'] = float(evaluated['per_class'].loc[minority, 'f1-score'])
                errors = X_test.rename(columns={c: f'feature::{c}' for c in X_test if c.startswith('_')}).copy()
                errors.insert(0, '_row_id', X_test.index)
                errors['_actual'] = y_test.to_numpy()
                errors['_predicted'] = predictions
                errors['_correct'] = np.asarray(y_test) == predictions
                if kind == 'probability':
                    errors['_confidence'] = response.max(axis=1)
                results.append({
                    'id': uuid.uuid4().hex[:12], 'batch_id': batch_id, 'source': 'Streamlit training',
                    'timestamp': datetime.now(timezone.utc).isoformat(), 'model_name': name,
                    'experiment_name': config.name, 'config': current.to_dict(), 'model_parameters': parameters,
                    'dataset_fingerprint': fingerprint, 'dataset_shape': list(frame.shape),
                    'split_id': split_id, 'train_indices': X.index[train].tolist(), 'test_indices': X.index[test].tolist(),
                    'classes': classes, 'positive_class': positive, 'minority_class': minority,
                    'pipeline': pipeline, 'evaluation': evaluated, 'errors': errors, 'distribution': distribution,
                    'cv_scores': cv_scores, 'cv_results': pd.DataFrame(cv_rows),
                    'cv_mean': float(np.mean(cv_scores)), 'cv_std': float(np.std(cv_scores)),
                    'fit_seconds': fit_seconds, 'total_seconds': time.perf_counter() - started,
                    'threshold': selected_threshold, 'threshold_table': threshold_table,
                    'resampled_delta': 0 if sampler is None else sampler.delta_,
                    'generated_rows': 0 if sampler is None else sampler.generated_,
                    'removed_rows': 0 if sampler is None else sampler.removed_,
                    'warnings': audit_messages + captured,
                    'versions': {package: importlib.metadata.version(package) for package in PACKAGES},
                })
                completed += 1
            except (ValueError, TypeError, MemoryError, RuntimeError) as exc:
                failures.append({'model': name, 'strategy': strategy, 'error': str(exc)})
                completed += config.split.folds * config.split.repeats + 1 - (completed % (config.split.folds * config.split.repeats + 1))
            if progress:
                progress(min(completed / fit_count, 1.0), label, 'Completed' if not failures or failures[-1]['model'] != name else 'Finished candidate')
    return results, failures


def comparison_table(results):
    rows = []
    for result in results:
        config = result.get('config', {})
        rows.append({'id': result['id'], 'model': result['model_name'], 'experiment': result.get('experiment_name', ''),
                     'source': result['source'], 'split_id': result.get('split_id', 'unknown'),
                     'primary_metric': config.get('primary_metric', 'unknown'),
                     'scaling': config.get('preprocessing', {}).get('scaling', 'unknown'),
                     'resampling': config.get('resampling', {}).get('method', 'unknown'),
                     **result['evaluation']['metrics'], 'cv_mean': result.get('cv_mean'),
                     'cv_std': result.get('cv_std'), 'training_seconds': result.get('fit_seconds')})
    return pd.DataFrame(rows)

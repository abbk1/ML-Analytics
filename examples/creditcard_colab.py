"""Standalone credit-card fraud training for Colab; no Streamlit dependency.

Batch mode compares ANN (MLP), Random Forest and a baseline using validation AP.
Incremental mode reads the complete CSV in chunks with a frozen training-only
scaler and SGD logistic classifier. Neither mode samples the input implicitly.
The generated notebook embeds this exact module so no repository clone is needed.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.metadata
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
    precision_recall_curve, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits


@dataclass
class FraudColabConfig:
    csv_path: str = '/content/drive/MyDrive/creditcard.csv'
    output_dir: str = '/content/drive/MyDrive/fraud_results'
    mode: str = 'batch'                    # batch or incremental
    target: str = 'Class'
    time_column: str = 'Time'
    split_method: str = 'chronological'    # chronological or stratified (batch only)
    models: tuple = ('Random Forest', 'MLP', 'Dummy baseline')
    seed: int = 42
    chunk_size: int = 50000
    epochs: int = 3                       # incremental passes, each visits every training row
    memory_budget_mb: int = 1024          # batch raw-feature budget; estimator overhead is extra
    max_evaluation_rows: int = 1000000     # both validation and test; no silent evaluation sampling
    min_precision: float = 0.10           # validation threshold constraint, user/business choice
    beta: float = 2.0                     # F2 gives fraud recall more weight than precision
    rf_estimators: int = 80
    mlp_iterations: int = 40
    export_model: bool = True


def _validate(config):
    if config.mode not in ('batch', 'incremental') or config.split_method not in ('chronological', 'stratified'):
        raise ValueError('Choose batch/incremental and chronological/stratified splitting.')
    if config.mode == 'incremental' and config.split_method != 'chronological':
        raise ValueError('Incremental mode requires a CSV sorted by Time and chronological splitting.')
    if not 1 <= config.chunk_size <= 100000 or not 1 <= config.epochs <= 10:
        raise ValueError('Use chunks of 1–100,000 rows and 1–10 incremental epochs.')
    if not 0 <= config.min_precision <= 1 or not 0 < config.beta <= 5:
        raise ValueError('min_precision must be in [0,1] and beta in (0,5].')
    if not 1 <= config.rf_estimators <= 300 or not 1 <= config.mlp_iterations <= 300:
        raise ValueError('Limit forest estimators and MLP iterations to 1–300.')
    if not 1 <= config.memory_budget_mb <= 16384 or not 10 <= config.max_evaluation_rows <= 10000000:
        raise ValueError('Invalid memory or evaluation-row budget.')
    if not config.models or set(config.models) - {'Random Forest', 'MLP', 'SGD logistic', 'Dummy baseline'}:
        raise ValueError('Supported batch models: Random Forest, MLP, SGD logistic, Dummy baseline.')


def _chunks(config, features=None):
    header = pd.read_csv(config.csv_path, nrows=0).columns.tolist()
    if config.target not in header or len(header) != len(set(header)):
        raise ValueError('CSV needs unique column names and the Class target.')
    features = features or [c for c in header if c != config.target]
    if not features or len(features) > 100:
        raise ValueError('Use 1–100 numerical transaction features.')
    for frame in pd.read_csv(config.csv_path, chunksize=config.chunk_size):
        numeric = frame[features].apply(pd.to_numeric, errors='raise')
        X = numeric.to_numpy(dtype=np.float32, na_value=np.nan)
        if np.isinf(X).any():
            raise ValueError('Infinite feature values are unsupported; correct the CSV first.')
        labels = pd.to_numeric(frame[config.target], errors='raise')
        if labels.isna().any() or not labels.isin([0, 1]).all():
            raise ValueError('Class must contain only 0 (legitimate) and 1 (fraud), without missing targets.')
        yield X, labels.to_numpy(dtype=np.int8), features


def _require_classes(y, partition):
    if set(np.unique(y)) != {0, 1}:
        raise ValueError(f'{partition} needs legitimate and fraud examples. Use a larger valid time window; '
                         'do not silently shuffle future transactions into training.')


def threshold_on_validation(y, probabilities, min_precision=0.1, beta=2.0):
    """Select F-beta under an explicit precision floor, using validation only."""
    _require_classes(y, 'Validation')
    precision, recall, thresholds = precision_recall_curve(y, probabilities)
    p, r = precision[:-1], recall[:-1]
    denominator = beta**2 * p + r
    fbeta = np.divide((1 + beta**2) * p * r, denominator,
                      out=np.zeros_like(p), where=denominator != 0)
    eligible = np.flatnonzero((p >= min_precision) & (r > 0))
    if not len(eligible):
        raise ValueError('No validation threshold meets the precision floor while detecting fraud. '
                         'Review the constraint/model; the test data has not been used to choose a threshold.')
    best = eligible[np.argmax(fbeta[eligible])]
    return float(thresholds[best]), {'precision': float(p[best]), 'recall': float(r[best]),
                                   'f_beta': float(fbeta[best]), 'beta': beta,
                                   'min_precision': min_precision}


def _response(pipeline, X):
    classes = list(pipeline.classes_)
    return pipeline.predict_proba(X)[:, classes.index(1)]


def _compact_curve(x, y):
    indices = np.unique(np.linspace(0, len(x) - 1, min(len(x), 400), dtype=int))
    return np.asarray(x)[indices].tolist(), np.asarray(y)[indices].tolist()


def _evaluate(y, probabilities, threshold):
    _require_classes(y, 'Holdout test')
    predicted = (probabilities >= threshold).astype(np.int8)
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    values = {'accuracy': accuracy_score(y, predicted), 'balanced_accuracy': balanced_accuracy_score(y, predicted),
              'f1_macro': f1_score(y, predicted, average='macro', zero_division=0),
              'f1_positive': f1_score(y, predicted, zero_division=0),
              'recall_positive': recall_score(y, predicted, zero_division=0),
              'specificity': float(tn / (tn + fp)), 'false_positive_rate': float(fp / (tn + fp)),
              'false_negative_rate': float(fn / (fn + tp)),
              'average_precision': average_precision_score(y, probabilities), 'roc_auc': roc_auc_score(y, probabilities)}
    fpr, tpr, _ = roc_curve(y, probabilities)
    precision, recall, _ = precision_recall_curve(y, probabilities)
    fpr, tpr = _compact_curve(fpr, tpr)
    recall, precision = _compact_curve(recall, precision)
    return values, matrix, {'1': {'fpr': fpr, 'tpr': tpr, 'precision': precision, 'recall': recall}}, {
        'fraud_detected': int(tp), 'fraud_missed': int(fn),
        'legitimate_flagged': int(fp), 'legitimate_cleared': int(tn)}


def _batch(config):
    blocks, labels, size = [], [], 0
    for X, y, features in _chunks(config):
        size += X.nbytes + y.nbytes
        # Concatenation, splitting and preprocessing make multiple copies.
        if size * 6 > config.memory_budget_mb * 1024**2:
            raise MemoryError('Estimated batch arrays exceed the memory budget. Set mode="incremental" '
                              'with a Time-sorted CSV, or explicitly increase the budget on a suitable runtime.')
        blocks.append(X)
        labels.append(y)
    if not blocks:
        raise ValueError('CSV is empty.')
    X, y = np.concatenate(blocks), np.concatenate(labels)
    del blocks, labels
    rows = np.arange(len(y))
    if config.split_method == 'chronological':
        if config.time_column not in features:
            raise ValueError('Chronological splitting needs Time. Use a valid time column or explicitly choose stratified research splitting.')
        times = X[:, features.index(config.time_column)]
        if not np.isfinite(times).all():
            raise ValueError('Time must be finite and nonmissing.')
        rows = rows[np.argsort(times, kind='stable')]
        # Equal timestamps belong to one partition so boundary ties cannot leak forward.
        sorted_times = times[rows]
        cut1 = int(np.searchsorted(sorted_times, sorted_times[int(len(rows) * 0.6)], side='left'))
        cut2 = int(np.searchsorted(sorted_times, sorted_times[int(len(rows) * 0.8)], side='left'))
        train, validation, test = rows[:cut1], rows[cut1:cut2], rows[cut2:]
    else:
        train, rest = train_test_split(rows, test_size=0.4, stratify=y, random_state=config.seed)
        validation, test = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=config.seed)
    for name, index in [('Training', train), ('Validation', validation), ('Holdout test', test)]:
        _require_classes(y[index], name)
    if max(len(validation), len(test)) > config.max_evaluation_rows:
        raise MemoryError('Evaluation exceeds max_evaluation_rows. Increase the explicit budget on a suitable runtime.')
    best, comparison, captured = None, [], []
    weights = compute_sample_weight('balanced', y[train])
    for name in config.models:
        if name == 'Random Forest':
            model = RandomForestClassifier(n_estimators=config.rf_estimators, max_depth=12,
                min_samples_leaf=2, class_weight='balanced_subsample', n_jobs=2, random_state=config.seed)
        elif name == 'MLP':
            model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=config.mlp_iterations,
                batch_size=min(256, len(train)), early_stopping=False, random_state=config.seed)
        elif name == 'SGD logistic':
            model = SGDClassifier(loss='log_loss', class_weight='balanced', random_state=config.seed, max_iter=1000)
        else:
            model = DummyClassifier(strategy='prior')
        pipeline = Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                             ('scale', 'passthrough' if name in ('Random Forest', 'Dummy baseline') else StandardScaler()),
                             ('model', model)])
        started = time.perf_counter()
        with threadpool_limits(limits=2), warnings.catch_warnings(record=True) as log:
            warnings.simplefilter('always')
            kwargs = {'model__sample_weight': weights} if name == 'MLP' else {}
            pipeline.fit(X[train], y[train], **kwargs)
        captured.extend(f'{name}: {item.message}' for item in log)
        probabilities = _response(pipeline, X[validation])
        ap = float(average_precision_score(y[validation], probabilities))
        comparison.append({'model': name, 'validation_average_precision': ap,
                           'fit_seconds': time.perf_counter() - started})
        print(f'{name}: validation AP={ap:.5f}; fit {comparison[-1]["fit_seconds"]:.1f}s')
        if best is None or ap > best[0]:
            best = (ap, name, pipeline, probabilities)
    ap, name, pipeline, validation_probabilities = best
    threshold, validation_metrics = threshold_on_validation(y[validation], validation_probabilities,
                                                            config.min_precision, config.beta)
    # The selected model is not refitted after threshold selection; scaling and score scale stay fixed.
    test_probabilities = _response(pipeline, X[test])
    counts = {part: np.bincount(y[index], minlength=2).tolist() for part, index in
              [('training', train), ('validation', validation), ('test', test)]}
    return pipeline, name, y[test], test_probabilities, threshold, validation_metrics, comparison, {
        'partition_counts': counts, 'total_rows': len(y), 'features': features,
        'split_method': config.split_method, 'training_rows_used': len(train),
        'training_class_weights': 'none' if name == 'Dummy baseline' else
                                  'balanced_subsample' if name == 'Random Forest' else 'balanced',
        'resampling': 'none',
        'validation_average_precision': ap, 'warnings': captured}


def _incremental(config):
    """Pass 1 audits order/counts, pass 2 fits scaler, subsequent passes train.

    All passes use the same immutable row ranges. Validation and test rows never
    participate in scaler statistics, class weights, or SGD updates.
    """
    total, previous = 0, -np.inf
    for X, y, features in _chunks(config):
        if config.time_column not in features:
            raise ValueError('Incremental mode requires the Time column and a CSV sorted by Time.')
        times = X[:, features.index(config.time_column)]
        if not np.isfinite(times).all() or (np.diff(times) < 0).any() or (len(times) and times[0] < previous):
            raise ValueError('Incremental CSV must be sorted by nondecreasing Time. Sort externally or use batch mode; no automatic shuffling occurs.')
        previous = float(times[-1])
        total += len(y)
    if total < 10:
        raise ValueError('At least ten transactions are required.')
    # Locate timestamp boundaries without loading the CSV. Ties move to the later partition.
    cuts, position = {}, 0
    for X, y, _ in _chunks(config, features):
        times = X[:, features.index(config.time_column)]
        for part, row in [('validation', int(total * 0.6)), ('test', int(total * 0.8))]:
            if position <= row < position + len(y):
                cuts[part] = float(times[row - position])
        position += len(y)
    counts = {part: np.zeros(2, dtype=np.int64) for part in ['training', 'validation', 'test']}
    scale = StandardScaler()
    def partitions(X):
        t = X[:, features.index(config.time_column)]
        return {'training': t < cuts['validation'], 'validation': (t >= cuts['validation']) & (t < cuts['test']),
                'test': t >= cuts['test']}
    for X, y, _ in _chunks(config, features):
        masks = partitions(X)
        for part, mask in masks.items():
            counts[part] += np.bincount(y[mask], minlength=2)
        if masks['training'].any():
            # Fixed zero imputation is explicit; streaming does not approximate a global median.
            scale.partial_fit(np.nan_to_num(X[masks['training']], nan=0.0))
    for part, value in counts.items():
        if (value == 0).any():
            raise ValueError(f'{part} needs both classes. Choose an adequate time window.')
    if max(counts['validation'].sum(), counts['test'].sum()) > config.max_evaluation_rows:
        raise MemoryError('Evaluation arrays exceed max_evaluation_rows. Increase the explicit budget on a suitable runtime.')
    train_counts = counts['training']
    class_weights = {i: float(train_counts.sum() / (2 * train_counts[i])) for i in [0, 1]}
    classifier = SGDClassifier(loss='log_loss', random_state=config.seed, learning_rate='optimal', alpha=0.0001)
    with threadpool_limits(limits=2):
        for epoch in range(config.epochs):
            seen = 0
            for X, y, _ in _chunks(config, features):
                mask = partitions(X)['training']
                if not mask.any():
                    continue
                values = scale.transform(np.nan_to_num(X[mask], nan=0.0))
                target = y[mask]
                classifier.partial_fit(values, target, classes=np.array([0, 1]),
                    sample_weight=np.array([class_weights[int(label)] for label in target], dtype=np.float32))
                seen += len(target)
            print(f'Incremental epoch {epoch + 1}/{config.epochs}: {seen:,} training transactions; no row sampling.')
    imputer = SimpleImputer(strategy='constant', fill_value=0, keep_empty_features=True).fit(np.zeros((1, len(features))))
    pipeline = Pipeline([('impute', imputer), ('scale', scale), ('model', classifier)])
    def responses(part):
        ys, probabilities = [], []
        for X, y, _ in _chunks(config, features):
            mask = partitions(X)[part]
            if mask.any():
                ys.append(y[mask])
                probabilities.append(_response(pipeline, X[mask]))
        return np.concatenate(ys), np.concatenate(probabilities)
    validation_y, validation_probabilities = responses('validation')
    threshold, validation_metrics = threshold_on_validation(validation_y, validation_probabilities,
                                                            config.min_precision, config.beta)
    ap = float(average_precision_score(validation_y, validation_probabilities))
    test_y, test_probabilities = responses('test')
    return pipeline, 'SGD logistic (incremental)', test_y, test_probabilities, threshold, validation_metrics, [
        {'model': 'SGD logistic (incremental)', 'validation_average_precision': ap}], {
        'partition_counts': {key: value.tolist() for key, value in counts.items()},
        'total_rows': total, 'features': features, 'split_method': 'chronological',
        'training_rows_used': int(train_counts.sum()), 'epochs': config.epochs,
        'training_class_weights': class_weights, 'resampling': 'none',
        'validation_average_precision': ap,
        'warnings': ['Incremental learning uses fixed zero imputation and SGD logistic, not ANN or Random Forest.']}


def run_fraud_experiment(config):
    """Write compact, schema-compatible results and a trusted local model artifact."""
    _validate(config)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'fraud_results.json').exists():
        raise FileExistsError('Choose a new output_dir to preserve the previous experiment.')
    digest = hashlib.sha256()
    with Path(config.csv_path).open('rb') as source:
        for block in iter(lambda: source.read(1024**2), b''):
            digest.update(block)
    started = time.perf_counter()
    result = _batch(config) if config.mode == 'batch' else _incremental(config)
    pipeline, name, y, probabilities, threshold, validation_metrics, comparison, extra = result
    values, matrix, curves, outcomes = _evaluate(y, probabilities, threshold)
    item = {'model_name': name, 'experiment_name': 'Credit card fraud · Colab', 'classes': ['0', '1'],
        'positive_class': '1', 'threshold': threshold, 'metrics': values,
        'confusion_matrix': matrix.tolist(), 'curves': curves, 'validation_metrics': validation_metrics,
        'validation_comparison': comparison, 'dataset_fingerprint': digest.hexdigest(),
        'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'joblib']},
        'config': {'target': config.target, 'features': extra['features'], 'primary_metric': 'average_precision',
                   'preprocessing': {'scaling': 'standard unless tree/baseline',
                                     'imputation': 'median' if config.mode == 'batch' else 'constant zero'},
                   'resampling': {'method': 'none'}, 'split': {'method': config.split_method, 'fractions': [0.6, 0.2, 0.2]}},
        'colab_metadata': {**extra, 'settings': asdict(config), 'fraud_outcomes': outcomes,
            'timestamp': datetime.now(timezone.utc).isoformat(), 'total_seconds': time.perf_counter() - started,
            'threshold_selection': 'Validation F-beta with precision floor; holdout excluded',
            'candidate_selection': 'Validation average precision; holdout evaluated only for selected candidate'}}
    # No transaction features or row-level predictions in these compact application imports.
    (output / 'fraud_results.json').write_text(json.dumps({'schema_version': 1, 'experiments': [item]}, indent=2, allow_nan=False))
    pd.DataFrame([{'model': name, **values}]).to_csv(output / 'metrics.csv', index=False)
    pd.DataFrame(comparison).to_csv(output / 'validation_comparison.csv', index=False)
    if config.export_model:
        joblib.dump({'pipeline': pipeline, 'threshold': threshold, 'features': extra['features'],
                     'metadata': item}, output / 'trusted_fraud_model.joblib', compress=3)
    print('Fraud outcomes:', outcomes)
    print(f'Results saved in {output}. Import fraud_results.json into the app; keep the full CSV and model in Drive.')
    return item

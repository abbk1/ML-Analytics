"""Two bounded CPU models with one shared, untouched 6,000-customer test set."""
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import time
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, log_loss, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits
from default_studio.data import FEATURES, TARGET, NUMERICAL, CATEGORICAL, validate_record

SEED = 42
MODEL_NAMES = ('Random Forest', 'ANN')
FOREST_PARAMETERS = {'n_estimators': 80, 'max_depth': 8, 'min_samples_leaf': 15,
                     'max_features': 'sqrt', 'class_weight': 'balanced_subsample', 'n_jobs': 1, 'random_state': SEED}
ANN_PARAMETERS = {'hidden_layer_sizes': (32,), 'activation': 'relu', 'solver': 'adam',
                  'alpha': 0.01, 'batch_size': 256, 'learning_rate_init': 0.001,
                  'random_state': SEED, 'max_iter': 1, 'shuffle': True}
ANN_MAX_EPOCHS = 80
ANN_PATIENCE = 8


def preprocessor(scale=False):
    numerical = [('impute', SimpleImputer(strategy='median', keep_empty_features=True))]
    if scale:
        numerical.append(('scale', StandardScaler()))
    return Pipeline([
        ('columns', ColumnTransformer([
            ('numeric', Pipeline(numerical), NUMERICAL),
            ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False,
                                          dtype=np.float32), CATEGORICAL),
        ])),
        ('float32', FunctionTransformer(np.float32, feature_names_out='one-to-one')),
    ])


def split_indices(labels):
    return train_test_split(np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=SEED)


def _evaluate(y, probabilities):
    predictions = (probabilities >= 0.5).astype(np.int32)
    values = {'Accuracy': accuracy_score(y, predictions), 'Balanced accuracy': balanced_accuracy_score(y, predictions),
              'Precision (default)': precision_score(y, predictions, zero_division=0),
              'Recall (default)': recall_score(y, predictions, zero_division=0),
              'F1 (default)': f1_score(y, predictions, zero_division=0),
              'ROC-AUC': roc_auc_score(y, probabilities),
              'Average precision': average_precision_score(y, probabilities)}
    fpr, tpr, _ = roc_curve(y, probabilities)
    precision, recall, _ = precision_recall_curve(y, probabilities)
    return {'metrics': values, 'confusion_matrix': confusion_matrix(y, predictions, labels=[0, 1]),
            'curves': {'fpr': fpr, 'tpr': tpr, 'precision': precision, 'recall': recall},
            'predictions': predictions, 'probabilities': probabilities}


def fit_ann(X, y, progress=None, max_epochs=ANN_MAX_EPOCHS):
    """Preprocessing sees only ANN fit rows, excluding early-stop validation and test.

    An explicit inner 90/10 split avoids fitting a scaler before sklearn's hidden
    early-stopping split. The held-out 6,000 rows are never passed here.
    """
    fit_rows, stop_rows = train_test_split(np.arange(len(y)), test_size=0.1,
                                          stratify=y, random_state=SEED)
    prepare = preprocessor(scale=True)
    X_fit = prepare.fit_transform(X.iloc[fit_rows], y.iloc[fit_rows])
    X_stop = prepare.transform(X.iloc[stop_rows])
    model = MLPClassifier(**ANN_PARAMETERS)
    fit_weights = compute_sample_weight('balanced', y.iloc[fit_rows]).astype(np.float32)
    stop_weights = compute_sample_weight('balanced', y.iloc[stop_rows])
    best_model, best_loss, stale, losses = None, np.inf, 0, []
    with threadpool_limits(limits=1):
        for epoch in range(1, max_epochs + 1):
            model.partial_fit(X_fit, y.iloc[fit_rows], classes=np.array([0, 1]), sample_weight=fit_weights)
            loss = float(log_loss(y.iloc[stop_rows], model.predict_proba(X_stop),
                                  labels=[0, 1], sample_weight=stop_weights))
            losses.append({'epoch': epoch, 'validation_log_loss': loss, 'training_loss': float(model.loss_)})
            if progress:
                progress('ANN', f'Epoch {epoch}/{max_epochs} · validation loss {loss:.4f}')
            if loss < best_loss - 0.0001:
                best_model, best_loss, stale = deepcopy(model), loss, 0
                best_epoch = epoch
            else:
                stale += 1
            if stale >= ANN_PATIENCE:
                break
    if best_model is None:
        raise ValueError('ANN failed to produce a finite validation loss.')
    pipeline = Pipeline([('prepare', prepare), ('model', best_model)])
    return pipeline, pd.DataFrame(losses), fit_rows, stop_rows, best_epoch


def train_models(frame, progress=None, ann_max_epochs=ANN_MAX_EPOCHS):
    if len(frame) < 100 or set(frame[TARGET].unique()) != {0, 1}:
        raise ValueError('Training requires both classes and at least 100 customers.')
    X, y = frame.loc[:, FEATURES], frame[TARGET]
    train, test = split_indices(y)
    if y.iloc[train].value_counts().min() < 10:
        raise ValueError('At least ten training examples per class are required.')
    started = time.perf_counter()
    if progress:
        progress('Random Forest', 'Fitting 80 trees on the training partition…')
    model_started = time.perf_counter()
    forest = Pipeline([('prepare', preprocessor(scale=False)),
                       ('model', RandomForestClassifier(**FOREST_PARAMETERS))])
    with threadpool_limits(limits=1):
        forest.fit(X.iloc[train], y.iloc[train])
    forest_seconds = time.perf_counter() - model_started
    if progress:
        progress('ANN', 'Fitting training-only preprocessing and preparing early stopping…')
    model_started = time.perf_counter()
    ann, loss_history, ann_fit, ann_stop, best_epoch = fit_ann(
        X.iloc[train], y.iloc[train], progress, max_epochs=ann_max_epochs)
    ann_seconds = time.perf_counter() - model_started
    # Final evaluation occurs after both candidates and the ANN epoch count are fixed.
    results = {}
    for name, pipeline, seconds in [('Random Forest', forest, forest_seconds), ('ANN', ann, ann_seconds)]:
        with threadpool_limits(limits=1):
            probabilities = pipeline.predict_proba(X.iloc[test])[:, 1]
        results[name] = {'pipeline': pipeline, 'fit_seconds': seconds, **_evaluate(y.iloc[test], probabilities)}
    majority = int(y.iloc[train].value_counts().idxmax())
    return {'models': results, 'train_indices': train, 'test_indices': test,
            'ann_fit_indices': train[ann_fit], 'ann_stop_indices': train[ann_stop],
            'test_customer_ids': frame.ID.iloc[test].to_numpy(), 'test_labels': y.iloc[test].to_numpy(),
            'ann_history': loss_history, 'ann_best_epoch': best_epoch,
            'majority_baseline': float(np.mean(y.iloc[test].to_numpy() == majority)),
            'elapsed_seconds': time.perf_counter() - started,
            'trained_at': datetime.now(timezone.utc).isoformat(),
            'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'streamlit']}}


def comparison_table(benchmark):
    return pd.DataFrame([{'Model': name, **result['metrics'], 'Training seconds': result['fit_seconds']}
                         for name, result in benchmark['models'].items()])


def predict_record(benchmark, model_name, record):
    if model_name not in MODEL_NAMES:
        raise ValueError('Choose ANN or Random Forest.')
    values = validate_record(record)
    if len(values) != 1:
        raise ValueError('Enter exactly one customer record.')
    with threadpool_limits(limits=1):
        probabilities = benchmark['models'][model_name]['pipeline'].predict_proba(values)[0]
    return {'probability': float(probabilities[1]), 'label': int(probabilities[1] >= 0.5)}

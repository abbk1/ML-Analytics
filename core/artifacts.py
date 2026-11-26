"""Safe tabular/JSON exchange. Uploaded executable model artifacts are never loaded."""
from datetime import datetime, timezone
from io import BytesIO
import json
import math
import uuid
import joblib
import numpy as np
import pandas as pd
from core.evaluation import evaluate_predictions

METRIC_NAMES = {'accuracy', 'balanced_accuracy', 'error_rate', 'mcc', 'cohen_kappa',
    'precision_macro', 'recall_macro', 'f1_macro', 'precision_weighted', 'recall_weighted', 'f1_weighted',
    'precision_micro', 'recall_micro', 'f1_micro', 'f1_positive', 'recall_positive', 'specificity',
    'false_positive_rate', 'false_negative_rate', 'roc_auc', 'average_precision',
    'roc_auc_ovr_macro', 'average_precision_ovr_macro', 'minority_recall', 'minority_f1'}


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.DataFrame):
        return value.reset_index().to_dict(orient='records')
    raise TypeError(f'Cannot export {type(value).__name__}')


def dumps(value):
    return json.dumps(value, default=json_default, indent=2, allow_nan=False)


def compact_curves(curves, limit=1000):
    """Bound chart payloads while leaving metrics computed from all responses intact."""
    output = {}
    for label, curve in curves.items():
        output[label] = {}
        for x, y in [('fpr', 'tpr'), ('recall', 'precision')]:
            indices = np.unique(np.linspace(0, len(curve[x]) - 1, min(len(curve[x]), limit), dtype=int))
            output[label][x] = np.asarray(curve[x])[indices].tolist()
            output[label][y] = np.asarray(curve[y])[indices].tolist()
    return output


def metadata(result):
    keys = ['id', 'batch_id', 'source', 'timestamp', 'model_name', 'experiment_name', 'config',
            'model_parameters', 'dataset_name', 'dataset_fingerprint', 'dataset_shape', 'split_id',
            'train_indices', 'test_indices', 'classes', 'positive_class', 'minority_class',
            'cv_scores', 'cv_mean', 'cv_std', 'fit_seconds', 'total_seconds', 'threshold',
            'resampled_delta', 'generated_rows', 'removed_rows', 'warnings', 'versions',
            'validation_metrics', 'validation_comparison', 'colab_metadata']
    output = {key: result[key] for key in keys if key in result}
    output['metrics'] = result['evaluation']['metrics']
    output['confusion_matrix'] = result['evaluation'].get('confusion_matrix')
    output['curves'] = compact_curves(result['evaluation'].get('curves', {}))
    return output


def export_experiments(results):
    return dumps({'schema_version': 1, 'experiments': [metadata(r) for r in results]})


def excel_bytes(tables):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for name, frame in tables.items():
            # Prevent spreadsheet formula interpretation of uploaded text.
            safe = frame.map(lambda v: "'" + v if isinstance(v, str) and v.startswith(('=', '+', '-', '@')) else v)
            safe.to_excel(writer, sheet_name=name[:31], index=False)
    return output.getvalue()


def csv_bytes(frame):
    safe = frame.map(lambda v: "'" + v if isinstance(v, str) and v.startswith(('=', '+', '-', '@')) else v)
    return safe.to_csv(index=False).encode('utf-8')


def pipeline_bytes(result):
    output = BytesIO()
    joblib.dump({'pipeline': result['pipeline'], 'metadata': metadata(result)}, output, compress=3)
    return output.getvalue()


def validate_metrics(values):
    if not isinstance(values, dict) or not values:
        raise ValueError('Metrics must be a nonempty object of supported numeric metrics.')
    output = {}
    for key, value in values.items():
        if key not in METRIC_NAMES:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not math.isfinite(value):
            raise ValueError(f'{key} must be finite and numeric.')
        minimum = -1 if key in ('mcc', 'cohen_kappa') else 0
        if not minimum <= value <= 1:
            raise ValueError(f'{key} must be in [{minimum}, 1].')
        output[key] = float(value)
    if not output:
        raise ValueError('No supported metrics found. Export fractional metrics in [0, 1], not percentages.')
    return output


def validate_matrix(matrix, classes):
    if not isinstance(classes, list) or not 2 <= len(classes) <= 50:
        raise ValueError('Supply 2–50 class labels.')
    classes = [str(c) for c in classes]
    if len(set(classes)) != len(classes):
        raise ValueError('Class labels must be unique.')
    values = np.asarray(matrix)
    if values.shape != (len(classes), len(classes)) or not np.issubdtype(values.dtype, np.number):
        raise ValueError('Confusion matrix shape must match the ordered classes.')
    if not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, np.floor(values)).all():
        raise ValueError('Confusion matrix counts must be finite nonnegative integers.')
    if values.sum() < 1 or values.sum() > 1e12:
        raise ValueError('Confusion matrix total must be between 1 and one trillion.')
    return values.astype(int), classes


def imported_result(item, evaluation=None):
    name = item.get('model_name', item.get('model', 'Imported model'))
    if not isinstance(name, str) or not 1 <= len(name) <= 120:
        raise ValueError('Model name must be a string of 1–120 characters.')
    config = item.get('config', {})
    if not isinstance(config, dict):
        raise ValueError('Imported config must be an object.')
    for section in ('preprocessing', 'resampling', 'split'):
        if section in config and not isinstance(config[section], dict):
            raise ValueError(f'Imported config.{section} must be an object.')
    for section, key in [('preprocessing', 'scaling'), ('resampling', 'method')]:
        value = config.get(section, {}).get(key, 'unknown')
        if not isinstance(value, str) or len(value) > 100:
            raise ValueError(f'Imported {section}.{key} must be a short string.')
    if 'primary_metric' in config and not isinstance(config['primary_metric'], str):
        raise ValueError('Imported primary_metric must be a string.')
    if evaluation is None:
        evaluation = {'metrics': validate_metrics(item.get('metrics', {})), 'per_class': pd.DataFrame(),
                      'confusion_matrix': None, 'curves': {}, 'unavailable': {}, 'response_kind': 'unavailable'}
    classes = item.get('classes') or []
    positive = str(item.get('positive_class', classes[-1] if classes else ''))
    threshold = item.get('threshold')
    if threshold is not None and (isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold)):
        raise ValueError('Threshold must be finite and numeric.')
    validation = item.get('validation_metrics', {})
    comparison = item.get('validation_comparison', [])
    if not isinstance(validation, dict) or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in validation.values()):
        raise ValueError('Validation metrics must be a finite numeric object.')
    if not isinstance(comparison, list) or len(comparison) > 20:
        raise ValueError('Validation comparison must contain at most 20 candidates.')
    for candidate in comparison:
        if not isinstance(candidate, dict) or not isinstance(candidate.get('model'), str):
            raise ValueError('Validation comparison needs named candidates.')
        ap = candidate.get('validation_average_precision')
        if not isinstance(ap, (int, float)) or not math.isfinite(ap) or not 0 <= ap <= 1:
            raise ValueError('Validation average precision must be in [0,1].')
    return {'id': uuid.uuid4().hex[:12], 'model_name': name, 'experiment_name': str(item.get('experiment_name', 'Colab import'))[:120],
            'source': 'Imported · unverified external evaluation', 'timestamp': datetime.now(timezone.utc).isoformat(),
            # Imported split claims cannot establish equality with locally held-out observations.
            'split_id': 'external-' + uuid.uuid4().hex[:8], 'classes': item.get('classes', []),
            'positive_class': positive, 'threshold': threshold,
            'validation_metrics': validation, 'validation_comparison': comparison,
            'colab_metadata': item.get('colab_metadata', {}),
            'config': config,
            'evaluation': evaluation, 'pipeline': None, 'warnings': ['External methodology and split identity are not independently verified.'],
            'versions': item.get('versions', {}), 'dataset_fingerprint': item.get('dataset_fingerprint', 'unknown')}


def matrix_evaluation(matrix, classes, positive=None):
    matrix, classes = validate_matrix(matrix, classes)
    # At most classes² weighted pairs, independent of the number of transactions.
    actual, predicted, weights = [], [], []
    for i, actual_label in enumerate(classes):
        for j, predicted_label in enumerate(classes):
            count = int(matrix[i, j])
            if count:
                actual.append(actual_label)
                predicted.append(predicted_label)
                weights.append(count)
    return evaluate_predictions(actual, predicted, classes, positive=positive, sample_weight=weights)


def validate_curves(document, classes):
    if not isinstance(document, dict) or set(document) - set(classes):
        raise ValueError('Curve keys must be ordered class labels.')
    curves = {}
    for label, curve in document.items():
        if not isinstance(curve, dict) or set(curve) != {'fpr', 'tpr', 'precision', 'recall'}:
            raise ValueError('Each curve needs fpr, tpr, precision, recall arrays.')
        arrays = {key: np.asarray(value, dtype=float) for key, value in curve.items()}
        if any(a.ndim != 1 or not 2 <= len(a) <= 1000 or not np.isfinite(a).all()
               or (a < 0).any() or (a > 1).any() for a in arrays.values()):
            raise ValueError('Curves require 2–1,000 finite values in [0, 1].')
        if len(arrays['fpr']) != len(arrays['tpr']) or len(arrays['precision']) != len(arrays['recall']):
            raise ValueError('Curve coordinate lengths must match.')
        if (np.diff(arrays['fpr']) < 0).any() or (np.diff(arrays['recall']) > 0).any():
            raise ValueError('ROC FPR must increase and PR recall must decrease.')
        curves[label] = arrays
    return curves


def import_results(content, filename, kind='Metrics table'):
    if len(content) > 10 * 1024**2:
        raise ValueError('Result imports are limited to 10 MB.')
    if filename.lower().endswith('.csv'):
        table = pd.read_csv(BytesIO(content), nrows=50001)
        if len(table) > 50000:
            raise ValueError('Prediction imports are limited to 50,000 rows.')
        if kind == 'Predictions':
            if not {'actual', 'predicted'}.issubset(table):
                raise ValueError('Prediction CSV needs actual and predicted columns.')
            if table[['actual', 'predicted']].isna().any().any():
                raise ValueError('Predictions and actual labels cannot be missing.')
            actual, predicted = table.actual.astype(str), table.predicted.astype(str)
            classes = sorted(set(actual) | set(predicted))
            if not 2 <= len(classes) <= 50:
                raise ValueError('Prediction import requires 2–50 classes.')
            result = imported_result({'model_name': 'Colab predictions', 'classes': classes},
                                     evaluate_predictions(actual, predicted, classes))
            result['errors'] = pd.DataFrame({'_actual': actual, '_predicted': predicted, '_correct': actual == predicted})
            return [result]
        if not 1 <= len(table) <= 200 or 'model' not in table:
            raise ValueError('Metric CSV needs a model column and 1–200 rows.')
        return [imported_result({'model': row['model'], 'metrics': {
                    key: row[key] for key in METRIC_NAMES & set(table.columns) if pd.notna(row[key])}})
                for _, row in table.iterrows()]
    if not filename.lower().endswith('.json'):
        raise ValueError('Use CSV or JSON. Uploaded Python, notebooks, Pickle and Joblib files are never executed or loaded.')
    try:
        document = json.loads(content)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError('Invalid UTF-8 JSON file.') from exc
    if not isinstance(document, dict):
        raise ValueError('JSON root must be an object.')
    if 'experiments' in document:
        if document.get('schema_version') != 1:
            raise ValueError('Unsupported experiment schema. Expected schema_version: 1.')
        items = document['experiments']
    else:
        items = [document]
    if not isinstance(items, list) or not 1 <= len(items) <= 200:
        raise ValueError('Import 1–200 experiment objects.')
    results = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError('Every experiment must be an object.')
        if item.get('confusion_matrix') is not None:
            _, item['classes'] = validate_matrix(item['confusion_matrix'], item.get('classes'))
            positive = str(item.get('positive_class', item['classes'][-1]))
            if positive not in item['classes']:
                raise ValueError('Positive class must be one of the supplied classes.')
            evaluated = matrix_evaluation(item['confusion_matrix'], item.get('classes'), positive)
            if 'metrics' in item:
                supplied = validate_metrics(item['metrics'])
                for key in set(supplied) & set(evaluated['metrics']):
                    if not np.isclose(supplied[key], evaluated['metrics'][key], atol=1e-6):
                        raise ValueError(f'Supplied {key} disagrees with the confusion matrix.')
                evaluated['metrics'].update(supplied)
                for key in supplied:
                    evaluated['unavailable'].pop(key, None)
            evaluated['curves'] = validate_curves(item.get('curves', {}), item['classes'])
            if evaluated['curves']:
                evaluated['response_kind'] = 'external continuous responses (unverified)'
            results.append(imported_result(item, evaluated))
        else:
            results.append(imported_result(item))
    return results

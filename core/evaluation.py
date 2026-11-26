"""Metrics from class predictions and genuine probability/decision responses."""
import numpy as np
import pandas as pd
from sklearn import metrics


def model_response(pipeline, X):
    if hasattr(pipeline, 'predict_proba'):
        return np.asarray(pipeline.predict_proba(X)), 'probability'
    if hasattr(pipeline, 'decision_function'):
        return np.asarray(pipeline.decision_function(X)), 'decision score'
    return None, 'unavailable'


def positive_scores(response, classes, positive):
    if response is None:
        return None
    index = list(classes).index(positive)
    if response.ndim == 1:
        return response if index == 1 else -response
    return response[:, index]


def evaluate_predictions(y_true, y_pred, classes, response=None, response_kind='unavailable', positive=None):
    classes = list(classes)
    positive = positive or classes[-1]
    values = {'accuracy': metrics.accuracy_score(y_true, y_pred),
              'balanced_accuracy': metrics.balanced_accuracy_score(y_true, y_pred),
              'error_rate': 1 - metrics.accuracy_score(y_true, y_pred),
              'mcc': metrics.matthews_corrcoef(y_true, y_pred),
              'cohen_kappa': metrics.cohen_kappa_score(y_true, y_pred)}
    for average in ('macro', 'weighted', 'micro'):
        for name, function in [('precision', metrics.precision_score), ('recall', metrics.recall_score),
                               ('f1', metrics.f1_score)]:
            values[f'{name}_{average}'] = function(y_true, y_pred, labels=classes, average=average, zero_division=0)
    precision, recall, f1, support = metrics.precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0)
    report = pd.DataFrame({'precision': precision, 'recall': recall, 'f1-score': f1,
                           'support': support}, index=classes)
    matrix = metrics.confusion_matrix(y_true, y_pred, labels=classes)
    tp = matrix.diagonal()
    fn = matrix.sum(axis=1) - tp
    fp = matrix.sum(axis=0) - tp
    tn = matrix.sum() - tp - fn - fp
    report['specificity'] = np.divide(tn, tn + fp, out=np.zeros(len(classes), dtype=float), where=tn + fp != 0)
    report['false_positive_rate'] = 1 - report['specificity']
    report['false_negative_rate'] = np.divide(fn, fn + tp, out=np.zeros(len(classes), dtype=float), where=fn + tp != 0)
    unavailable = {}
    curves = {}
    if len(classes) == 2:
        index = classes.index(positive)
        values['f1_positive'] = float(report.loc[positive, 'f1-score'])
        values['recall_positive'] = float(report.loc[positive, 'recall'])
        values['specificity'] = float(report.loc[positive, 'specificity'])
        values['false_positive_rate'] = float(report.loc[positive, 'false_positive_rate'])
        values['false_negative_rate'] = float(report.loc[positive, 'false_negative_rate'])
        binary = np.asarray(y_true) == positive
        scores = positive_scores(response, classes, positive)
        if scores is not None and len(np.unique(binary)) == 2:
            values['roc_auc'] = metrics.roc_auc_score(binary, scores)
            values['average_precision'] = metrics.average_precision_score(binary, scores)
            fpr, tpr, _ = metrics.roc_curve(binary, scores)
            precision, recall, _ = metrics.precision_recall_curve(binary, scores)
            curves[positive] = {'fpr': fpr, 'tpr': tpr, 'precision': precision, 'recall': recall}
        else:
            unavailable.update(roc_auc='No continuous response or only one observed test class.',
                               average_precision='No continuous response or only one observed test class.')
    elif response is not None and response.ndim == 2 and response.shape[1] == len(classes):
        aucs, aps = [], []
        for i, label in enumerate(classes):
            binary = np.asarray(y_true) == label
            if len(np.unique(binary)) < 2:
                continue
            scores = response[:, i]
            aucs.append(metrics.roc_auc_score(binary, scores))
            aps.append(metrics.average_precision_score(binary, scores))
            fpr, tpr, _ = metrics.roc_curve(binary, scores)
            precision, recall, _ = metrics.precision_recall_curve(binary, scores)
            curves[label] = {'fpr': fpr, 'tpr': tpr, 'precision': precision, 'recall': recall}
        if len(aucs) == len(classes):
            values['roc_auc_ovr_macro'] = float(np.mean(aucs))
            values['average_precision_ovr_macro'] = float(np.mean(aps))
        else:
            unavailable['roc_auc_ovr_macro'] = 'One or more classes are absent from this evaluation set.'
    else:
        unavailable['roc_auc_ovr_macro'] = 'Classifier has no compatible continuous class responses.'
        unavailable['average_precision_ovr_macro'] = 'Classifier has no compatible continuous class responses.'
    return {'metrics': values, 'per_class': report, 'confusion_matrix': matrix,
            'curves': curves, 'unavailable': unavailable, 'response_kind': response_kind}


def threshold_analysis(y_true, scores, positive, response_kind):
    binary = np.asarray(y_true) == positive
    if response_kind == 'probability':
        thresholds = np.linspace(0, 1, 101)
    else:
        thresholds = np.unique(np.r_[np.quantile(scores, np.linspace(0, 1, 101)), 0.0])
    rows = []
    for threshold in thresholds:
        predicted = scores >= threshold
        tn, fp, fn, tp = metrics.confusion_matrix(binary, predicted, labels=[False, True]).ravel()
        rows.append({'threshold': float(threshold), 'precision': metrics.precision_score(binary, predicted, zero_division=0),
                     'recall': metrics.recall_score(binary, predicted, zero_division=0),
                     'f1_positive': metrics.f1_score(binary, predicted, zero_division=0),
                     'false_positives': int(fp), 'false_negatives': int(fn)})
    table = pd.DataFrame(rows)
    # Stable tie-breaking prefers the usual operating threshold.
    center = 0.5 if response_kind == 'probability' else 0.0
    table['distance_to_default'] = abs(table['threshold'] - center)
    chosen = table.sort_values(['f1_positive', 'distance_to_default'], ascending=[False, True]).iloc[0]
    return float(chosen['threshold']), table.drop(columns='distance_to_default')

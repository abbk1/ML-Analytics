"""Credit-card schema hints and factual outcome summaries."""
from core.config import PreprocessingConfig, ResamplingConfig, SplitConfig
from core.models import DEFAULTS


def is_creditcard_dataset(frame):
    return {'Class', 'Amount', 'Time'}.issubset(frame.columns) and frame['Class'].dropna().isin([0, 1]).all()


def creditcard_preset(frame):
    if not is_creditcard_dataset(frame) or set(frame.Class.dropna().unique()) != {0, 1}:
        raise ValueError('The preset needs Class=0/1, Amount and Time. Apply a custom target selection for other datasets.')
    positive = str(next(label for label in frame.Class.dropna().unique() if label == 1))
    return {'target': 'Class', 'feature_selection': [c for c in frame if c != 'Class'],
            'text_column': None, 'positive_class': positive, 'primary_metric': 'average_precision',
            'experiment_name': 'Credit card fraud', 'prep_config': PreprocessingConfig(scaling='standard'),
            'resample_config': ResamplingConfig(method='none'), 'split_config': SplitConfig(folds=3),
            'model_settings': {'Dummy baseline': dict(DEFAULTS['Dummy baseline']),
                'Random Forest': {**DEFAULTS['Random Forest'], 'class_weight': 'balanced_subsample'},
                'Logistic Regression': {**DEFAULTS['Logistic Regression'], 'class_weight': 'balanced'}},
            'active_ids': [], 'current_config_id': None, 'trained_config_id': None}


def fraud_outcomes(result):
    classes = list(result.get('classes', []))
    positive = result.get('positive_class')
    matrix = result['evaluation'].get('confusion_matrix')
    if len(classes) != 2 or positive not in classes or matrix is None:
        return None
    positive_index = classes.index(positive)
    negative_index = 1 - positive_index
    return {'Fraud detected': int(matrix[positive_index, positive_index]),
            'Fraud missed': int(matrix[positive_index, negative_index]),
            'Legitimate flagged': int(matrix[negative_index, positive_index]),
            'Legitimate cleared': int(matrix[negative_index, negative_index])}

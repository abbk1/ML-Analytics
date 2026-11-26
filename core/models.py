"""Conservative classifier defaults and validated, bounded user overrides."""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import __version__ as sklearn_version
from packaging.version import Version
import numpy as np

MODEL_CLASSES = {
    'Dummy baseline': DummyClassifier, 'MLP': MLPClassifier, 'Random Forest': RandomForestClassifier,
    'Logistic Regression': LogisticRegression, 'SVM': SVC, 'Linear SVM': LinearSVC,
    'Decision Tree': DecisionTreeClassifier, 'KNN': KNeighborsClassifier,
    'Gaussian NB': GaussianNB, 'Multinomial NB': MultinomialNB,
    'Bernoulli NB': BernoulliNB, 'Gradient Boosting': GradientBoostingClassifier,
}
DEFAULTS = {
    'Dummy baseline': {'strategy': 'prior'},
    'MLP': {'hidden_layer_sizes': [64, 32], 'activation': 'relu', 'solver': 'adam',
            'learning_rate': 'constant', 'learning_rate_init': 0.001, 'alpha': 0.0001,
            'batch_size': 'auto', 'max_iter': 150, 'early_stopping': False},
    'Random Forest': {'n_estimators': 100, 'max_depth': 12, 'min_samples_split': 2,
                       'min_samples_leaf': 1, 'max_features': 'sqrt', 'class_weight': None},
    'Logistic Regression': {'C': 1.0, 'solver': 'lbfgs', 'penalty': 'l2', 'l1_ratio': 0.5, 'max_iter': 300, 'class_weight': None},
    'SVM': {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale', 'degree': 3, 'class_weight': None},
    'Linear SVM': {'C': 1.0, 'max_iter': 1000, 'class_weight': None},
    'Decision Tree': {'max_depth': 12, 'min_samples_split': 2, 'min_samples_leaf': 1, 'class_weight': None},
    'KNN': {'n_neighbors': 5, 'weights': 'uniform', 'p': 2},
    'Gaussian NB': {'var_smoothing': 1e-9}, 'Multinomial NB': {'alpha': 1.0},
    'Bernoulli NB': {'alpha': 1.0, 'binarize': 0.0},
    'Gradient Boosting': {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1},
}
DENSE_MODELS = {'Gaussian NB', 'Gradient Boosting'}
SCALE_SENSITIVE = {'MLP', 'Logistic Regression', 'SVM', 'Linear SVM', 'KNN'}


def make_model(name, overrides=None, seed=42):
    if name not in MODEL_CLASSES:
        raise ValueError(f'Unknown classifier: {name}')
    parameters = dict(DEFAULTS[name])
    overrides = overrides or {}
    if not isinstance(overrides, dict) or set(overrides) - set(parameters):
        raise ValueError(f'Only documented parameters can be overridden for {name}: {list(parameters)}')
    parameters.update(overrides)
    for key, limit in {'n_estimators': 300, 'max_iter': 2000, 'max_depth': 30,
                       'n_neighbors': 100, 'degree': 5}.items():
        value = parameters.get(key)
        if key == 'max_depth' and key in parameters and value is None:
            raise ValueError('Set a finite max_depth (1–30) for cloud training.')
        if value is not None and (not isinstance(value, int) or not 1 <= value <= limit):
            raise ValueError(f'{key} must be an integer between 1 and {limit}.')
    if name == 'MLP':
        layers = parameters['hidden_layer_sizes']
        if not isinstance(layers, (list, tuple)) or not 1 <= len(layers) <= 4 or any(
                not isinstance(n, int) or not 1 <= n <= 256 for n in layers):
            raise ValueError('MLP requires 1–4 hidden layers with 1–256 neurons each.')
        parameters['hidden_layer_sizes'] = tuple(layers)
    if name == 'Logistic Regression':
        solver, penalty = parameters['solver'], parameters['penalty']
        allowed = {'lbfgs': ['l2', None], 'liblinear': ['l1', 'l2'],
                   'saga': ['l1', 'l2', 'elasticnet', None], 'newton-cg': ['l2', None], 'sag': ['l2', None]}
        if solver not in allowed or penalty not in allowed[solver]:
            raise ValueError('Incompatible logistic regression solver and penalty.')
    estimator = MODEL_CLASSES[name]()
    supported = estimator.get_params()
    if 'random_state' in supported:
        parameters['random_state'] = seed
    if 'n_jobs' in supported and name != 'Logistic Regression':
        parameters['n_jobs'] = 1
    if name == 'Logistic Regression':
        if Version(sklearn_version) >= Version('1.8'):
            penalty = parameters.pop('penalty')
            if penalty != 'elasticnet':
                parameters['l1_ratio'] = 1.0 if penalty == 'l1' else 0.0
            if penalty is None:
                parameters['C'] = np.inf
        elif parameters['penalty'] != 'elasticnet':
            parameters.pop('l1_ratio')
    if name == 'SVM':
        parameters.update(probability=False, cache_size=128, max_iter=5000)
    estimator.set_params(**parameters)
    # sklearn validates parameter types/ranges before fit; doing it here avoids expensive CV failures.
    if hasattr(estimator, '_validate_params'):
        estimator._validate_params()
    return estimator

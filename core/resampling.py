"""Training-only resampling with fold-specific minority and allocation validation."""
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from imblearn.utils import check_sampling_strategy

METHODS = ['none', 'random_over', 'random_under', 'smote', 'smotenc', 'smote_tomek']


class RecordingSampler(BaseEstimator):
    def __init__(self, method='random_over', strategy='auto', k_neighbors=5,
                 random_state=42, max_rows=100000, categorical_features=None):
        self.method = method
        self.strategy = strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.max_rows = max_rows
        self.categorical_features = categorical_features

    def fit_resample(self, X, y):
        if self.method not in METHODS[1:]:
            raise ValueError('Unknown resampling method.')
        kind = 'under-sampling' if self.method == 'random_under' else 'over-sampling'
        strategy = check_sampling_strategy(self.strategy, y, kind)
        labels, counts = np.unique(y, return_counts=True)
        self.before_counts_ = dict(zip(labels.tolist(), counts.tolist()))
        estimated_rows = int(sum(strategy.get(label, count) for label, count in zip(labels, counts))) \
            if kind == 'under-sampling' else len(y) + int(sum(strategy.values()))
        if estimated_rows > self.max_rows:
            raise ValueError(f'Resampling would produce about {estimated_rows:,} rows; limit {self.max_rows:,}. '
                             'Use undersampling, class weights, a smaller sampling ratio, or Colab.')
        if sparse.issparse(X):
            estimated_bytes = (X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) * estimated_rows / max(len(y), 1) * 2
        else:
            estimated_bytes = estimated_rows * X.shape[1] * 8 * 3
        if estimated_bytes > 256 * 1024**2:
            raise ValueError('Estimated resampling workspace exceeds 256 MB. Reduce rows/features or use class weights.')
        if self.method.startswith('smote'):
            too_small = [str(label) for label, extra in strategy.items()
                         if extra > 0 and self.before_counts_[label] <= self.k_neighbors]
            if too_small:
                raise ValueError(f'SMOTE k_neighbors={self.k_neighbors} needs more training-fold observations '
                                 f'for classes {too_small}. Lower k, reduce folds, or use random oversampling/class weights.')
        common = {'sampling_strategy': self.strategy, 'random_state': self.random_state}
        if self.method == 'random_over':
            sampler = RandomOverSampler(**common)
        elif self.method == 'random_under':
            sampler = RandomUnderSampler(**common)
        elif self.method == 'smote':
            sampler = SMOTE(k_neighbors=self.k_neighbors, **common)
        elif self.method == 'smotenc':
            sampler = SMOTENC(categorical_features=self.categorical_features, k_neighbors=self.k_neighbors, **common)
        else:
            sampler = SMOTETomek(smote=SMOTE(k_neighbors=self.k_neighbors, **common), **common)
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        labels, counts = np.unique(y_resampled, return_counts=True)
        self.after_counts_ = dict(zip(labels.tolist(), counts.tolist()))
        self.delta_ = len(y_resampled) - len(y)
        self.generated_ = max(0, self.delta_)
        self.removed_ = max(0, -self.delta_)
        if self.method == 'smote_tomek':
            self.generated_ = int(sum(sampler.smote_.sampling_strategy_.values()))
            self.removed_ = len(y) + self.generated_ - len(y_resampled)
        return X_resampled, y_resampled

"""Fold-local feature transformations, including sparse text and mixed-data SMOTENC."""
import re
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
                                   OneHotEncoder, OrdinalEncoder)
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from core.models import make_model, DENSE_MODELS
from core.resampling import RecordingSampler


class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, lowercase=True, remove_punctuation=False, remove_numbers=False,
                 normalize_whitespace=True, remove_stopwords=False):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.normalize_whitespace = normalize_whitespace
        self.remove_stopwords = remove_stopwords

    def fit(self, X, y=None):
        return self

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)

    def transform(self, X):
        texts = pd.Series(np.asarray(X).ravel()).fillna('').astype(str)
        stopwords = ENGLISH_STOP_WORDS - {'no', 'not', 'nor', 'never'}
        def clean(text):
            if self.lowercase:
                text = text.lower()
            if self.remove_punctuation:
                text = re.sub(r'[^\w\s]', ' ', text)
            if self.remove_numbers:
                text = re.sub(r'\d+', ' ', text)
            if self.remove_stopwords:
                text = ' '.join(w for w in text.split() if w.lower() not in stopwords)
            if self.normalize_whitespace:
                text = ' '.join(text.split())
            return text
        return texts.map(clean).to_numpy()


class CategoricalStrings(BaseEstimator, TransformerMixin):
    """Normalize nullable pandas strings and mixed object values without inventing categories."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        frame = pd.DataFrame(X)
        return frame.map(lambda x: np.nan if pd.isna(x) else str(x)).to_numpy(dtype=object)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


class NumericalValues(BaseEstimator, TransformerMixin):
    """Normalize nullable/object prediction input to float NaN before imputation."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X).apply(pd.to_numeric, errors='raise').to_numpy(dtype=float, na_value=np.nan)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


class BoundedDense(BaseEstimator, TransformerMixin):
    def __init__(self, max_cells=2000000):
        self.max_cells = max_cells

    def fit(self, X, y=None):
        self._check(X)
        return self

    def _check(self, X):
        if X.shape[0] * X.shape[1] > self.max_cells:
            raise ValueError('This classifier needs dense input; matrix exceeds 2 million cells. '
                             'Choose a sparse-compatible classifier or fewer features/rows.')

    def transform(self, X):
        self._check(X)
        return X.toarray() if sparse.issparse(X) else X


class MatrixBudget(BaseEstimator, TransformerMixin):
    """Reject oversized fitted representations without densifying sparse matrices."""
    def __init__(self, model_name=''):
        self.model_name = model_name

    def fit(self, X, y=None):
        self._check(X)
        return self

    def _check(self, X):
        size = X.data.nbytes + X.indices.nbytes + X.indptr.nbytes if sparse.issparse(X) else X.nbytes
        if X.shape[1] > 50000 or size > 150 * 1024**2:
            raise ValueError('Transformed feature matrix exceeds 50,000 columns or 150 MB. Reduce features, categories or rows.')
        if self.model_name == 'MLP' and (X.shape[1] > 10000 or X.shape[0] > 20000):
            raise ValueError('Cloud MLP limit: 10,000 transformed features and 20,000 training rows. Use a smaller configuration or Colab.')

    def transform(self, X):
        self._check(X)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def scaler(name):
    choices = {'none': 'passthrough', 'standard': StandardScaler(), 'minmax': MinMaxScaler(),
               'robust': RobustScaler(), 'maxabs': MaxAbsScaler()}
    if name not in choices:
        raise ValueError('Unknown scaling method.')
    return choices[name]


def columns_by_type(X):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    if any(pd.api.types.is_datetime64_any_dtype(X[c]) for c in categorical):
        raise ValueError('Datetime features require explicit feature engineering or exclusion.')
    if np.isinf(X[numeric].to_numpy(dtype=float)).any():
        raise ValueError('Numerical features contain infinity. Correct these values before training.')
    return numeric, categorical


def build_pipeline(X, config, model_name, overrides=None):
    p = config.preprocessing
    method = config.resampling.method
    model = make_model(model_name, overrides, config.split.random_state)
    if p.imputation not in ('mean', 'median', 'most_frequent', 'constant'):
        raise ValueError('Unsupported missing-value strategy.')
    if p.encoding not in ('onehot', 'ordinal'):
        raise ValueError('Choose onehot or ordinal encoding.')
    if model_name == 'Multinomial NB' and p.task != 'Text':
        # Training min-max ranges cannot guarantee nonnegative unseen observations.
        raise ValueError('Multinomial NB is enabled for nonnegative TF-IDF text only. '
                         'Use Gaussian NB for continuous numerical input.')
    steps = []
    if p.task == 'Text':
        if config.text_column not in config.features or len(config.features) != 1:
            raise ValueError('Text mode requires exactly the selected text column as its feature.')
        if model_name in DENSE_MODELS or method == 'smotenc':
            raise ValueError('This model/resampler is incompatible with sparse text. Use Linear SVM, '
                             'logistic regression, Multinomial NB, or a random resampler.')
        if not 1 <= p.ngram_max <= 3 or not 1 <= p.max_features <= 30000 or p.min_df < 1 or not 0 < p.max_df <= 1:
            raise ValueError('Invalid or excessive TF-IDF configuration.')
        text = Pipeline([
            ('clean', TextCleaner(p.lowercase, p.remove_punctuation, p.remove_numbers,
                                  p.normalize_whitespace, p.remove_stopwords)),
            ('tfidf', TfidfVectorizer(lowercase=False, ngram_range=(1, p.ngram_max), min_df=p.min_df,
                                    max_df=p.max_df, max_features=p.max_features,
                                    sublinear_tf=p.sublinear_tf, norm=p.norm)),
        ])
        preprocessor = ColumnTransformer([('text', text, config.text_column)], sparse_threshold=1.0)
        # TF-IDF has its own sparse-compatible normalization. Numerical scalers do not apply.
    else:
        numeric, categorical = columns_by_type(X)
        if sum(X[c].nunique(dropna=False) for c in categorical) > 30000:
            raise ValueError('Selected categories exceed 30,000 distinct values. Exclude identifiers or use Text mode.')
        num = Pipeline([('values', NumericalValues()), ('impute', SimpleImputer(strategy=p.imputation, fill_value=p.fill_value,
                                                keep_empty_features=True)), ('scale', scaler(p.scaling))])
        cat_impute = Pipeline([('strings', CategoricalStrings()),
                              ('impute', SimpleImputer(strategy='most_frequent' if p.imputation != 'constant'
                                                      else 'constant', fill_value='Missing', keep_empty_features=True))])
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True) if p.encoding == 'onehot' else \
            OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        if method in ('smote', 'smote_tomek') and categorical:
            raise ValueError('Ordinary SMOTE cannot interpolate categorical features. Choose SMOTENC '
                             'for mixed data, or random over/undersampling.')
        if method == 'smotenc':
            if not numeric or not categorical:
                raise ValueError('SMOTENC needs both numerical and categorical columns. '
                                 'Use SMOTE for numerical data or random resampling for categorical-only data.')
            cat = Pipeline([('impute', cat_impute), ('encode', OrdinalEncoder(
                handle_unknown='use_encoded_value', unknown_value=-1))])
        else:
            cat = Pipeline([('impute', cat_impute), ('encode', encoder)])
        transforms = []
        if numeric:
            transforms.append(('numeric', num, numeric))
        if categorical:
            transforms.append(('categorical', cat, categorical))
        preprocessor = ColumnTransformer(transforms, sparse_threshold=0.0 if method == 'smotenc' else 1.0)
    steps.append(('preprocess', preprocessor))
    steps.append(('budget', MatrixBudget(model_name)))
    if method != 'none':
        cat_indices = list(range(len(numeric), len(numeric) + len(categorical))) if method == 'smotenc' else None
        steps.append(('resample', RecordingSampler(method, config.resampling.strategy,
                        config.resampling.k_neighbors, config.split.random_state,
                        config.max_resampled_rows, cat_indices)))
        if method == 'smotenc' and p.encoding == 'onehot':
            steps.append(('post_encode', ColumnTransformer([
                ('numeric', 'passthrough', list(range(len(numeric)))),
                ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=True), cat_indices),
            ], sparse_threshold=1.0)))
    if model_name in DENSE_MODELS:
        steps.append(('dense', BoundedDense()))
    steps.append(('final_budget', MatrixBudget(model_name)))
    steps.append(('model', model))
    return ImbalancedPipeline(steps)

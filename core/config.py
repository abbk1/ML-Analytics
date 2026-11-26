"""Serializable experiment settings; all fitted state lives in model pipelines."""
from dataclasses import asdict, dataclass, field
import hashlib
import json


@dataclass
class PreprocessingConfig:
    task: str = "Tabular"
    imputation: str = "median"
    fill_value: float = 0.0
    encoding: str = "onehot"
    scaling: str = "standard"
    drop_missing_features: bool = False
    drop_missing_target: bool = False
    drop_duplicates: bool = False
    lowercase: bool = True
    remove_punctuation: bool = False
    remove_numbers: bool = False
    normalize_whitespace: bool = True
    remove_stopwords: bool = False
    ngram_max: int = 1
    min_df: int = 1
    max_df: float = 1.0
    max_features: int = 10000
    sublinear_tf: bool = True
    norm: str | None = "l2"


@dataclass
class SplitConfig:
    test_size: float = 0.2
    random_state: int = 42
    folds: int = 5
    repeats: int = 1


@dataclass
class ResamplingConfig:
    method: str = "none"
    strategy: str | float = "auto"
    k_neighbors: int = 5


@dataclass
class ExperimentConfig:
    target: str
    features: list[str]
    text_column: str | None = None
    name: str = "Experiment"
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    resampling: ResamplingConfig = field(default_factory=ResamplingConfig)
    primary_metric: str = "f1_macro"
    positive_class: str | None = None
    optimize_threshold: bool = False
    threshold_metric: str = "f1_positive"
    max_rows: int = 50000
    max_resampled_rows: int = 100000

    def to_dict(self):
        return asdict(self)

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


RANKING_METRICS = {
    "Macro F1": "f1_macro", "Balanced accuracy / macro recall": "balanced_accuracy",
    "Weighted F1": "f1_weighted", "Positive-class F1 (binary)": "f1_positive",
    "Average precision (binary)": "average_precision", "Positive-class recall (binary)": "recall_positive",
}
SCALERS = {
    "standard": "Centers and scales numerical columns; recommended for MLP, SVM, KNN and logistic regression.",
    "minmax": "Maps numerical training ranges to [0, 1]; future values can lie outside that range.",
    "robust": "Uses training medians and interquartile ranges; useful with outliers.",
    "maxabs": "Scales by training maximum absolute values and preserves sparsity.",
    "none": "Keeps original units; usually appropriate for trees and forests.",
}

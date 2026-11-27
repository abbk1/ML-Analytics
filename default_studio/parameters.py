"""Serializable training controls bounded for the public Streamlit CPU workflow."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math

SEED = 42


@dataclass(frozen=True)
class TrainingParameters:
    forest_trees: int = 80
    forest_depth: int = 8
    forest_min_leaf: int = 15
    forest_min_split: int = 2
    forest_max_features: str = 'sqrt'
    ann_neurons: int = 32
    ann_second_neurons: int = 0
    ann_activation: str = 'relu'
    ann_learning_rate: float = 0.001
    ann_batch_size: int = 256
    ann_epochs: int = 80
    ann_patience: int = 8
    ann_alpha: float = 0.01
    balance_classes: bool = True

    def validate(self):
        bounds = {'forest_trees': (20, 200), 'forest_depth': (2, 12),
                  'forest_min_leaf': (1, 50), 'forest_min_split': (2, 20),
                  'ann_neurons': (8, 64), 'ann_second_neurons': (0, 32),
                  'ann_epochs': (10, 150), 'ann_patience': (3, 20), 'ann_batch_size': (64, 512)}
        for name, (minimum, maximum) in bounds.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise ValueError(f'{name} must be an integer from {minimum} to {maximum}.')
        if 0 < self.ann_second_neurons < 8:
            raise ValueError('Second ANN layer must be disabled (0) or contain 8–32 neurons.')
        if self.ann_batch_size not in (64, 128, 256, 512):
            raise ValueError('ANN batch size must be 64, 128, 256 or 512.')
        if self.forest_max_features not in ('sqrt', 'log2') or self.ann_activation not in ('relu', 'tanh'):
            raise ValueError('Choose supported maximum-feature and activation settings.')
        for name, minimum, maximum in [('ann_learning_rate', 0.0001, 0.01), ('ann_alpha', 0.0001, 1.0)]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not minimum <= value <= maximum:
                raise ValueError(f'{name} must be a finite number from {minimum} to {maximum}.')
        if not isinstance(self.balance_classes, bool):
            raise ValueError('Class balancing must be enabled or disabled.')
        return self

    def to_dict(self):
        return asdict(self)

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    def forest_parameters(self):
        return {'n_estimators': self.forest_trees, 'max_depth': self.forest_depth,
                'min_samples_leaf': self.forest_min_leaf, 'min_samples_split': self.forest_min_split,
                'max_features': self.forest_max_features, 'max_leaf_nodes': 512,
                'class_weight': 'balanced_subsample' if self.balance_classes else None,
                'n_jobs': 1, 'random_state': SEED}

    def ann_parameters(self):
        layers = (self.ann_neurons,) + ((self.ann_second_neurons,) if self.ann_second_neurons else ())
        return {'hidden_layer_sizes': layers, 'activation': self.ann_activation, 'solver': 'adam',
                'alpha': self.ann_alpha, 'batch_size': self.ann_batch_size,
                'learning_rate_init': self.ann_learning_rate, 'random_state': SEED,
                'max_iter': 1, 'shuffle': True}

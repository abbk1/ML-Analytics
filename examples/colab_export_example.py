"""Run in Colab: export safe result formats accepted by ML Analytics Studio.

Replace this public synthetic example with your own experiment. Do not upload
confidential datasets to a public notebook. No executable model artifacts are imported.
"""
import json
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=300, n_features=8, weights=[0.9, 0.1], random_state=42)
y = y.astype(str)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                      LogisticRegression(class_weight='balanced', max_iter=300, random_state=42))
model.fit(X_train, y_train)
predicted = model.predict(X_test)
values = {'accuracy': accuracy_score(y_test, predicted),
          'balanced_accuracy': balanced_accuracy_score(y_test, predicted),
          'f1_macro': f1_score(y_test, predicted, average='macro', zero_division=0)}
pd.DataFrame([{'model': 'Colab logistic regression', **values}]).to_csv('metrics.csv', index=False)
pd.DataFrame({'actual': y_test, 'predicted': predicted}).to_csv('predictions.csv', index=False)
classes = sorted(np.unique(y).tolist())
with open('experiment.json', 'w') as output:
    json.dump({'schema_version': 1, 'experiments': [{
        'model_name': 'Colab logistic regression', 'experiment_name': 'Synthetic example',
        'classes': classes, 'metrics': values,
        'confusion_matrix': confusion_matrix(y_test, predicted, labels=classes).tolist(),
        'config': {'preprocessing': {'scaling': 'standard'}, 'resampling': {'method': 'none'}},
    }]}, output, indent=2)
print('Created metrics.csv, predictions.csv and experiment.json. Download them from the Colab file browser.')

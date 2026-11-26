"""Prediction reuses the fitted pipeline; it never fits preprocessing."""
import numpy as np
import pandas as pd
from core.evaluation import model_response, positive_scores


def predict_batch(result, frame):
    if result.get('pipeline') is None:
        raise ValueError('Imported metrics do not contain a fitted prediction pipeline.')
    features = result['config']['features']
    missing = set(features) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing required input columns: {sorted(missing)}')
    if len(frame) > 50000:
        raise ValueError('Prediction batches are limited to 50,000 rows.')
    X = frame.loc[:, features].copy()
    preprocessor = result['pipeline'].named_steps['preprocess']
    if result['config']['preprocessing']['task'] != 'Text':
        numeric = next((cols for name, _, cols in preprocessor.transformers_ if name == 'numeric'), [])
        for column in numeric:
            try:
                X[column] = pd.to_numeric(X[column], errors='raise')
            except (ValueError, TypeError) as exc:
                raise ValueError(f'{column} must contain numerical values.') from exc
    pipeline = result['pipeline']
    predictions = pipeline.predict(X)
    response, kind = model_response(pipeline, X)
    if result.get('threshold') is not None:
        positive = result['positive_class']
        negative = next(c for c in result['classes'] if c != positive)
        predictions = np.where(positive_scores(response, result['classes'], positive) >= result['threshold'], positive, negative)
    output = frame.copy()
    output['prediction'] = predictions
    if kind == 'probability':
        for i, label in enumerate(result['classes']):
            output[f'probability::{label}'] = response[:, i]
    elif response is not None and response.ndim == 1:
        output['decision_score'] = response
    return output

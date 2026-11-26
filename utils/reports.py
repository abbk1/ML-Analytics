"""Evidence-based, portable experiment reports."""
from core.artifacts import dumps, metadata


def scientific_report(result, comparisons=None):
    config = result.get('config', {})
    lines = ['# ML Analytics Studio — Experiment report', '',
             f"Model: {result['model_name']}", f"Experiment: {result.get('experiment_name', '')}",
             f"Source: {result['source']}", f"Timestamp (UTC): {result.get('timestamp', '')}",
             f"Dataset: {result.get('dataset_name', 'External / unspecified')}",
             f"Dataset dimensions: {result.get('dataset_shape', 'Not supplied')}",
             f"Dataset fingerprint: {result.get('dataset_fingerprint', 'unknown')}",
             f"Holdout identity: {result.get('split_id', 'unknown')}", '', '## Methodology', '',
             'Local training uses stratified holdout splitting. Imputation, encoding, scaling, TF-IDF, '
             'and resampling are fitted inside training folds. Test observations keep their original distribution. '
             'Candidate ranking uses training-only cross-validation, not holdout results. '
             'External imports have unverified methodology.', '', '## Configuration', '', '```json',
             dumps(config), '```', '', '## Model parameters', '', '```json',
             dumps(result.get('model_parameters', {})), '```', '', '## Results', '']
    for name, value in result['evaluation']['metrics'].items():
        lines.append(f'- {name}: {value:.6f}')
    lines.extend(['', f"Cross-validation mean: {result.get('cv_mean', 'Not available')}",
                  f"Cross-validation standard deviation: {result.get('cv_std', 'Not available')}",
                  f"Final fit time (seconds): {result.get('fit_seconds', 'Not available')}", '',
                  '## Class imbalance', ''])
    for name, distribution in result.get('distribution', {}).items():
        counts = distribution['count']
        lines.append(f"- {name}: {dict(zip(distribution['class'], counts))}; "
                     f"majority/minority ratio: {counts.max() / counts.min():.3f}")
    lines.extend(['', '## Model comparison', ''])
    if comparisons is not None and not comparisons.empty:
        lines.extend(['Candidate selection uses training CV. Holdout comparisons with different or external '
                      'split identities are non-equivalent. CV means with different primary metrics or validation '
                      'designs are not directly comparable.', '', '```csv', comparisons.to_csv(index=False), '```'])
    else:
        lines.append('No additional candidate comparison was included in this report.')
    lines.extend(['', '## Limitations', '',
                  'Single-session results are exploratory and do not establish causal or universal model superiority. '
                  'Repeated comparison on the same holdout may influence subsequent choices; reserve an independent '
                  'final dataset for confirmatory work. CV folds are dependent; standard deviation is descriptive, '
                  'not a confidence interval. Grouped or temporal data require a separate validation design. '
                  'Class-weighting and resampling can change probability calibration. Free cloud resources are bounded.', ''])
    lines.extend(f'- {message}' for message in result.get('warnings', []))
    return '\n'.join(lines)

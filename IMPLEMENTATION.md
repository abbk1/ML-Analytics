# Focused redesign

The application now uses one complete public UCI dataset (30,000 rows, 23 predictors) and exactly two classifiers, ANN and Random Forest. It replaces the preceding general-purpose/fraud/Colab interface; that code remains in Git history at `0e85ed4`.

## Files

- `app.py`: stable Streamlit entry point.
- `default_studio/data.py`: checksum-verified loading, fixed feature schema and prediction validation.
- `default_studio/modeling.py`: shared stratified holdout, bounded forest, explicit ANN validation/early stopping, metrics and prediction.
- `default_studio/ui.py`: Dataset, Train, Compare and Predict pages; session-scoped fitted models.
- `data/`: full compressed dataset, provenance, attribution and checksum metadata.
- `scripts/prepare_dataset.py`: checksum-pinned, reproducible dataset preparation.
- `tests/`: integrity, real full-dataset model training, leakage, metrics and four-page UI integration.
- `requirements.txt` / `requirements-dev.txt`: runtime and test dependencies separated.

The source dataset was acquired from a pinned public mirror and its CSV matched every cell in the mirror's original-format workbook. The compressed bundle is 979,768 bytes. The running app needs no dataset download, Excel parser, external service or secret.

## Validation

The development validation run passed **18 tests**, including actual full-data training of both models. A clean installation of the 44 development/runtime packages also passed all 18 tests in approximately 9.8 seconds, with peak test-process RSS approximately **343 MiB**. The retained setup script passed the same suite, and Streamlit health/HTTP startup checks passed on port 8501. These observations describe this machine and test process; they do not establish identical timing/memory on Streamlit Community Cloud.

The tests assert 24,000 development + 6,000 untouched test rows, no missing or discarded records, correct target counts, ID exclusion, deterministic shared holdout indices, ANN fit/validation/test disjointness, preprocessing statistics learned only from eligible fitting rows, real probability-based metrics, bounded model parameters, safe prediction input, all four pages, explicit training action and reuse across navigation.

The ANN fits preprocessing/weights on 21,600 development rows and selects its checkpoint on 2,400 separate validation rows. The forest fits on all 24,000 development rows. No held-out test record fits preprocessing, classifier weights, epoch count or threshold. A fixed 0.50 threshold is used for both models.

See README.md for deployment and limitations. The app has been validated locally in the prepared cloud environment; publishing a remote Streamlit app and restoring an environment snapshot require independent verification.

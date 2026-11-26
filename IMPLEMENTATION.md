# Implementation record

The starter repository contained `app.py`, an unpinned analytics dependency list, a short README, and a Python 3.11 devcontainer. The implementation followed the requested six phases, prioritizing the scientific engine before expanding presentation. Validation was consolidated after the phases; the counts below refer to the actual final suite rather than separate phase-time runs.

## Phase 1 — Foundation

**Created:** `.gitignore`, `.streamlit/config.toml`, `core/__init__.py`, `core/config.py`, `core/data.py`, `utils/__init__.py`, `utils/session.py`, `views/__init__.py`, `views/data_pages.py` (dataset interface).

**Modified:** `app.py` replaces the starter with the retained Streamlit entry point, sidebar navigation, Beginner/Advanced modes and explicit session reset. `requirements.txt` pins the tested direct dependencies and adds the requested pytest runner. The existing `.devcontainer/devcontainer.json` remains unchanged.

**Implemented:** CSV/TSV/.xlsx uploads, bounded parsing and memory, audits, target/features/task selection, explicit sample/reset/replacement and result invalidation with retained history. Dashboard metrics derive only from actual uploaded data and results.

**Validation:** `tests/test_data.py` contains 10 passing cases for loading, invalid inputs, audit, cleaning, fingerprinting and reproducible stratified sampling. UI tests cover empty state and dataset selection.

**Limitations:** `.xlsx` first worksheet only; dates need feature engineering. Audits cannot identify all domain-specific leakage. Grouped and temporal validation require a separate design.

## Phase 2 — Data preparation

**Created:** `core/preprocessing.py`, `core/resampling.py`, `utils/charts.py`. **Extended:** `views/data_pages.py` with EDA, preprocessing and imbalance pages.

**Implemented:** selected-feature EDA, explicit visualization samples, text statistics, training-only imputation/scaling/encoding/TF-IDF, unknown-category handling, preservation of negation, configurable class-balance heuristics, distributions before/after sampling and unchanged test distributions. SMOTENC uses categorical-aware resampling before optional one-hot encoding.

**Validation:** `tests/test_preprocessing.py` contains 20 passing cases, including all five numerical scaling choices, all four imputation choices, unseen categories, training-only statistics and vocabulary, sparse text, random sampling/SMOTE/SMOTENC/SMOTE-Tomek, incompatible combinations, minority support and allocation limits. Mixed/text UI pages render without exceptions.

**Limitations:** English stop-word filtering is optional; ordinal category ordering must be justified. TF-IDF SMOTE synthesizes vectors, not language. Dense-only models explicitly reject large matrices.

## Phase 3 — Machine-learning engine

**Created:** `core/models.py`, `core/training.py`. **Created/extended:** `views/model_pages.py` for settings and training.

**Implemented:** all requested classifiers plus Dummy baseline and Linear SVM; parameter validation and bounded complexity; stratified/repeated CV; sequential training with one worker and bounded native threads; original/resampled/paired/custom strategies; shared holdout indices; fold-local pipelines; measured progress/time and warnings. Every failure is surfaced without changing the user's resampling method.

**Validation:** the training suite tests every classifier (Multinomial NB in the text-specific cases), binary/multiclass tasks, repeated CV, a successful 39:1 imbalance scenario, class weighting, and fold-specific SMOTE failures. Tests inspect every imputer's actual fit inputs to establish that CV and final fitting never see holdout rows.

**Limitations:** no automated hyperparameter search or OS-level per-fit resource sandbox. Accepted workloads can still exceed free hosting resources. MLP, SVC/KNN, tree depth/size, rows/features and fit counts have explicit safeguards; heavy work belongs in Colab.

## Phase 4 — Evaluation

**Created:** `core/evaluation.py`. **Extended:** `core/training.py`, `utils/charts.py`, `views/model_pages.py` with evaluation, comparison and error analysis.

**Implemented:** accuracy/balanced accuracy/error, macro/weighted/micro precision-recall-F1, MCC/Kappa, class support/specificity/FPR/FNR, binary positive-label metrics, ROC/AP from continuous responses only, one-vs-rest multiclass curves, matrices, class heatmaps, CV distributions/mean/std, training times, MLP loss, feature importance, filtered errors and downloads. Comparison warns on differing or external test identities and differing CV ranking metrics.

**Validation:** metric cases verify known values, matrix orientation, reversed positive labels, continuous-score AUC despite poor hard predictions and the absence of fake AUC when only labels exist. Shared holdout indices and unchanged class counts are asserted across candidates/strategies.

**Limitations:** descriptive CV standard deviations are not confidence intervals. Feature importance is impurity-based. Repeated holdout inspection can contaminate the scientific selection process; use independent confirmatory data.

## Phase 5 — Advanced functionality

**Created:** `core/prediction.py`, `core/artifacts.py`, `utils/reports.py`, `views/exchange_pages.py`, `examples/colab_export_example.py`. **Extended:** evaluation/training pages with OOF threshold analysis.

**Implemented:** fitted-pipeline single/batch prediction; binary threshold selection from training OOF scores only; session history and deletion; safe CSV/JSON metric/prediction/matrix imports and validated chart images; Colab export example; CSV/Excel/JSON/Joblib and scientific Markdown downloads; PNG through chart controls. Spreadsheet formula-like text is escaped. No uploaded script, notebook or executable model artifact is run or deserialized.

**Validation:** `tests/test_artifacts.py` contains 11 passing cases for schema validation, fractional metrics, matrix consistency, safe formats, JSON round-trip and spreadsheet exports. Training tests assert prediction does not change fitted imputer statistics and threshold selection receives only training row indices. The example script was executed outside the checkout; `metrics.csv`, `predictions.csv` and `experiment.json` each imported successfully.

**Limitations:** imports have unverified external methodology and cannot prove split equivalence; metric imports cannot predict. Sessions are temporary. Trusted pipeline exports require matching versions and the recorded threshold. Optional learning curves and nested-CV threshold evaluation are not included.

## Phase 6 — Quality assurance and documentation

**Created:** `tests/conftest.py`, `tests/test_data.py`, `tests/test_preprocessing.py`, `tests/test_training.py`, `tests/test_artifacts.py`, `tests/test_ui.py`, this record. **Modified:** `README.md` with installation, Community Cloud deployment, workflow guide, model/preprocessing compatibility, scientific limits and resource limits.

**Validation:** final suite: **69 passed, 0 failed, 0 skipped**. Six UI tests render all 15 pages in empty and trained sessions, cover imported results and sparse/mixed inputs, exercise actual file upload → target selection → paired training → evaluation → prediction → exports, and verify active-result invalidation with history retention. `uv pip check` validates all 55 installed packages. `git diff --check` passes. The real Streamlit service responds successfully to its health and HTTP application-shell checks; startup is idempotent and restartable using the environment's saved instructions.

**Environment:** isolated Python 3.11.18 at `/workspace/.cloud-env/ML-Analytics/venv`; helper scripts, caches and validation outputs remain outside the checkout. Installation uses the existing GitHub/package-manager network access and requires no secrets. Environment installation and startup instructions are saved separately in environment settings for review/publication.

**Remaining external steps:** review the code changes, commit/push them when appropriate, and deploy `app.py` with Python 3.11 in Streamlit Community Cloud. This work validates the current cloud instance; it does not claim a GitHub push, Community Cloud deployment, environment publication, or fresh-task snapshot restoration. Review/save the environment draft and publish it through the product to reuse the prepared snapshot.

## Credit-card fraud and large-dataset Colab update

The user confirmed the standard creditcard.csv schema (`Class=0/1`, `Time`, `Amount`, `V1`–`V28`). Added a fraud-specific workflow, explicit dataset preset and operating-point scorecards while preserving the general tools and Streamlit entry point.

**Created:** `core/fraud.py`, `views/fraud_page.py`, `examples/creditcard_colab.py`, `examples/build_fraud_notebook.py`, the self-contained `examples/credit_card_fraud_colab.ipynb`, `docs/CREDIT_CARD_FRAUD_COLAB.md`, and `tests/test_fraud_colab.py`.

**Modified:** `app.py`, `core/artifacts.py`, `core/evaluation.py`, `views/data_pages.py`, `views/model_pages.py`, `views/exchange_pages.py`, `utils/reports.py`, `tests/test_artifacts.py`, `README.md`, and this record.

**Behavior:** the full CSV can stay in Google Drive. Batch Colab training compares ANN/MLP, Random Forest and a baseline by validation average precision, selects a validation F2 operating point under an explicit precision floor, then tests only the selected candidate. The lower-memory mode reads every training row in chunks using SGD logistic and a frozen training-only scaler. Chronological boundaries keep equal timestamps together. Compact JSON imports preserve fraud label, threshold, validation comparison and bounded ROC/PR curves, without transaction features or executable model loading. Confusion metrics use weighted class pairs, so large test counts are not expanded into rows.

**Validation:** 81 automated tests pass, including 12 new cases for full-row batch/chunked training, train-only preprocessing, separate validation/test accesses, ANN/RF comparison, schema/order/memory/threshold failures, large-count imports, notebook-source consistency, bounded curve export round-trips, preset application, and a complete imported fraud dashboard without any uploaded source CSV.

**Limits:** actual user data was not supplied or trained. Colab execution/authentication must be performed by the user; trainer and generated notebook code are tested in the prepared Python environment using synthetic data. Incremental mode is SGD logistic rather than ANN/RF, needs a Time-sorted CSV, uses fixed zero imputation, and bounds evaluation arrays. Colab remains subject to finite RAM/CPU/session limits; no automatic checkpoint/resume is implemented. Imported results retain an unverified external split identity. The local app's prototype training uses stratified CV; the notebook defaults to chronological validation.

# ML Analytics Studio — Credit Card Fraud

A server-side classification workspace focused on **credit-card fraud detection**. For your full standard `creditcard.csv` (`Class=0` legitimate, `Class=1` fraud), train in **Google Colab** and import a small results JSON into Streamlit. For small prototypes, use the app's credit-card preset, training-only preprocessing and class-weighted models. The existing general tabular/text tools remain available.

**Dataset too large for Streamlit?** [Open the fraud Colab notebook](https://colab.research.google.com/github/abbk1/ML-Analytics/blob/main/examples/credit_card_fraud_colab.ipynb) and follow the [step-by-step guide](docs/CREDIT_CARD_FRAUD_COLAB.md). Put the CSV in Drive → train/validate in Colab → import only `fraud_results.json` in the app. No local Python installation or full-dataset Streamlit upload is required.

The notebook compares **ANN/MLP and Random Forest** when the full file fits Colab RAM. Its separate **incremental SGD logistic** mode reads a Time-sorted CSV in chunks when it does not fit; that mode uses a different lightweight model family. Both keep validation/test data out of fitted preprocessing and threshold/model fitting. Colab capacity is finite, and a GPU does not accelerate these scikit-learn models.

## Features

The sidebar contains Dashboard, Credit Card Fraud & Colab, Dataset Management, Exploratory Data Analysis, Data Preprocessing, Class Imbalance Analysis, Model Configuration, Model Training, Performance Evaluation, Model Comparison, Error Analysis, Prediction, Experiment History, Import Colab Results, Export and Reports, and Help and Documentation.

- Fraud-specific setup for `Class=1`, average-precision ranking and class weighting; scorecards show fraud detected/missed, legitimate transactions flagged/cleared, fraud recall and false alarm rate.
- A self-contained Colab notebook with batch ANN/Random Forest comparison or chunked incremental training, chronological train/validation/test separation, validation-only F2 threshold selection, compact report/curve imports and trusted models kept in Drive.

- CSV, TSV and Excel `.xlsx` loading with bounded uploads, previews, types, missing values, duplicates, unique counts, memory estimates, target/features, identifier/leakage/rare-class warnings and explicit stratified sampling.
- Numerical/categorical EDA, correlations, outliers, missingness, class distributions, and text lengths, word counts, vocabulary and frequent words. Charts explicitly sample up to 5,000 rows; training is never silently sampled.
- Training-only mean/median/mode/constant imputation, one-hot or ordinal encoding, Standard/MinMax/Robust/MaxAbs/no scaling. Optional, explicit dataset-level removal of missing targets, incomplete records and duplicate selected records.
- Text cleaning and TF-IDF with unigram/bigram/trigram features, document-frequency bounds, feature limits, sublinear TF, L1/L2/no normalization and optional English stop words that preserve `no`, `not`, `nor`, `never`.
- Original, random over/undersampling, SMOTE, mixed-feature SMOTENC and SMOTE-Tomek experiments. Class weights are available on supporting classifiers. Fold-specific neighbor and allocation checks reject incompatible combinations without substitution.
- Shared stratified holdout indices, stratified/repeated CV, explicit sequential training, measured progress/time, captured convergence warnings, and training-CV candidate ranking.
- Accuracy, balanced accuracy, error rate, macro/weighted/micro precision/recall/F1, MCC, Kappa, per-class support/specificity/FPR/FNR, and genuine probability/decision-response ROC-AUC and AP. Binary positive labels are configurable. Multiclass ROC/AP use macro one-vs-rest averaging.
- Confusion matrices, normalized matrices, metric/per-class charts, ROC/PR curves, CV distributions and mean/std charts, MLP loss and tree feature importance, and filtered row-level errors.
- Optional binary threshold selection using out-of-fold **training** predictions only; final predictions use the chosen threshold.
- Single and batch prediction through fitted pipelines, including unknown categories and validated numeric input.
- Session history, JSON/CSV Colab imports, optional reference chart images, CSV/Excel/JSON/Joblib exports, scientific Markdown reports, and PNG chart export through Plotly's camera control.

## Installation and development

Use Python **3.11**, matching the existing `.devcontainer/devcontainer.json`. The app also requires no GPU, database, Node.js, paid API, or backend service. Exact direct dependency versions in `requirements.txt` are the versions validated in this environment.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Run the automated suite from the repository root:

```bash
python -m pytest -q
```

In the prepared Codex environment the isolated interpreter is `/workspace/.cloud-env/ML-Analytics/venv/bin/python`. Setup artifacts and caches are outside the checkout. Use the existing checkout in each isolated cloud task; a separate Git worktree is unnecessary.

## Streamlit Community Cloud deployment

1. Commit the repository changes to GitHub when ready.
2. Create/select a Streamlit Community Cloud app for this repository and the intended branch.
3. Set the main file to **`app.py`** and Python to **3.11** in advanced settings.
4. Deploy. Streamlit installs `requirements.txt`; `.streamlit/config.toml` sets the theme, 25 MB upload limit, headless operation and disabled usage statistics.
5. Validate with a small nonconfidential classification CSV: load it, apply target/features, train the default candidates, inspect evaluation, and download a report.

The existing devcontainer image, forwarding configuration and deployment entry point remain intact. No secrets are required. CORS/XSRF protection remain enabled in the cloud environment startup command. Community Cloud deployment requires the user's GitHub/Streamlit access; local validation is not a remote deployment.

## User guide

1. **Dataset Management**: load a file and explicitly apply a target and features. For text, choose Text mode and one text column. `.xlsx` uses its first worksheet. Date features need engineering or exclusion; target labels are converted to strings.
2. **EDA**: inspect selected features, label balance, missingness and text characteristics. Audit warnings are advisory; the app never silently deletes data.
3. **Preprocessing**: save your missing-value, encoding, scaling and TF-IDF settings. Numerical scaling is separate from TF-IDF normalization. Preview numerical scaling fitted only to the training partition.
4. **Class Imbalance**: inspect the majority baseline and minority support. Severity thresholds are configurable heuristics, not scientific standards. Save an appropriate sampler, strategy and SMOTE neighbor count.
5. **Model Configuration**: save up to six candidates. Beginner mode exposes selected-model settings; Advanced mode exposes all model settings as validated JSON. Set `class_weight` to `"balanced"` for supporting models. Model parameters use scikit-learn names.
6. **Model Training**: choose A (original), B (resampled), C (paired), or a custom saved strategy. Choose test fraction, CV folds/repeats, seed, primary metric and binary positive class. Click **Train selected models**. Editing widgets never triggers training.
7. **Evaluate**: inspect confusion matrices, class recall/F1, continuous-response metrics, training time, CV and warnings. Macro F1 is the default ranking metric; candidate selection uses training CV. A majority/prior Dummy model provides a baseline.
8. **Compare**: filter experiments, inspect scaling/resampling trade-offs and confirm matching split identities. Different or external split identities are prominently marked non-equivalent. CV means with different primary metrics are not comparable.
9. **Predict**: choose a locally trained result. Enter one record or upload a CSV containing its required feature columns; the exact fitted pipeline and threshold are reused.
10. **History / Export**: download experimental metadata, actual split indices, package versions, reports, metrics, errors and fitted pipelines. Remove old results to release memory. Dataset/config changes invalidate current active results but preserve historical experiments with their original settings.

### Models and compatibility

| Classifier | Settings / compatibility |
|---|---|
| Dummy baseline | Prior/majority reference |
| MLP | Hidden layers, activation, solver, learning rate/init, alpha, batch size, iterations, early stopping; scaling recommended |
| Random Forest | Estimators, finite depth, split/leaf size, max features, class weights |
| Logistic Regression | C, solver, penalty, elastic-net ratio, iterations, class weights; solver/penalty compatibility checked |
| SVM / Linear SVM | C, kernel/gamma/degree where applicable, class weights; SVC probability estimation off; decision scores available |
| Decision Tree | Finite depth, split/leaf size, class weights |
| KNN | Neighbors, weights, distance parameter; scaling recommended |
| Gaussian NB | Small tabular inputs only; bounded dense conversion |
| Multinomial NB | Nonnegative sparse TF-IDF text only; no scaler that can introduce negatives |
| Bernoulli NB | Alpha and binarization threshold |
| Gradient Boosting | Estimators, finite depth, learning rate; small tabular inputs only |

One-hot encoding handles unseen categories safely. Ordinal encoding imposes artificial ordering for nominal columns and requires justification. SMOTENC imputes/ordinal-encodes mixed data inside a training fold, resamples categorical-aware values, then optionally one-hot encodes them. Ordinary SMOTE on categorical features is rejected. Sparse text stays sparse. Gaussian NB and Gradient Boosting are disabled for text; dense conversion for these tabular models is explicit and bounded.

### Colab interoperability

Download `examples/colab_export_example.py` from the import page or run it in Colab. It uses synthetic public data and produces:

- `metrics.csv`: a `model` column plus supported fractional metric columns such as `accuracy`, `balanced_accuracy`, `f1_macro`.
- `predictions.csv`: `actual,predicted`, with nonmissing class labels. No probability metrics are invented from these hard labels.
- `experiment.json`: `{"schema_version":1,"experiments":[...]}`, including ordered `classes`, a nonnegative integer `confusion_matrix`, model name and metrics. Matrix-derived metrics must agree with supplied metrics.

A single JSON confusion-matrix object is also accepted. Metric tables are limited to 200 results and prediction tables to 50,000 rows. Imports cannot establish identical holdout observations and are marked external. Uploaded scripts/notebooks are never executed, and uploaded Pickle/Joblib models are never deserialized. Optional PNG/JPEG reference charts are bounded and validated. Exported Joblib artifacts are for **trusted** use only and include the fitted pipeline and metadata (including any selected threshold); loading the pipeline alone does not apply the separately recorded threshold.

## Scientific safeguards and limitations

Preprocessing and resampling are inside sklearn/imbalanced-learn pipelines and fit exclusively on training data, including each CV fold. The same holdout observations are shared across candidates and strategies within a run. Test labels are not used to fit features, tune thresholds or rank candidates. Imputation/scaling tests directly check training-only statistics; CV tests inspect rows seen by each fold's imputer.

- This is independent-record, binary/multiclass classification (2–50 classes), not regression, multilabel, temporal or group-aware validation. Group/patient/time-dependent data require an appropriate design outside this version.
- Duplicate and conflicting-label warnings cannot prevent every form of domain-specific leakage. Review identifiers, repeated entities and post-outcome features. Duplicate removal is opt-in; remaining repeated feature records can appear across partitions.
- Balanced accuracy equals macro recall for standard single-label classification. Accuracy alone does not establish superiority; inspect minority recall/F1 and PR analysis.
- Training-CV means/std are descriptive; overlapping CV folds are dependent. Threshold search uses OOF training F1; its optimized score is exploratory, not an unbiased nested-CV estimate. CV candidate ranking uses default decision thresholds.
- The final test set remains untouched by fitting. However, repeatedly inspecting its scores and changing the experiment contaminates model selection. Reserve an additional independent final dataset for confirmatory claims.
- Decision scores are not calibrated probabilities. Resampling and class weighting can change calibration. SMOTE on TF-IDF creates synthetic vectors rather than real language.
- Feature importance is impurity-based and can favor high-cardinality features; it is not causal interpretation. Exports make no unsupported scientific superiority claims.
- Missing/undefined response metrics are labeled Not applicable. Learning curves and automatic hyperparameter searches are not included; controlled explicit experiments avoid unbounded compute.

## Cloud resource and storage limits

Uploads are capped at 25 MB; parsed frames at 150 MB, 500 columns and 500,000 rows. Expanded Excel workbooks have a separate 200 MB limit. Local training accepts at most 50,000 rows. Larger data requires explicit reproducible stratified sampling or Colab; training data is never reduced silently.

Training is sequential with one worker and bounded native threads. There are at most 6 candidates, 3 strategies and 120 fits per run, 30 results per session, 100,000 resampled rows, a 256 MB estimated resampling workspace, and a 150 MB / 50,000-column transformed-matrix limit. Dense-only models are capped at 2 million cells. MLP permits at most 4 layers of 256 neurons, 10,000 transformed features and 20,000 training rows. SVC/KNN permit at most 10,000 training rows. Tree depths, estimator counts, iterations, vocabulary and repeats are bounded. These guards reduce resource risk; they do not guarantee every accepted dataset fits free hosting or impose an OS-level per-fit time/memory sandbox. Reduce complexity or move heavy experiments to Colab.

Data and models are session-scoped and never globally cached. Community Cloud storage/session persistence is not guaranteed. Download results for long-term storage. Error tables and exports may include original records. CSV/Excel downloads escape formula-like text. Clearing the session releases its references; server infrastructure still determines data retention.

## Structure

- `app.py`: deployable entry point and navigation.
- `views/`: data, training/evaluation, and exchange page renderers.
- `core/`: serializable settings, data auditing, model registry, fold-local preprocessing/resampling, training, evaluation, prediction and safe artifacts.
- `utils/`: charts, state invalidation and scientific reports.
- `tests/`: synthetic-data unit/integration/UI tests.
- `examples/`: executable Colab export example.

See `IMPLEMENTATION.md` for phase-by-phase file changes, validation evidence and remaining limitations.

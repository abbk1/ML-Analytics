# Credit Default Studio

A simple Streamlit application for **credit-card payment default prediction** using one complete public dataset and exactly two models: **ANN** and **Random Forest**.

The full UCI **Default of Credit Card Clients** dataset is bundled with the repository: **30,000 customers, 23 predictors**, and a binary next-month default label. It is not a transaction-fraud dataset. No upload, Colab session, database, GPU, API key or runtime dataset download is required.

## Four pages

1. **Dataset** — understand the question, actual class distribution, feature groups and source; preview the data.
2. **Train** — one button trains both bounded CPU models. Training only happens on this action; fitted models remain in session state for page changes and prediction.
3. **Compare** — view metrics, confusion matrices, ROC/precision–recall curves and ANN learning progress; download the comparison CSV and reproducibility JSON.
4. **Predict** — edit a public example customer, enter six months of repayment history, and obtain predictions from both fitted models.

There are no general-purpose dataset selectors, uploaders, model registries, experiment-history pages, or Colab workflows. The preceding fraud application remains available in Git history at commit `0e85ed4`.

## The complete dataset

All **30,000 original records** are retained, with **23,364 non-defaults** and **6,636 defaults** (22.12%). No sampling, category cleanup, duplicate removal or resampling changes them. Only the target column header is normalized to `default_payment_next_month`.

- Target **1**: default payment next month; **0**: no default.
- Inputs: credit limit, customer characteristics, repayment status, six bill balances and six previous payment amounts.
- Currency: New Taiwan dollars (NT$). Historical payment data covers April–September 2005 in Taiwan.
- `ID` is retained for traceability and split export, **excluded from model inputs**.
- Bundled gzip CSV: approximately **0.94 MiB**, so fresh Streamlit deployments do not fetch or parse the original Excel workbook.
- [UCI dataset / citation](https://doi.org/10.24432/C55S3H), I-Cheng Yeh (2009), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

See [data/README.md](data/README.md) and `data/metadata.json` for provenance, checksums and category details. The dataset was acquired from an immutable public mirror; every CSV cell was checked against the original-format workbook in that mirror. The application verifies the bundled checksum, schema, IDs and original class counts before use.

## Training and evaluation

Both models share a deterministic, stratified **80/20 split** (seed 42):

| Partition | Customers | Use |
|---|---:|---|
| Development / training | 24,000 | Fit the models; includes the ANN's internal validation subset |
| Held-out test | 6,000 | Evaluate both models after fitting is complete |

**Random Forest:** 80 trees, maximum depth 8, minimum leaf size 15, balanced-subsample class weights, one worker. Categorical predictors are one-hot encoded; numerical values use a training-only median imputer without scaling.

**ANN:** one 32-neuron ReLU hidden layer with Adam, batch size 256, learning rate 0.001, alpha 0.01 and balanced training sample weights. It divides the 24,000 development rows into **21,600 fit rows** and **2,400 early-stopping validation rows**. Its imputer, encoder and StandardScaler fit only the 21,600 rows. An explicit epoch loop restores the best validation-loss checkpoint, stopping after eight unimproved epochs, up to 80 epochs. The hidden validation data never fits preprocessing.

Categorical inputs include the demographic and repayment-status codes, preserving unknown/other categories. One-hot encoders safely ignore categories not seen during fitting. Numerical features remain separate. All model computations use a single native thread; the forest also uses `n_jobs=1`.

The default decision threshold is fixed at **0.50**; test labels never select a threshold, model parameter or ANN epoch. Results include accuracy, balanced accuracy, default-class precision/recall/F1, ROC-AUC and average precision from genuine probability scores. Both models receive the same held-out customers. The majority-class accuracy is an explanatory reference statistic, not a third trained model. The app does not declare or tune a winner from the test scores.

## Install and run

Use **Python 3.11**, matching the existing development container:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

For development and tests:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The tests use the actual full bundled dataset and train both full-size models once per suite. They check integrity, full row coverage, split reproducibility, train/validation/test separation, training-only learned statistics, exact metric calculation, bounded classifiers, prediction validation, all four pages and button-only training.

## Deploy on Streamlit Community Cloud

1. Select GitHub repository **abbk1/ML-Analytics**, branch **main**.
2. Keep **app.py** as the main file and select **Python 3.11** in advanced settings.
3. Deploy or reboot the existing app after GitHub updates. The root `requirements.txt` supplies only the runtime dependencies; the dataset is already included.
4. Open **Train**, click **Train ANN and Random Forest**, then inspect **Compare** and **Predict**.

Training is performed by the Streamlit server; users need no local Python installation. Models persist only for that session, and clearing/restarting it requires another training run. The public dataset is safely cached; user-entered prediction profiles are not globally cached. Capacity and training time depend on the hosting runtime and concurrent sessions. The bounded dataset/model sizes reduce resource demand but do not establish deployment or unlimited free-hosting capacity.

The existing `app.py` entry point and `.streamlit/config.toml` configuration remain the deployment interface. No special secrets or additional outbound domains are required for the running app.

## Rebuild the bundled data

This is a maintainer operation, not a runtime requirement:

```bash
python scripts/prepare_dataset.py
# Or provide the exact original mirror CSV locally:
python scripts/prepare_dataset.py /path/to/default_of_credit_card_clients.csv
```

The script checks the pinned source SHA-256 before preparing the gzip CSV and metadata. Do not disable checksum or TLS verification to accept a different file. The optional original Excel workbook is not a runtime dependency.

## Interpretation and limitations

This is a historical research comparison, not a current lending decision system. Demographic inputs require bias assessment; behavior and calibration may differ across populations and time. Class weighting changes the probability scale, so displayed probabilities are **uncalibrated model estimates**. A fixed 0.50 threshold is a reproducible benchmark, not a business cost policy.

One held-out split is not cross-validation or evidence of universal superiority. Repeatedly inspecting and tuning to it weakens its independence; additional temporal/external validation is necessary for confirmatory claims. ANN development validation and RF training allocations are documented separately above. All 30,000 customers contribute to fitting, validation or final evaluation; the final 6,000 never train either model.

The code is covered by the repository's MIT license. The bundled public dataset retains its separate CC BY 4.0 attribution and terms.

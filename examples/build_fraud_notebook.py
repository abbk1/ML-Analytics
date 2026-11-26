"""Generate the self-contained Colab notebook from the tested training module."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def build_notebook():
    def cell(kind, source):
        result = {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(keepends=True)}
        if kind == 'code':
            result.update(execution_count=None, outputs=[])
        return result
    cells = [
        cell('markdown', '''# Credit Card Fraud Studio · Full-dataset training in Colab

Your **creditcard.csv** stays in your own Google Drive. Colab performs training; Streamlit receives only a small JSON report.
`Class=0` is legitimate, `Class=1` is fraud. `V1`–`V28`, `Amount`, and `Time` are numerical features.

1. Run the dependency cell. If Colab asks you to restart the session, restart once and continue below.
2. Mount your Drive and set the CSV and output-folder paths.
3. Choose **batch** for ANN/Random Forest comparisons, or **incremental** for a file that does not fit RAM.
4. Run training. Download/import **fraud_results.json** in the app's **Import Colab Results** page.

No GPU is required: these scikit-learn classifiers use CPU. Free Colab sessions have finite RAM and can disconnect; this notebook does not promise unlimited data or background execution.
'''),
        cell('code', '%pip install -q numpy==2.4.6 pandas==3.1.1 scikit-learn==1.9.1 joblib==1.6.0\n'),
        cell('code', 'from google.colab import drive\ndrive.mount("/content/drive")\n'),
        cell('markdown', '''## Training engine
The cell below is embedded from the repository's tested standalone trainer. No repository clone or private GitHub token is needed.
Chronological splitting keeps earlier transactions in training (60%), later validation (20%) and final testing (20%). Equal timestamps stay together. Every partition must contain both classes.
'''),
        cell('code', (ROOT / 'creditcard_colab.py').read_text()),
        cell('markdown', '''## Configure your run
**Batch:** uses all rows, imputes/scales from training only, compares Random Forest/ANN/baseline on validation average precision, and evaluates only the selected candidate on the final test partition. The raw-array budget does not cover every estimator allocation.

**Incremental:** change `mode` to `"incremental"`. The CSV must already be sorted by nondecreasing `Time`. It reads chunks, uses fixed zero imputation, fits scaler statistics on training chunks only, and trains **SGD logistic**, not ANN/Random Forest. This is a different, lighter model family, not an equivalent ANN/RF experiment. Each epoch visits every training row. Validation/test responses remain bounded in RAM by the explicit evaluation budget.

Fraud recall is important, but false alerts have a cost. The threshold maximizes **validation F2**, subject to the precision floor you set below. This is a research operating point, not a business-approved decision rule. A failed constraint stops the run without consulting the holdout. Do not tune after reading final test results.
'''),
        cell('code', '''from datetime import datetime, timezone
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
config = FraudColabConfig(
    csv_path="/content/drive/MyDrive/creditcard.csv",
    output_dir=f"/content/drive/MyDrive/fraud_results/{RUN_ID}",
    mode="batch",                     # "incremental" for Time-sorted files too large for RAM
    split_method="chronological",     # "stratified" only for an independent-record batch research baseline
    models=("Random Forest", "MLP", "Dummy baseline"),
    chunk_size=50000,
    memory_budget_mb=1024,
    max_evaluation_rows=1000000,        # per validation/test partition; no silent sampling
    epochs=3,
    min_precision=0.10,                # explicitly choose an acceptable validation precision floor
    beta=2.0,
)
result = run_fraud_experiment(config)
'''),
        cell('markdown', '''## Bring results back to Streamlit
Open **Import Colab Results** in your deployed app and upload `fraud_results.json` from the run folder in Drive. It contains the test confusion matrix, fraud metrics, bounded ROC/PR curves, validation candidate comparison, threshold and training metadata. It contains no raw transaction rows.

Inspect **Performance Evaluation**, **Credit Card Fraud & Colab**, **Model Comparison**, and **Export and Reports**. Imported split claims remain externally unverified. Validation comparisons are marked separately from final test metrics.

Keep `trusted_fraud_model.joblib` in Drive for prediction in this trusted notebook. The app never loads uploaded model artifacts. For a new transaction CSV in Colab:
'''),
        cell('code', '''# Optional prediction: run only with a model artifact you created and trust.
# bundle = joblib.load(Path(config.output_dir) / "trusted_fraud_model.joblib")
# for chunk in pd.read_csv("/content/drive/MyDrive/new_transactions.csv", chunksize=50000):
#     X_new = chunk[bundle["features"]].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
#     probability = bundle["pipeline"].predict_proba(X_new)[:, 1]
#     chunk["fraud_probability"] = probability
#     chunk["fraud_alert"] = (probability >= bundle["threshold"]).astype(int)
#     # Save each chunk to a new output file in Drive; never overwrite source transactions.
'''),
    ]
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
        'colab': {'name': 'credit_card_fraud_colab.ipynb'},
        'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
        'language_info': {'name': 'python'}}, 'cells': cells}
    for index, item in enumerate(cells):
        item['id'] = f'fraud-cell-{index}'
    path = ROOT / 'credit_card_fraud_colab.ipynb'
    path.write_text(json.dumps(notebook, indent=2) + '\n')
    return path


if __name__ == '__main__':
    print(build_notebook())

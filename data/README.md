# Complete UCI Default of Credit Card Clients data

**Creator/citation:** Yeh, I. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H

**Dataset page:** https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

**License:** Creative Commons Attribution 4.0 International (CC BY 4.0): https://creativecommons.org/licenses/by/4.0/

The dataset license is separate from the application's MIT code license. Retain this attribution with redistributed data.

## Provenance

The original UCI hostname was unavailable under the development environment's network policy. The source CSV and original-format XLS workbook were obtained from the public mirror pinned at:

`https://github.com/MatteoM95/Default-of-Credit-Card-Clients-Dataset-Analisys/tree/585f77c33dff5a14dac8e5396bec24820d3db2f8/dataset`

The CSV and workbook were compared cell-for-cell and matched in all 30,000 × 25 values. The mirror URL, source CSV checksum, original-workbook checksum, and bundled checksum are recorded in `metadata.json`. `scripts/prepare_dataset.py` downloads only that exact CSV and rejects a checksum mismatch. Neither this code nor the Streamlit app executes mirror notebooks or scripts.

The bundled gzip CSV retains every original value, all original records in their original order, and all 30,000 IDs. The only column-name change is `default payment next month` → `default_payment_next_month`. Gzip timestamps are fixed for reproducible preparation.

## Schema

- **ID:** original customer record identifier, 1–30,000; never used as a predictor.
- **LIMIT_BAL:** credit limit in New Taiwan dollars.
- **SEX:** original codes 1 (male), 2 (female).
- **EDUCATION:** 1 graduate school, 2 university, 3 high school, 4 other; additional original codes 0, 5 and 6 are retained and labeled unknown in the UI.
- **MARRIAGE:** 1 married, 2 single, 3 other; original 0 is retained as unknown.
- **AGE:** years.
- **PAY_0, PAY_2, …, PAY_6:** repayment status for September through April 2005. `PAY_0` is September; there is no `PAY_1` column in the distributed dataset. Codes −2/−1/0 and positive delay codes are retained as categorical values rather than silently recoded or interpolated.
- **BILL_AMT1–6:** bill balances, September through April; negative balances are preserved.
- **PAY_AMT1–6:** previous payment amounts, September through April.
- **default_payment_next_month:** binary outcome, 0 no default and 1 default.

There are 23 predictors after excluding ID and target, no missing cells, 23,364 no-default records and 6,636 default records. No class balancing, sampling or duplicate removal changes the bundled population. Class weighting is applied only while fitting the models.

The payment period is historical. These labels are payment default, not fraudulent transactions. Customer characteristics and recorded performance should not be assumed representative of current customers or lending policy.

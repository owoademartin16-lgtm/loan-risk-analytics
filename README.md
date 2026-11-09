# Loan Repayment Risk — LendingClub Dashboard

A white-canvas Streamlit report on **9,578 LendingClub loans**, built directly
from `Loan Repayment Risk Analysis Report.ipynb`. It answers one question:
which borrowers end up **not fully paying** their loan?

* **Dashboard** — KPI summary and twelve figures reproduced from the report
  notebook, driven by nine slicers.
* **Predict Default** — score a single loan application with the report's best
  model, the tuned XGBoost pipeline.

The app never retrains the deployed model and never rewrites the data. The
only computations it performs are the notebook's own descriptive figures and a
deterministic re-run of the notebook's 80/20 test split for the model tab.

## The data

| Property | Value |
|---|---|
| Source | `loan_data.csv` from the *Machine-Learning-with-Python* dataset collection |
| Rows | 9,578 |
| Columns | 14 |
| Target | `not.fully.paid` (1 = not fully paid) |
| Fully paid | 8,045 loans (83.99%) |
| Not fully paid | 1,533 loans (16.01%) |
| Average interest rate | 12.26% |

Features the model consumes (notebook cell 25): `credit.policy`, `purpose`,
`int.rate`, `installment`, `log.annual.inc`, `dti`, `fico`,
`days.with.cr.line`, `revol.bal`, `revol.util`, `inq.last.6mths`,
`delinq.2yrs`, `pub.rec`.

## Features

### Dashboard page

* **KPI row** — total loans, not fully paid, non-full repayment rate and
  average interest rate for the current selection.
* **Nine slicers** — loan purpose, interest rate, monthly installment, credit
  policy, FICO, debt-to-income, inquiries, delinquencies and public records.
  They live in `app.py`, keep their state across pages, and every figure
  reacts to all of them at once.
* **Twelve figures**, each rebuilt with the exact notebook logic — same bins,
  band labels, titles, ordering, one colour per figure, outside percentage
  labels and `max × 1.15` axis headroom:

  | Tab | Figures |
  |---|---|
  | Overview | Non-Full Repayment Rate by Loan Purpose · by Interest Rate |
  | Borrower | by FICO Score · by Debt-to-Income Ratio |
  | Credit history | by Recent Credit Inquiries · Previous Delinquencies · Public Records |
  | Installments | by Monthly Installment (ten `pd.cut` bins) |
  | Model | model comparison · Decision Tree, XGBoost and tuned XGBoost confusion matrices · top-15 feature importance |

* The bar colours match the notebook cell by cell: `#1f77b4` (purpose),
  `#2ca02c` (interest), `#d62728` (FICO), `#ff7f0e` (DTI), `#9467bd`
  (inquiries), `#8c564b` (delinquencies), `#bcbd22` (public records) and
  `#1f77b4` (installment).
* Empty bands are kept rather than dropped, exactly as in the notebook
  (for example the empty "Below 600" FICO band).

### Predict Default page

* A form built from the 13 training columns. Income is entered in dollars and
  log-transformed because the model consumes `log.annual.inc`.
* Numeric defaults are the observed medians; `purpose` defaults to the mode.
* On submit the one-row frame is scored by the saved tuned XGBoost pipeline and
  shown as a colour-coded card: **Low** (< 20%, green), **Medium** (20–50%,
  amber), **High** (> 50%, red), with the probability, a risk meter and the
  three bands explained.

## Model

The notebook compares three models on a stratified 80/20 split
(`random_state=42`, 1,916 test loans):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Decision Tree | 0.735 | 0.184 | 0.189 | 0.186 | 0.514 | 0.165 |
| XGBoost | 0.687 | 0.243 | 0.450 | 0.315 | 0.656 | 0.259 |
| **Tuned XGBoost** | **0.642** | **0.240** | **0.567** | **0.337** | **0.667** | **0.285** |

The tuned XGBoost (`RandomizedSearchCV`, 30 iterations, 5-fold stratified CV)
is the selected model: it has the best recall, F1, ROC-AUC and PR-AUC, which is
what a repayment-risk screen needs. Its confusion matrix is
`[[1057, 552], [133, 174]]` — 174 of the 307 problem loans are caught.

The Model tab recomputes all three models deterministically from the same
split, so its scores and matrices reproduce the notebook exactly.

## Project structure

```
.
├── app.py                        # entry point: page config, navigation, slicers
├── loan_report.py                # data, slicers, notebook figures, model evaluation
├── pages/
│   ├── dashboard.py              # KPI row + the twelve figures
│   └── predict_default.py        # application form + risk assessment card
├── Loan Repayment Risk Analysis Report.ipynb   # the source analysis
├── data/
│   └── loan_data.csv             # the 9,578-loan dataset
├── model/
│   └── repayment_xgb_model.pkl   # the tuned XGBoost pipeline
├── utils.py                      # shared white-theme CSS and risk-card helpers
└── requirements.txt
```

The earlier analysis (`Loan Risk Analytics.ipynb`, `data/loans_clean.parquet`,
`model/default_model.joblib`, `pages/` history and the legacy single-page
modules) is kept in the repository for reference but is no longer used by the
app.

## Installation

Requires Python 3.13 (3.9+ should work).

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open <http://localhost:8501>.

## Styling

The global stylesheet lives in `utils.py` (`STYLE_CSS`) and is injected by
`app.py` and each page script. The design is a flat white canvas with a soft
neutral sidebar rail (`#E7EBEA`), quiet white stat cards with a hairline
border (`#E2E5E9`) and one colour rule per KPI. Typography is **Inter**.

The figures keep the report's own palette rather than the app accent colour,
so each bar looks exactly as it does in the notebook. The bars themselves use
a Tableau-style treatment: left-aligned titles, slim rounded bars, quiet zero
baselines, grey axis labels and no grid lines or tick marks. The prediction
risk card keeps green `#0F7A52` / amber `#F59E0B` / red `#C0392B` for low /
medium / high.

## Analytical limitations

- **Correlation, not causation.** Every dashboard figure is a descriptive
  comparison between groups.
- **Small book.** 9,578 loans and 1,533 problem cases; some slicer
  combinations leave very few rows, and rates move quickly.
- **Class imbalance.** Only 16% of loans are not fully paid. Accuracy is a
  misleading headline metric, which is why the model tab reports recall, F1
  and PR-AUC next to it.
- **Prediction is not a decision.** The probability is a relative risk score.
  A single score should never be the sole basis for approving or rejecting an
  application.
- **Historical data.** The book is a fixed historical extract, not a live
  lending portfolio.

# Loan Risk Analytics — Loan Default Dashboard

A multi-page Streamlit app for monitoring loan default risk on the *LC loans
granting model dataset* (1,347,681 loans, Jun 2007 – Dec 2018).

* **Dashboard** — KPI summary and eight Plotly charts that react to the
  sidebar filters.
* **Predict Default** — score a single loan application with the trained
  Random Forest pipeline and get a green / amber / red risk assessment.

The analysis that produced the data and the model lives in
`Loan Risk Analytics.ipynb`; the app itself never retrains the model, never
re-cleans the data and never writes to either asset.

## Features

### Dashboard page

* **KPI row** — total loans, defaulted loans, overall default rate and
  average loan amount for the current selection. Each card carries an icon
  badge in its own accent colour.
* **Sidebar filters** — loan purpose and loan issue year range. The filters live in `app.py`, so they keep their state while you
  move between pages.
* **Eight charts**, each with a hover tooltip that shows the group's total
  loans, its defaulted loans and its default rate:

  | Chart | Notes |
  |---|---|
  | Outcome Distribution | Fully paid (`#2E86AB`) vs defaulted (`#E4572E`) |
  | Default Rate By issue year | Line chart; warns that the newest cohorts are understated |
  | Default Rate By loan amount band | 0–5k … 30k+ |
  | Default Rate By FICO band | Below 580 … 740+ (empty bands are dropped) |
  | Default Rate By DTI band | 0–10 … 50+ |
  | Default Rate By employment length | Natural order, `Missing` last |
  | Default Rate By loan purpose | Horizontal, ≥ 100 loans, sorted by rate |
  | Top 10 States By Default Rate | Horizontal, ≥ 100 loans, full state names |

* Bars carry percentage labels and every y-axis has headroom so the labels
  are never clipped. The charts are grouped into three tabs
  (*Overview*, *Credit Risk*, *Purpose & Geography*) and laid out in
  two columns. The tabs are styled as a segmented control with icons and a
  sliding highlight.

### Predict Default page

* A `st.form` built from `model/features.joblib` — the exact columns the
  model was trained on:
  `loan_amnt`, `fico_n`, `dti_n`, `emp_length`, `purpose`,
  `home_ownership_n`, `addr_state`.
* Numeric inputs are `number_input`s bounded by the observed minimum and
  maximum and defaulting to the median; categorical inputs are `selectbox`es
  populated with the values found in the data.
* Features that carry too little of the model's importance to be worth
  asking about (currently employment length and state) are filled in
  automatically with the median (numeric) or the mode (categorical).
* On submit a one-row DataFrame with exactly the model's columns, in order
  and with matching dtypes, is passed to `predict_proba`. The result is shown
  as a colour-coded risk assessment card:
  **Low** (< 20 %, green), **Medium** (20–50 %, amber), **High** (> 50 %, red).
  The card carries the predicted probability, the verdict, a segmented risk
  meter with a marker at the predicted probability, an explanation of the
  three bands and a tinted summary strip.

## Project structure

```
.
├── app.py                       # entry point: page config, navigation, sidebar filters
├── utils.py                     # cached loaders, banding, summaries, Plotly helpers
├── pages/
│   ├── dashboard.py             # KPI row + the eight charts
│   └── predict_default.py       # application form + risk assessment card
├── Loan Risk Analytics.ipynb    # the analysis notebook that produces the assets below
├── data/
│   └── loans_clean.parquet      # the cleaned loan data
├── model/
│   ├── default_model.joblib     # the trained sklearn pipeline (preprocessor + forest)
│   └── features.joblib          # the exact input column list
├── requirements.txt
└── legacy_single_page_app.py    # the earlier single-page version, kept for reference
```

The supporting modules of the first single-page build (`dashboard.py`,
`prediction_page.py`, `predictor.py`, `loan_data.py`, `charts.py`,
`styles.py`) are kept in the repository as well, next to
`legacy_single_page_app.py`, so the earlier version stays runnable.

`utils.py` holds everything that is shared — the `@st.cache_data` data
loader, the `@st.cache_resource` model loader, the band definitions, the
grouped summaries and the Plotly figure builders — so both pages stay short.

## Installation

Requires Python 3.13 (3.9+ should work).

```bash
pip install -r requirements.txt
```

`scikit-learn` is pinned to **1.9.1** because the saved pipeline has to be
unpickled with a compatible version.

## Usage

```bash
streamlit run app.py
```

Then open the URL printed in the terminal (default <http://localhost:8501>).

## Data and model assets

The three assets the app reads are the artefacts written by
`Loan Risk Analytics.ipynb`:

| Asset | What it is |
|---|---|
| `data/loans_clean.parquet` | The cleaned loan data (1,347,681 rows, 15 columns). |
| `model/default_model.joblib` | The trained pipeline: median imputation → one-hot encoding → 200-tree Random Forest with balanced class weights. |
| `model/features.joblib` | `['loan_amnt', 'fico_n', 'dti_n', 'emp_length', 'purpose', 'home_ownership_n', 'addr_state']` — the columns the pipeline consumes. |

The raw `loan_risk_analytics.csv` (167 MB) is **not** part of the repository:
GitHub rejects any file above 100 MB. The notebook downloads the same file
directly from its Zenodo record (record `11295916`) and writes the cleaned
parquet plus the model artefacts, so the pipeline below reproduces everything
this app reads.

`random_forest_loan_model.pkl` is the same pipeline, kept in the earlier
pickle format for the legacy `predictor.py` module.

### Columns

| Column | Meaning |
|---|---|
| `id` | Unique loan identifier |
| `issue_d` | Loan issue month, `Mon-YYYY` (e.g. `Dec-2015`) |
| `revenue` | Borrower revenue/income figure as published; not used in any KPI |
| `dti_n` | Debt-to-income ratio as published (observed range 0–999) |
| `loan_amnt` | Funded loan amount in dollars |
| `fico_n` | FICO score (observed range 612–847.5) |
| `experience_c` | Binary experience flag (0/1) |
| `emp_length` | Employment length (`< 1 year` … `10+ years`, `NI` = not indicated) |
| `purpose` | Stated loan purpose (14 categories) |
| `home_ownership_n` | `MORTGAGE`, `RENT`, `OWN`, `OTHER` |
| `addr_state` | Borrower state (51 values incl. DC) |
| `zip_code` | Masked zip code |
| `Default` | **Target**: 0 = fully paid, 1 = defaulted |
| `title` | Loan title free text |
| `fico_group` | Binned FICO score (`600–649` … `800–849`), written by the notebook |

### Data handling

* No cleaning, imputation or feature engineering is performed by the app.
  The only derived column is `issue_year`, which is parsed from `issue_d` so
  the trend chart and the year filter have something to group on.
* `emp_length = "NI"` is shown as **Missing** on the employment-length
  chart; the stored value is untouched.
* Default rate = mean of `Default` × 100 within each group, using that
  group's own loan count as the denominator.
* Purpose and state rankings only include groups with at least **100 loans**
  in the current selection.
* Empty groups are dropped rather than charted as zeroes; an empty filter
  selection shows a notice instead of a crash.

## Analytical limitations

- **Correlation, not causation.** Every chart is a descriptive comparison
  between groups.
- **Cohort maturity.** Loans issued in recent years have had less time to
  default, so the issue-year default rate is not comparable across cohorts.
  The newest cohorts are therefore understated.
- **Minimum-sample threshold.** Purpose and state rankings exclude groups
  with fewer than 100 loans; small groups produce noisy rates.
- **Data ends December 2018.** The portfolio is historical.
- **Prediction is not a decision.** The model was trained with balanced
  class weights, so its probabilities lean towards the minority class and
  should be read as a relative risk score, not as a portfolio default
  frequency. A single score should never be the sole basis for approving or
  rejecting an application.

## Styling

The global stylesheet lives in `utils.py` (`STYLE_CSS` and `ICON_CSS`) and is
injected with `st.html()` by `app.py` (sidebar) and by each page script — with
`st.navigation` the main script's own elements are only rendered on the
default page, so the CSS has to be injected from the page scripts as well.

The palette is `#2E86AB` (primary) and `#E4572E` (defaults), with the risk
bands using green `#22C55D`, amber `#F59E0B` and red `#EF4444`.

### Icons

The KPI badges and the chart-switcher buttons carry icons. They are inline
SVG (Material Design paths) that are handed to the browser as CSS masks, so a
glyph always picks up the colour of the element it sits in — the tab icons
turn white with the active tab on their own.

| Where | Icon |
|---|---|
| Total loans | stacked layers |
| Defaulted loans | warning triangle |
| Overall default rate | ring |
| Average loan amount | wallet |
| Overview tab | dashboard grid |
| Credit Risk tab | credit card |
| Purpose & Geography tab | map pin |

### Motion

* The four KPI cards fade in one after the other (0.07 s apart).
* The chart switcher is a segmented control: a gradient pill slides to the
  selected tab, the icons shift from grey to white and the label lifts on
  hover.
* Switching tabs replays the entrance animation of the panel and of its chart
  cards, so every switch feels like the charts are being re-drawn. The
  animation name changes with the active tab, which is what makes the motion
  replay without any JavaScript.
* Chart cards and KPI cards lift slightly on hover.




# FinSight Bank — Loan Default Risk Analysis

## Project Overview
End-to-end data analysis project on FinSight Bank dataset containing 2 million loan records across 9 CSV files. The project covers data loading, cleaning, exploratory data analysis, feature engineering, and regression modelling to predict Loss Given Default (LGD).

---

## Dataset
- 9 CSV files linked by `loan_id`
- 2,000,000 rows, 182 columns after merging
- Key tables: loans_master, customer_bureau, payment_history, loan_performance, credit_card_behavior, collateral_assets, branch_region_economy, monthly_emi_track, loan_enquiry_bureau

---

## Tools & Libraries
- Python, Pandas, NumPy
- Matplotlib
- Scikit-learn
- Statsmodels
- PyArrow (Parquet)
- SciPy

---

## Project Structure

```
FinSight-Bank-Analysis/
│
├── data/                          # CSV files
├── notebooks/
│   ├── Q1_data_loading.ipynb      # Loading, cleaning, merging
│   ├── Q2_eda.ipynb               # Exploratory data analysis
│   ├── Q3_feature_engineering.ipynb
│   └── Q4_regression.ipynb        # Modelling
├── outputs/
│   └── finsight_final_master.parquet
└── README.md
```

---

## Question 1 — Data Loading & Cleaning

### (a) Data Loading
- Loaded all 9 CSV files using `pd.read_csv()`
- Downcasted numeric columns to reduce memory usage
- Saved merged dataset as Parquet file using PyArrow with Snappy compression
- Memory reduced significantly after downcasting
- Final Parquet file size reported

### (b) Merging
- Base table: `loans_master` (2,000,000 rows)
- Sequential left merges on `loan_id`
- Row count verified after every merge using `assert`
- **0 orphan records** found across all 8 joins
- Data integrity confirmed — all loan IDs consistent across files

### (c) Data Quality Issues — 8 Dirty Issues Found

| # | Column | Dirty Count | Issue | Fix |
|---|--------|-------------|-------|-----|
| 1 | `chargeoff_within_12_mths` | 77,556 | Max value 22 lakh impossible | Median |
| 2 | `annual_inc_inr` | 40,066 | Income missing, mandatory field | Median |
| 3 | `cibil_score_band` | 10 | Credit band missing | Mode |
| 4 | `rate_spread_pct` | 23,286 | Negative spread impossible | Replace with 0 |
| 5 | `dpd_bucket` | 15 | Format mismatch `90+ DPD / NPA` | Standardize label |
| 6 | `emi_coverage_ratio` | few k | Ratio of 2049 impossible | Median |
| 7 | `emp_length_years` | 1,80,539 | Employment years missing | Median |
| 8 | `revol_util_pct` | 1,40,097 | Revolving utilization missing | Median |

- Total dirty rows flagged = **4,61,569**
- Binary `dirty_flag` column created

### (d) Missing Value Classification

| Column | Type | Reason |
|--------|------|--------|
| `mths_since_last_delinq` | MNAR | Only defaulters have this value |
| `mort_acc` | MAR | Depends on loan type |
| `emp_length_years` | MAR | Depends on employment type |
| `il_util_pct` | MAR | Only exists if installment loan |

- All 4 columns imputed with median
- Verified with `.isnull().sum()` before and after

### (e) Winsorisation
- Applied at 1st and 99th percentile to 6 most skewed columns
- Columns: `collections_12mths_fee`, `collection_recovery_fee`, `recoveries_inr`, `emi_advance_paid_inr`, `expected_loss_inr`
- Mean, std, max all reduced significantly after winsorisation
- Note: `npa_flag` is binary — winsorisation should be excluded for binary columns

---

## Question 2 — Exploratory Data Analysis

### (a) Loan Status Distribution
- Performing loans: 19,22,444 (96.12%)
- Defaulted loans: 77,556 (3.88%)
- Severe class imbalance 25:1
- Fix: SMOTE oversampling or class weight adjustment

### (b) CIBIL Score — Default vs Performing
- KDE curves plotted for both groups
- Defaulter mean CIBIL lower than performing loans
- Some overlap exists — CIBIL alone not sufficient to predict default

### (c) Histogram Grid
- 12 numeric features plotted
- Right skewed columns identified (skew > 2.0)
- Log transformation applied to reduce skewness

### (d) Correlation Heatmap
- Pearson correlation matrix for top 20 features
- High correlation pairs (|r| > 0.75) identified
- High correlation = multicollinearity = problem for OLS regression

### (e) Boxplots
- 6 features compared between defaulters and performing loans
- `dti_pct` shows clearest separation (difference = 4.44)
- `int_rate_pct` and `cibil_score` also useful
- `annual_inc`, `revol_util`, `emp_length` show no difference

### (f) Default Rate by Loan Grade
- Grade A = lowest default, Grade G = highest
- Monotonically ordered confirmed
- Largest jump identified between grades

### (g) Default Rate by Loan Purpose
- Highest risk purposes identified
- Lowest risk purposes identified
- Ratio between highest and lowest calculated

### (h) Top 10 States by Default Rate
- States exceeding bank average by 5% flagged
- Regional risk patterns identified

### (i) Annual Default Rate 2010-2024
- COVID 2020 spike clearly visible
- Default rate increase from 2019 to 2020 quantified
- GDP growth rate explains the spike

### (j) RBI Repo Rate vs Default Rate
- Dual axis line chart plotted
- Inverse relationship observed
- Lag between rate change and default effect estimated

### (k) LGD Distribution
- Plotted for defaulted loans only
- Distribution shape described
- Log transformation need assessed

### (l) CIBIL Score vs LGD Scatter
- Negative correlation observed
- Lower CIBIL = higher loss when default occurs

---

## Question 3 — Feature Engineering

### 12 New Features Created:

| Feature | Formula |
|---------|---------|
| `emi_to_income_ratio` | installment ÷ (income ÷ 12) |
| `loan_to_income_ratio` | loan amount ÷ income |
| `rate_spread_pct` | int rate − repo rate |
| `real_interest_rate` | int rate − inflation |
| `credit_util_composite` | 0.5×revol + 0.3×bc + 0.2×all util |
| `delinq_severity_score` | delinq × (1 + 1/months since) |
| `enq_velocity_score` | enquiries 30d×4 + enquiries 90d |
| `income_stability_ratio` | income ÷ (emp years + 1) |
| `credit_depth_score` | total acc ÷ (credit hist + 1) |
| `collateral_coverage_ratio` | collateral value ÷ loan amount |
| `log_annual_inc` | log(1 + income) |
| `log_loan_amnt` | log(1 + loan amount) |

- COVID flag analysis with t-test performed
- Skewness before and after log transformation reported

---

## Question 4 — Regression Modelling

### (a) VIF Analysis
- Features with VIF > 10 removed
- Multicollinearity resolved before modelling

### (b) OLS Baseline Model
- Fitted using statsmodels
- R², Adjusted R², F-statistic, Durbin-Watson reported
- Top 5 coefficients interpreted in business terms

### (c) Regularization Models

| Model | Alpha | RMSE | R² |
|-------|-------|------|----|
| Ridge | best | reported | reported |
| Lasso | best | reported | reported |
| ElasticNet | best | reported | reported |

- Lasso zero-coefficient features identified
- 5-fold cross validation used

### (d) Diagnostic Plots
- Residuals vs Fitted
- QQ Plot
- Scale Location
- Cook's Distance

---

## Key Findings
1. Default rate = **3.88%** — severe class imbalance
2. `dti_pct` is strongest separator between defaulters and performing loans
3. Higher interest rate strongly associated with default
4. CIBIL score alone insufficient to predict default
5. COVID 2020 caused significant spike in default rate
6. Collateral coverage reduces Loss Given Default significantly


**Shubham Kathar**
Data Engineering Student
Mumbai, Maharashtra

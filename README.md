# IEEE-CIS Fraud Detection Pipeline

End-to-end fraud detection system built on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) dataset. Covers data engineering, EDA, feature engineering, supervised modelling (Logistic Regression, XGBoost), deep-learning anomaly detection (Autoencoder), and business-cost threshold optimisation.

---

## Project Structure

```
fraud-project/
├── src/                          # Numbered, runnable pipeline stages
│   ├── 01_merge_data.py          # Download & merge transaction + identity tables
│   ├── 02_eda_analysis.py        # Exploratory data analysis (12 plots + report)
│   ├── 03_feature_engineering.py # UID, D-column normalisation, freq encoding, null flags
│   ├── 04_cleanup_split_rank.py  # Drop redundant cols, time-based split, MI ranking
│   ├── 05_train_models.py        # Logistic Regression + XGBoost (top-30 features)
│   ├── 06_train_autoencoder.py   # PyTorch undercomplete autoencoder (non-fraud only)
│   ├── 07_final_evaluation.py    # Test-set evaluation, 3-model comparison, cost analysis
│   └── 08_extended_cost_sweep.py # Fine-grained threshold sweep (0.02 → 0.70)
│
├── notebooks/
│   └── Fraud_Detection_Pipeline.ipynb   # Original exploratory notebook
│
├── data/
│   ├── raw/                      # Source CSVs from Kaggle (gitignored)
│   │   ├── train_transaction.csv
│   │   ├── train_identity.csv
│   │   └── ...
│   └── processed/                # Engineered & split datasets
│       ├── engineered_train.parquet   # 144,233 × 437 cols after FE
│       ├── split_train.parquet        # 114,174 rows (days 0–127)
│       ├── split_val.parquet          # 13,225 rows (days 127–155)
│       ├── split_test.parquet         # 16,834 rows (days 155–182)
│       ├── top30_features.json        # Selected feature list
│       ├── mi_feature_ranking.csv     # All 433 features ranked by MI
│       ├── final_results.json         # Val + test metrics for all 3 models
│       └── extended_cost_sweep.csv    # Cost table across 19 thresholds
│
├── models/
│   ├── xgboost_model.json        # XGBoost booster (native format)
│   ├── xgboost_meta.pkl          # Feature list, medians, val metrics, importances
│   ├── logistic_regression.pkl   # Pipeline (StandardScaler + LR) + val metrics
│   ├── autoencoder_state.pt      # PyTorch state dict + thresholds (p95/p99)
│   └── ae_scaler_and_medians.pkl # Scaler + medians (fit on non-fraud train only)
│
├── reports/
│   ├── eda_report.md                         # Full EDA findings
│   ├── fraud_detection_interview_cheatsheet.md
│   ├── final_models_comparison.csv
│   └── figures/                              # All generated plots
│       ├── ae_reconstruction_error.png
│       ├── final_cost_analysis.png
│       ├── final_model_comparison.png
│       ├── extended_cost_sweep.png
│       └── ...
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download data via Kaggle API (requires ~/.kaggle/kaggle.json)
#    Accept competition rules first at:
#    https://www.kaggle.com/competitions/ieee-fraud-detection/rules
kaggle competitions download -c ieee-fraud-detection -p data/raw/
unzip data/raw/ieee-fraud-detection.zip -d data/raw/
```

---

## Running the Pipeline

Run stages in order from the project root:

```bash
OMP_NUM_THREADS=1 python src/01_merge_data.py          # ~30s
OMP_NUM_THREADS=1 python src/02_eda_analysis.py        # ~60s  → reports/figures/
OMP_NUM_THREADS=1 python src/03_feature_engineering.py # ~15s  → data/processed/
OMP_NUM_THREADS=1 python src/04_cleanup_split_rank.py  # ~45s  → data/processed/splits
OMP_NUM_THREADS=1 python src/05_train_models.py        # ~10s  → models/
OMP_NUM_THREADS=1 python src/06_train_autoencoder.py   # ~90s  → models/
OMP_NUM_THREADS=1 python src/07_final_evaluation.py    # ~10s  → data/processed/final_results.json
OMP_NUM_THREADS=1 python src/08_extended_cost_sweep.py # ~5s   → reports/figures/
```

> `OMP_NUM_THREADS=1` prevents a PyTorch/OpenMP fork hang on macOS Apple Silicon.

---

## Pipeline Summary

### Data
- **Source:** IEEE-CIS Fraud Detection (Kaggle) — transaction + identity tables
- **Join strategy:** Left join on `TransactionID` → **144,233 rows × 434 cols** (identity-joined view)
- **Fraud rate:** 8.69% overall; rises from ~4% to ~12% over the 182-day window

### Feature Engineering (`src/03`)
| Step | What | Result |
|------|------|--------|
| UID construction | `card1 + card2 + addr1 + round(D1)` | 27,848 unique entities |
| D-column normalisation | `D_n − TransactionDT/86400` | Removes time-drift from 14 delta cols |
| Frequency encoding | 10 high-cardinality columns | Proportion-based continuous encoding |
| Missingness flags | 21 columns where null% differs >5pp between fraud/legit | Binary is_null features |
| Drop 100%-null cols | 21 columns (M1–M9 except M4, V1–V11, D11, dist1) | |
| **Final shape** | | **144,233 × 437** |

### Feature Selection (`src/04`)
- Method: **Mutual Information** on train split only (no leakage)
- Evaluated 433 candidate features; top 30 selected
- Top feature: `id_02` (MI=0.236) — fully undocumented Vesta identity column
- Strong signals: C-columns (count features), D_norm-columns, V258/V257 block

### Train / Val / Test Split
Forward-chaining on `TransactionDT` (no random splits):

| Split | Rows | Fraud Rate | Days |
|-------|-----:|:---------:|:----:|
| Train | 114,174 | 7.46% | 0–127 |
| Val | 13,225 | 9.67% | 127–155 |
| Test | 16,834 | 9.06% | 155–182 |

### Models

#### Logistic Regression (baseline)
- `class_weight='balanced'`, L2 reg C=0.1, StandardScaler
- Val AUC-ROC: **0.771** | Test AUC-ROC: **0.793**

#### XGBoost (primary model)
- `scale_pos_weight=12.41`, early stopping on val PR-AUC, stopped at iteration 144
- Val AUC-ROC: **0.869** | Test AUC-ROC: **0.839**
- Val PR-AUC: **0.640** | Test PR-AUC: **0.581**

#### Autoencoder (anomaly detection)
- Architecture: `30→64→32→[16]→32→64→30`, BatchNorm+ReLU, MSE loss
- Trained **only on non-fraud rows** (105,660 rows) — never sees fraud
- Threshold: p99 of non-fraud val reconstruction error = **0.2839**
- Val PR-AUC: **0.395** | Test PR-AUC: **0.265**
- Highest precision at p99 threshold: **0.308** (flags confidently, rarely)

### Final Model Comparison (Test Set)

| Metric | Log. Reg. | XGBoost | Autoencoder |
|--------|:---------:|:-------:|:-----------:|
| AUC-ROC | 0.793 | **0.839** | 0.712 |
| PR-AUC | 0.538 | **0.581** | 0.265 |
| F1 | 0.297 | **0.407** | 0.238 |
| Precision | 0.186 | 0.288 | **0.308** |
| Recall | **0.737** | 0.695 | 0.194 |

### Business-Cost Threshold Optimisation (XGBoost)

Cost model: FN = full `TransactionAmt`; FP = $4.00 (review + churn).
Extended sweep from threshold 0.02 to 0.70 (19 points).

| | Value |
|---|---|
| **Mathematically optimal threshold** | **0.26** |
| Total cost at 0.26 | $36,014 |
| Total cost at 0.50 (default) | $46,766 |
| Savings | **$10,752 (23%)** |
| Fraud caught | 83.1% |
| Alert volume at 0.26 | **40.7%** of transactions ⚠ |
| **Operationally recommended** | **0.30–0.35** (34–40% alert rate, auto-triage required) |

The cost curve is a genuine U-shape with an interior minimum at 0.26 — not an edge artifact. However, flagging 40% of all transactions requires automated pre-screening before human review.

> **Important:** Dollar figures use illustrative assumptions. A production deployment must substitute actual chargeback costs, analyst review times, and empirically measured churn rates.

---

## Requirements

See [`requirements.txt`](requirements.txt) for the pinned dependency list. Key packages:

```
pandas >= 2.0        scikit-learn >= 1.3
xgboost >= 2.0       torch >= 2.13 (CPU)
pyarrow >= 12        seaborn >= 0.13
kaggle               matplotlib
```

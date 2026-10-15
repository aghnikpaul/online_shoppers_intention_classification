# Online Shoppers Purchasing Intention: Classification with a Stacking Ensemble

Predicting whether an e-commerce browsing session ends in a purchase, using three classifiers and a stacking ensemble, in a single reproducible Python script.

![Conversion rate by month](results/02_conversion_by_month.png)

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Repository Structure](#repository-structure)
4. [Pipeline Walkthrough](#pipeline-walkthrough)
5. [Key Findings](#key-findings)
6. [Model Results](#model-results)
7. [Why a Stacking Ensemble?](#why-a-stacking-ensemble)
8. [Limitations and Next Steps](#limitations-and-next-steps)
9. [How to Run](#how-to-run)
10. [Author](#author)

---

## Project Overview

**Business problem:** only about 1 in 6 website sessions ends in a purchase. If a shop can spot likely buyers (or likely non-buyers) while they are still browsing, it can target discounts, chat prompts or retargeting where they matter.

**Task:** binary classification of the `Revenue` column (purchase / no purchase).

**What this project does:**
- cleans and explores the data to produce business findings
- trains Logistic Regression, Random Forest and XGBoost, with imbalance handling
- compares them with precision, recall, F1, ROC-AUC and confusion matrices
- combines them in a stacking ensemble and reports honestly whether it helps

Everything runs from one file, `shopper_intent_master.py`, and every figure is made with Seaborn.

## Dataset

[UCI Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset): one row per browsing session, each from a different user over one year (12,330 rows, 17 features plus the target). The data is not included in this repository; download `online_shoppers_intention.csv` from the link above.

| Group | Columns | Notes |
|---|---|---|
| Page-visit behavior | `Administrative`, `Informational`, `ProductRelated` and their `_Duration` columns | Page counts and seconds spent per page type |
| Google Analytics metrics | `BounceRates`, `ExitRates`, `PageValues` | `PageValues` is the strongest signal of purchase |
| Calendar | `SpecialDay`, `Month`, `Weekend` | Only 10 months appear in the data |
| Visitor and technical | `VisitorType`, `OperatingSystems`, `Browser`, `Region`, `TrafficType` | The last four are anonymized integer codes |
| **Target** | `Revenue` | True if the session ended in a purchase |

After removing 125 duplicate rows, **12,205 sessions** remain and **15.6%** of them are purchases, so the classes are imbalanced.

## Repository Structure

```
.
├── shopper_intent_master.py   # the full pipeline (steps 1 to 8)
├── requirements.txt           # Python dependencies
├── README.md
├── data/                      # put online_shoppers_intention.csv here (not committed)
└── results/
    ├── report.md              # auto-generated report with all findings
    ├── model_comparison.csv   # metrics table for all four models
    └── *.png                  # 12 Seaborn figures
```

## Pipeline Walkthrough

| Step | What happens |
|---|---|
| **1. Load and inspect** | Checks shape, data types, missing values (none), duplicates (125) and class balance |
| **2. EDA** | Conversion rate by month and visitor type, a visitor type × month heatmap, distributions of skewed columns, and a correlation heatmap |
| **3. Preprocessing** | Drops duplicates, converts `Weekend` and `Revenue` to 0/1, treats the four integer-coded columns as categories, and applies `log1p` to seven right-skewed columns (e.g. `PageValues` skew drops from 6.4 to 1.8) |
| **4. Train/test split** | Stratified 80/20 split (9,764 train / 2,441 test, seed 42), done **before** any encoder or scaler is fitted, so nothing leaks from the test set |
| **5. Model training** | Logistic Regression, Random Forest and XGBoost, each inside a pipeline that one-hot encodes categories (73 features after encoding) and scales numerics only where needed. Imbalance is handled with class weights |
| **6. Evaluation** | Accuracy, precision, recall, F1 and ROC-AUC on the held-out test set, plus confusion matrices, ROC curves and feature importances |
| **7. Justification** | Each algorithm is matched to a property of the data (see the generated `report.md`) |
| **8. Ensemble** | Stacking classifier with a Logistic Regression meta-learner trained on 5-fold out-of-fold probabilities |

Because accuracy is misleading here (always predicting "no purchase" would score about 84%), the comparison focuses on recall, precision, F1 and ROC-AUC.

## Key Findings

**Business insights**
- **Seasonality is strong.** Conversion peaks in **November (25.5%)** and is lowest in **February (1.7%)**, against an overall 15.6%. Conversion rises through the second half of the year (Jul to Nov) and drops again in December. February is likely a small sample, so treat that single number with caution.
- **New visitors convert more often than returning ones:** New 24.9% (1,693 sessions), Other 19.8% (81 sessions, too few to rely on), Returning 14.1% (10,431 sessions). Returning visitors are the large majority of traffic, so improving their conversion matters most in absolute terms.
- **`PageValues` dominates.** Buyers average 27.3 against 2.0 for non-buyers, and it is the top feature in all three models.
- **Many features are redundant.** `BounceRates` and `ExitRates` correlate at 0.90, and `ProductRelated` with `ProductRelated_Duration` at 0.86. This hurts linear models more than tree models.

**Modeling insights**
- Calendar information carries real signal: `Month_Nov`, `Month_May` and `Month_Mar` rank near the top for XGBoost.
- The three models agree on `PageValues` and `ExitRates` as key drivers, but they disagree on the rest, which is why comparing them is useful.
- The Logistic Regression coefficients for some `Browser` and `OperatingSystems` codes are large, but these categories are rare, so they are probably noise rather than a real effect.

![Feature importance](results/12_feature_importance.png)

![Correlation heatmap](results/07_correlation_heatmap.png)

## Model Results

Evaluated on the 2,441-session test set (382 buyers); the positive class is "purchase".

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8710 | 0.5601 | 0.8168 | 0.6645 | 0.9256 |
| Random Forest | **0.8976** | **0.6755** | 0.6649 | **0.6702** | 0.9287 |
| XGBoost | 0.8710 | 0.5585 | 0.8377 | **0.6702** | **0.9364** |
| Stacking Ensemble | 0.8640 | 0.5422 | **0.8403** | 0.6591 | 0.9354 |

**Reading the table**
- **Random Forest** is the most conservative: it has the best precision (about two in three flagged sessions buy) but misses 128 of 382 buyers.
- **XGBoost** has the best ROC-AUC (0.9364) and finds 320 of 382 buyers, at the cost of 253 false alarms.
- **Logistic Regression** is a solid baseline. It is close to the others on recall and AUC, which shows that the preprocessing (log transform, scaling) does most of the work.
- **No model wins on every metric.** The right choice depends on cost: if missing a buyer is worse than wasting a discount, prefer the high-recall models; if outreach is expensive, prefer Random Forest.

![Confusion matrices](results/09_confusion_matrices.png)

![Model comparison](results/11_model_comparison.png)

## Why a Stacking Ensemble?

I combined the models with **stacking** rather than simple voting or averaging, for these reasons:

1. **The base models make different kinds of errors.** Logistic Regression is linear and favors recall, Random Forest averages many deep trees and favors precision, and XGBoost corrects its mistakes sequentially. Their confusion matrices differ clearly, which is the condition under which an ensemble can help.
2. **A learned combiner beats a fixed rule.** Stacking trains a meta-learner on the base models' out-of-fold probabilities, so it learns how much to trust each model instead of giving them equal weight as voting does. Using out-of-fold predictions keeps the meta-learner from learning from predictions the base models made on their own training data.
3. **The meta-learner is deliberately simple.** A class-weighted Logistic Regression is hard to overfit and easy to interpret.

**What actually happened:** the ensemble did **not** beat the best single models. Its F1 (0.6591) is 0.011 below Random Forest and XGBoost, and its ROC-AUC (0.9354) is 0.001 below XGBoost. It did achieve the highest recall (0.8403, with only 61 buyers missed), but with more false positives (271) than any single model.

The likely reason is that all three models rely on the same few strong signals (`PageValues`, `ExitRates`, month), so their errors overlap and there is little left for the meta-learner to combine. A gap of about 0.001 in AUC is well within the noise of a single split, so the sensible conclusion is "no reliable gain from stacking on this dataset", and XGBoost alone is the simpler choice. Reporting that honestly is part of the analysis.

## Limitations and Next Steps

- **One split is noisy.** Results come from a single 80/20 split; repeated stratified cross-validation would give confidence intervals and make model differences easier to judge.
- **Fixed 0.5 threshold.** Tuning the decision threshold (or optimizing for a business cost) could improve F1 and change the precision/recall balance, especially for the ensemble.
- **No hyperparameter search.** The models use reasonable defaults; a randomized search would likely help XGBoost and Random Forest.
- **Rare categories.** Grouping rare `Browser`, `OperatingSystems` and `TrafficType` codes could make the linear model more stable.
- **Interpretability.** Adding SHAP values would give per-session explanations.

## How to Run

```bash
# 1. Clone and install
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt

# 2. Download online_shoppers_intention.csv from UCI into data/

# 3. Run the pipeline
python shopper_intent_master.py --data data/online_shoppers_intention.csv --out results
```

**Options:** `--data` (CSV path), `--out` (output folder), `--test-size` (default 0.2), `--seed` (default 42).

**Running in Google Colab:** upload the CSV and script, then run `!python shopper_intent_master.py --data /content/online_shoppers_intention.csv --out results`.

**Outputs:** the 12 PNG figures, `model_comparison.csv` and `report.md` are written to the output folder. If `xgboost` is not installed, the script falls back to scikit-learn's `HistGradientBoostingClassifier`.

**`requirements.txt`**
```
pandas
numpy
scikit-learn
seaborn
matplotlib
xgboost
```

## Author

Freelance data analyst focused on data analysis, machine learning and business analysis.

- Portfolio: https://aghnikpaul.page.gd
- LinkedIn: https://www.linkedin.com/in/aghnikpaul
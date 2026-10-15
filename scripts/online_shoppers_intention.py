"""
Online Shoppers Purchasing Intention - classification master script
====================================================================
Dataset : UCI "Online Shoppers Purchasing Intention" (online_shoppers_intention.csv)
Target  : Revenue (did the browsing session end in a purchase?)

Pipeline
    1. Load and inspect
    2. EDA (business findings)
    3. Preprocessing (dedupe, 0/1 flags, log transform; encoders/scalers live in pipelines)
    4. Stratified 80/20 train/test split (BEFORE fitting any encoder/scaler)
    5. Train Logistic Regression, Random Forest, Gradient Boosting (XGBoost)
    6. Evaluate and compare (precision, recall, F1, ROC-AUC, confusion matrices)
    7. Justify the algorithm choices using dataset facts
    8. Stacking ensemble and honest comparison

Usage
    pip install pandas numpy scikit-learn seaborn matplotlib xgboost
    python shopper_intent_master.py --data online_shoppers_intention.csv --out results

All figures are made with Seaborn
model_comparison.csv is saved in output folder
and report.md (findings, justification, results) in report folder.
"""

import argparse
import os
import time
import warnings

import matplotlib

matplotlib.use("Agg")  # save figures without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")

try:
    from xgboost import XGBClassifier

    HAVE_XGB = True
except ImportError:  # falls back to scikit-learn's histogram gradient boosting
    HAVE_XGB = False

###########################################################################################
# Column definitions

TARGET = "Revenue"
# integer coded columns are labels, not quantities, treat them as categorical
CAT_COLS = ["Month", "VisitorType", "OperatingSystems", "Browser", "Region", "TrafficType"]
# Right-skewed, zero-heavy columns -> log1p
SKEWED_COLS = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "PageValues",
]
MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

REPORT = []  #for report.md

FIGURES = [
    ("Class balance", "01_class_balance.png"),
    ("Conversion rate by month", "02_conversion_by_month.png"),
    ("Conversion rate by visitor type", "03_conversion_by_visitor_type.png"),
    ("Visitor type x month", "04_conversion_month_visitor_heatmap.png"),
    ("PageValues by outcome", "05_pagevalues_by_outcome.png"),
    ("Raw skewed distributions", "06_skewed_distributions_raw.png"),
    ("Correlation heatmap", "07_correlation_heatmap.png"),
    ("Distributions after log transform", "08_skewed_distributions_log.png"),
    ("Confusion matrices", "09_confusion_matrices.png"),
    ("ROC curves", "10_roc_curves.png"),
    ("Model comparison", "11_model_comparison.png"),
    ("Feature importance", "12_feature_importance.png"),
]


def df_to_markdown(frame, floatfmt=".4f"):
    """Dependency-free Markdown table (works without `tabulate`)."""
    cols = [frame.index.name or ""] + list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for idx, row in frame.iterrows():
        cells = [str(idx)] + [f"{v:{floatfmt}}" if isinstance(v, (int, float, np.floating)) else str(v)
                              for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def log(text=""):
    """Print and also store the line for report.md."""
    print(text)
    REPORT.append(text)


def save_fig(fig, out_dir, name):
    fig.tight_layout()
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   [saved] {path}")


def to_binary(series):
    """Converting True/False (bool or text) to 1/0."""
    return series.astype(str).str.strip().str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0}).astype(int)


def plot_distributions(frame, cols, title, out_dir, fname):
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    for ax, col in zip(axes.ravel(), cols):
        sns.histplot(frame[col], bins=40, ax=ax, color="#3b7dd8")
        ax.set_title(f"{col}\nskew = {frame[col].skew():.2f}", fontsize=10)
        ax.set_xlabel("")
    for ax in axes.ravel()[len(cols):]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=14, y=1.01)
    save_fig(fig, out_dir, fname)


################################################################################
# model builders (fresh, unfitted objects each call)

def make_prep(scale, num_cols):
    """One-hot encode categoricals; optionally scale numerics. Fitted on TRAIN only."""
    numeric = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
            ("num", numeric, num_cols),
        ],
        verbose_feature_names_out=False,
    )


def build_models(num_cols, pos_weight, seed):
    lr = Pipeline([
        ("prep", make_prep(True, num_cols)),  # scaling matters for LR
        ("clf", LogisticRegression(class_weight="balanced", max_iter=3000, random_state=seed)),
    ])
    rf = Pipeline([
        ("prep", make_prep(False, num_cols)),  # trees do not need scaling
        ("clf", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed)),
    ])
    if HAVE_XGB:
        boost_name = "XGBoost"
        boost_clf = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=pos_weight, eval_metric="logloss",
            n_jobs=-1, random_state=seed)
    else:
        boost_name = "HistGradientBoosting"
        boost_clf = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, max_depth=4, class_weight="balanced",
            random_state=seed)
    boost = Pipeline([("prep", make_prep(False, num_cols)), ("clf", boost_clf)])
    return {"Logistic Regression": lr, "Random Forest": rf, boost_name: boost}


def evaluate(name, model, X_te, y_te):
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_te, pred),
        "Precision": precision_score(y_te, pred, zero_division=0),
        "Recall": recall_score(y_te, pred),
        "F1": f1_score(y_te, pred),
        "ROC-AUC": roc_auc_score(y_te, proba),
    }
    return metrics, proba, pred


def main():
    parser = argparse.ArgumentParser(description="Online Shoppers classification pipeline")
    parser.add_argument("--data", default="online_shoppers_intention.csv", help="path to the CSV")
    parser.add_argument("--out", default="results", help="folder for figures and reports")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    # parse_known_args ignores extra flags that Colab/Jupyter inject (e.g. "-f kernel.json")
    args, _unknown = parser.parse_known_args()
    os.makedirs(args.out, exist_ok=True)
    np.random.seed(args.seed)

#################################################################################
# 1. LOAD AND INSPECT
    
    section("1. LOAD AND INSPECT")
    df_raw = pd.read_csv(args.data)
    print(f"Shape            : {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns")
    print("\nData types:")
    print(df_raw.dtypes.to_string())
    nulls = df_raw.isnull().sum()
    print(f"\nTotal missing values: {int(nulls.sum())}")
    if nulls.sum() > 0:
        print(nulls[nulls > 0].to_string())
    n_dupes = int(df_raw.duplicated().sum())
    print(f"Duplicate rows   : {n_dupes}")

    missing_cols = [c for c in CAT_COLS + SKEWED_COLS + [TARGET, "Weekend"] if c not in df_raw.columns]
    if missing_cols:
        raise SystemExit(f"Expected columns not found in CSV: {missing_cols}")

    balance = to_binary(df_raw[TARGET]).value_counts().sort_index()
    print("\nClass balance of Revenue (raw data):")
    print(f"  No purchase (0): {balance[0]:,} ({balance[0] / balance.sum():.1%})")
    print(f"  Purchase    (1): {balance[1]:,} ({balance[1] / balance.sum():.1%})")

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x=to_binary(df_raw[TARGET]).map({0: "No purchase", 1: "Purchase"}), ax=ax,
                  order=["No purchase", "Purchase"], palette=["#9aa5b1", "#2e8b57"])
    ax.set_title("Class balance of Revenue")
    ax.set_xlabel("")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=10)
    save_fig(fig, args.out, "01_class_balance.png")

###########################################################################################################

# 2. EDA (on de-duplicated data, with Revenue as 0/1)
    section("2. EXPLORATORY DATA ANALYSIS")
    eda = df_raw.drop_duplicates().reset_index(drop=True).copy()
    eda[TARGET] = to_binary(eda[TARGET])
    overall_rate = eda[TARGET].mean()
    log("## Business findings")
    log(f"- Overall conversion rate: {overall_rate:.1%} of {len(eda):,} unique sessions.")

# conversion by month
    months = [m for m in MONTH_ORDER if m in eda["Month"].unique()]
    month_tbl = eda.groupby("Month")[TARGET].agg(rate="mean", sessions="count").reindex(months)
    print("\nConversion rate by month:")
    print(month_tbl.assign(rate=month_tbl["rate"].map("{:.1%}".format)).to_string())
    best_m, worst_m = month_tbl["rate"].idxmax(), month_tbl["rate"].idxmin()
    log(f"- Best month: {best_m} ({month_tbl.loc[best_m, 'rate']:.1%}); "
        f"weakest month: {worst_m} ({month_tbl.loc[worst_m, 'rate']:.1%}). "
        f"Note that {worst_m if month_tbl.loc[worst_m, 'sessions'] < 500 else 'some months'} "
        f"may have few sessions, so check the counts above.")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=eda, x="Month", y=TARGET, order=months, errorbar=None, color="#3b7dd8", ax=ax)
    ax.axhline(overall_rate, color="red", ls="--", lw=1, label=f"Overall {overall_rate:.1%}")
    ax.set_ylabel("Conversion rate")
    ax.set_title("Conversion rate by month")
    ax.legend()
    save_fig(fig, args.out, "02_conversion_by_month.png")

    
# conversion by visitor type
    vt_tbl = eda.groupby("VisitorType")[TARGET].agg(rate="mean", sessions="count")
    print("\nConversion rate by visitor type:")
    print(vt_tbl.assign(rate=vt_tbl["rate"].map("{:.1%}".format)).to_string())
    for vt, row in vt_tbl.iterrows():
        log(f"- {vt}: {row['rate']:.1%} conversion across {int(row['sessions']):,} sessions.")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.barplot(data=eda, x="VisitorType", y=TARGET, errorbar=None, palette="viridis", ax=ax)
    ax.set_ylabel("Conversion rate")
    ax.set_title("Conversion rate by visitor type")
    save_fig(fig, args.out, "03_conversion_by_visitor_type.png")

    
# month x visitor type heatmap
    pivot = eda.pivot_table(index="VisitorType", columns="Month", values=TARGET, aggfunc="mean")[months]
    fig, ax = plt.subplots(figsize=(11, 3.5))
    sns.heatmap(pivot, annot=True, fmt=".1%", cmap="YlGnBu", cbar_kws={"label": "Conversion"}, ax=ax)
    ax.set_title("Conversion rate: visitor type x month")
    save_fig(fig, args.out, "04_conversion_month_visitor_heatmap.png")

    
# pagevals vs purchase
    pv = eda.groupby(TARGET)["PageValues"].mean()
    log(f"- Mean PageValues: {pv[1]:.1f} for buyers vs {pv[0]:.1f} for non-buyers; "
        f"it is the strongest behavioural signal of purchase intent.")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.boxplot(x=eda[TARGET].map({0: "No purchase", 1: "Purchase"}), y=np.log1p(eda["PageValues"]),
                palette=["#9aa5b1", "#2e8b57"], ax=ax)
    ax.set_ylabel("log(1 + PageValues)")
    ax.set_xlabel("")
    ax.set_title("PageValues by outcome")
    save_fig(fig, args.out, "05_pagevalues_by_outcome.png")

# Skewed distributions (before log transform)
    skew_before = eda[SKEWED_COLS].skew()
    print("\nSkewness of heavy-tailed columns (raw):")
    print(skew_before.round(2).to_string())
    plot_distributions(eda, SKEWED_COLS, "Raw distributions of skewed columns", args.out,
                       "06_skewed_distributions_raw.png")

# heatmap
    corr_cols = SKEWED_COLS + ["BounceRates", "ExitRates", "SpecialDay", TARGET]
    corr = eda[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool)), annot=True, fmt=".2f",
                cmap="coolwarm", center=0, annot_kws={"size": 8}, ax=ax)
    ax.set_title("Correlation heatmap (numeric features)")
    save_fig(fig, args.out, "07_correlation_heatmap.png")
    corr_be = eda["BounceRates"].corr(eda["ExitRates"])
    corr_pr = eda["ProductRelated"].corr(eda["ProductRelated_Duration"])
    log(f"- BounceRates and ExitRates correlate at {corr_be:.2f}; ProductRelated and "
        f"ProductRelated_Duration at {corr_pr:.2f} (redundant, which matters for linear models).")

########################################################################################################################
# 3. PREPROCESSING (stateless steps only, forfitted steps go into pipelines)

    section("3. PREPROCESSING")
    df = df_raw.drop_duplicates().reset_index(drop=True).copy()
    print(f"Dropped {n_dupes} duplicate rows -> {len(df):,} rows remain")
    df["Weekend"] = to_binary(df["Weekend"])
    df[TARGET] = to_binary(df[TARGET])
    for col in CAT_COLS:
        df[col] = df[col].astype(str)  # integer codes become categories
    df[SKEWED_COLS] = np.log1p(df[SKEWED_COLS])  # log1p has no fitted parameters -> no leakage
    skew_after = df[SKEWED_COLS].skew()
    print("\nSkewness before -> after log1p:")
    print(pd.DataFrame({"before": skew_before, "after": skew_after}).round(2).to_string())
    plot_distributions(df, SKEWED_COLS, "Distributions after log1p transform", args.out,
                       "08_skewed_distributions_log.png")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    num_cols = [c for c in X.columns if c not in CAT_COLS]
    print(f"\nCategorical columns to one-hot encode: {CAT_COLS}")
    print(f"Numeric columns ({len(num_cols)}): {num_cols}")

# 4. STRATIFIED train test split (before fitting encoders/scalers)

    section("4. STRATIFIED 80/20 TRAIN/TEST SPLIT")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed)
    print(f"Train: {X_tr.shape[0]:,} rows | buyer share {y_tr.mean():.2%}")
    print(f"Test : {X_te.shape[0]:,} rows | buyer share {y_te.mean():.2%}")
    pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
    print(f"Imbalance ratio in train (negatives per positive): {pos_weight:.2f}")
    print("Encoders and scalers are fitted inside pipelines on the training set only.")

########################################################################################################################
# 5. TRAIN MODELS

    section("5. TRAIN BASE MODELS (class weighting handles imbalance)")
    models = build_models(num_cols, pos_weight, args.seed)
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_tr, y_tr)
        print(f"  {name:<22} trained in {time.time() - t0:5.1f}s")

################################################################################    
# 6. EVALUATE AND COMPARE

    section("6. EVALUATION")
    rows, probas, preds = [], {}, {}
    for name, model in models.items():
        m, proba, pred = evaluate(name, model, X_te, y_te)
        rows.append(m)
        probas[name], preds[name] = proba, pred

##############################################################################################################
#8 stacking ensemble computed here so figures can include it

    t0 = time.time()
    stack = StackingClassifier(
        estimators=[(n.lower().replace(" ", "_"), build_models(num_cols, pos_weight, args.seed)[n])
                    for n in models],
        final_estimator=LogisticRegression(class_weight="balanced", max_iter=3000,
                                           random_state=args.seed),
        stack_method="predict_proba",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed),
        n_jobs=1,
    )
    stack.fit(X_tr, y_tr)
    print(f"  {'Stacking Ensemble':<22} trained in {time.time() - t0:5.1f}s")
    m, proba, pred = evaluate("Stacking Ensemble", stack, X_te, y_te)
    rows.append(m)
    probas["Stacking Ensemble"], preds["Stacking Ensemble"] = proba, pred

    results = pd.DataFrame(rows).set_index("Model")
    results.to_csv(os.path.join(args.out, "model_comparison.csv"))
    print("\nModel comparison on the held-out test set (positive class = purchase):")
    print(results.round(4).to_string())

# confusion matrix for buy no buy
    n_models = len(preds)
    fig, axes = plt.subplots(1, n_models, figsize=(4.2 * n_models, 4))
    for ax, (name, pred) in zip(np.atleast_1d(axes), preds.items()):
        cm = confusion_matrix(y_te, pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["No buy", "Buy"], yticklabels=["No buy", "Buy"])
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle("Confusion matrices (test set)", y=1.03, fontsize=14)
    save_fig(fig, args.out, "09_confusion_matrices.png")

#ROC curves
    roc_rows = []
    for name, proba in probas.items():
        fpr, tpr, _ = roc_curve(y_te, proba)
        roc_rows.append(pd.DataFrame({"FPR": fpr, "TPR": tpr,
                                      "Model": f"{name} (AUC {results.loc[name, 'ROC-AUC']:.3f})"}))
    roc_df = pd.concat(roc_rows)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.lineplot(data=roc_df, x="FPR", y="TPR", hue="Model", errorbar=None, ax=ax)
    sns.lineplot(x=[0, 1], y=[0, 1], color="grey", ls="--", ax=ax, label="Chance")
    ax.set_title("ROC curves (test set)")
    save_fig(fig, args.out, "10_roc_curves.png")


## metrics comparison bars
    melted = results.reset_index().melt(id_vars="Model", value_vars=["Precision", "Recall", "F1", "ROC-AUC"],
                                        var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=melted, x="Metric", y="Score", hue="Model", palette="Set2", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison (test set)")
    ax.legend(loc="lower right", fontsize=9)
    save_fig(fig, args.out, "11_model_comparison.png")



# feature importance(coefficients)
    boost_name = [n for n in models if n not in ("Logistic Regression", "Random Forest")][0]
    panels = []
    lr_pipe = models["Logistic Regression"]
    names_lr = lr_pipe.named_steps["prep"].get_feature_names_out()
    coef = pd.Series(lr_pipe.named_steps["clf"].coef_[0], index=names_lr)
    panels.append(("Logistic Regression: coefficients", coef.reindex(coef.abs().sort_values(ascending=False).index)[:12]))
    for mname in ["Random Forest", boost_name]:
        pipe = models[mname]
        clf = pipe.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            imp = pd.Series(clf.feature_importances_, index=pipe.named_steps["prep"].get_feature_names_out())
            panels.append((f"{mname}: importance", imp.sort_values(ascending=False)[:12]))
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5.5))
    for ax, (title, series) in zip(np.atleast_1d(axes), panels):
        sns.barplot(x=series.values, y=series.index, palette="mako", ax=ax)
        ax.set_title(title, fontsize=11)
    save_fig(fig, args.out, "12_feature_importance.png")


## 7. justifying the algo #report

    section("7. WHY THESE ALGORITHMS?")
    n_feat = len(names_lr)
    log("\n## Algorithm justification")
    log(f"Dataset facts: {len(df):,} sessions, {n_feat} features after encoding, only "
        f"{y.mean():.1%} buyers, heavy right skew (e.g. PageValues skew {skew_before['PageValues']:.1f} "
        f"-> {skew_after['PageValues']:.1f} after log1p), correlated pairs "
        f"(BounceRates/ExitRates r={corr_be:.2f}), and a mix of numeric and categorical columns.")
    log("")
    log("- **Logistic Regression** (baseline): suits a binary target, gives interpretable coefficients "
        "that a business client can read, and is fast. It needs scaling and log-transformed inputs "
        "and is sensitive to correlated features, so it shows what the preprocessing buys. "
        "`class_weight='balanced'` counters the imbalance.")
    log("- **Random Forest**: handles mixed feature types, skew and non-linear interactions "
        "(for example month x visitor type) without transforms. Bagging reduces variance, it is robust "
        "to correlated and noisy features, and it provides feature importance. "
        "`class_weight='balanced_subsample'` handles imbalance.")
    log(f"- **{boost_name}** (gradient boosting): usually the strongest model on tabular data. It builds "
        "trees sequentially to correct earlier errors, which captures subtle patterns in the minority "
        "class, and it supports imbalance directly "
        f"({'`scale_pos_weight`' if HAVE_XGB else '`class_weight`'}).")
    log("- **Stacking**: the three models have different inductive biases (linear, bagged trees, "
        "boosted trees), so their errors are not identical. A logistic-regression meta-learner "
        "trained on out-of-fold probabilities learns how much to trust each one.")
    log("- **Metrics**: with ~85/15 classes, accuracy alone is misleading (predicting 'no purchase' always "
        "scores ~85%). Recall shows how many buyers are found, precision how many flagged sessions "
        "really buy, F1 balances them, and ROC-AUC measures ranking quality independent of the threshold.")



################################################################################
### 8 ensemble: individual vs stacking

    section("8. STACKING ENSEMBLE vs INDIVIDUAL MODELS")
    base = results.drop(index="Stacking Ensemble")
    best_f1_model = base["F1"].idxmax()
    best_auc_model = base["ROC-AUC"].idxmax()
    d_f1 = results.loc["Stacking Ensemble", "F1"] - base["F1"].max()
    d_auc = results.loc["Stacking Ensemble", "ROC-AUC"] - base["ROC-AUC"].max()
    d_prec = results.loc["Stacking Ensemble", "Precision"] - base.loc[best_f1_model, "Precision"]
    d_rec = results.loc["Stacking Ensemble", "Recall"] - base.loc[best_f1_model, "Recall"]

    log("\n## Ensemble result")
    log(f"- Best single model by F1: {best_f1_model} ({base['F1'].max():.4f}); "
        f"by ROC-AUC: {best_auc_model} ({base['ROC-AUC'].max():.4f}).")
    log(f"- Stacking: F1 {results.loc['Stacking Ensemble', 'F1']:.4f} ({d_f1:+.4f}), "
        f"ROC-AUC {results.loc['Stacking Ensemble', 'ROC-AUC']:.4f} ({d_auc:+.4f}) "
        f"versus the best single model on each metric.")
    log(f"- Versus {best_f1_model}: precision {d_prec:+.4f}, recall {d_rec:+.4f}.")

    if d_auc > 0.002 and d_f1 > 0.005:
        verdict = ("Stacking improved both ranking quality (AUC) and the F1 score, so combining models "
                   "with different biases helped on this data.")
    elif d_auc > 0.002:
        verdict = ("Stacking improved ROC-AUC, meaning it ranks buyers better, but F1 did not move "
                   "much. F1 depends on the 0.5 threshold, so the gain may show up after threshold tuning.")
    elif d_f1 > 0.005:
        verdict = ("Stacking improved F1 mostly by shifting the precision/recall balance, while AUC "
                   "stayed about the same. That is a threshold effect more than better ranking.")
    else:
        verdict = ("Stacking gave no meaningful gain over the best single model. That is a legitimate "
                   "result: the gradient boosting model already captures most of the signal, and the "
                   "base learners' errors overlap heavily.")
    log(f"- **Interpretation:** {verdict}")
    log("- **Why gains are usually small here:** most predictive power sits in a few features "
        "(PageValues, ExitRates, month), which all models exploit, so their errors are highly correlated. "
        "Stacking helps most when base models disagree in useful ways. The balanced meta-learner "
        "also moves the decision threshold, which can trade precision for recall.")
    log("- **Caveat:** one 80/20 split on ~12k rows is a noisy estimate. Differences of a few "
        "thousandths are within noise; repeated cross-validation would be needed to claim a win.")


###########################################################################################################
# Save report (Markdown, renders on GitHub with the figures embedded)

    best_model = results["F1"].idxmax()
    report_path = os.path.join(args.out, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Online Shoppers Purchasing Intention: Classification Report\n\n")
        f.write("Auto-generated by `shopper_intent_master.py`. Dataset: "
                "[UCI Online Shoppers Purchasing Intention]"
                "(https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset).\n\n")

        f.write("## Setup\n\n")
        f.write(f"- Sessions after removing {n_dupes} duplicates: {len(df):,}\n")
        f.write(f"- Train/test split: {len(X_tr):,} / {len(X_te):,} "
                f"(stratified, test size {args.test_size:.0%}, seed {args.seed})\n")
        f.write(f"- Buyer share: {y_tr.mean():.1%} (train), {y_te.mean():.1%} (test)\n")
        f.write(f"- Models: Logistic Regression, Random Forest, {boost_name}, and a stacking ensemble\n")
        f.write("- Imbalance handling: class weights; encoders and scalers fitted on the training set only\n\n")

        f.write("## Results (test set, positive class = purchase)\n\n")
        f.write(df_to_markdown(results.round(4)) + "\n\n")
        f.write(f"Best model by F1: **{best_model}** ({results.loc[best_model, 'F1']:.4f}).\n\n")

        f.write("\n".join(REPORT) + "\n\n")

        f.write("## Figures\n\n")
        for caption, fname in FIGURES:
            f.write(f"### {caption}\n\n![{caption}]({fname})\n\n")

    section("DONE")
    print(f"Figures, model_comparison.csv and report.md saved in: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
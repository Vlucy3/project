"""
Walk-forward time-series cross-validation.

Each fold expands the training window by one month and evaluates the next
16 days — matching the actual test horizon (Aug 16-31, 2017).

Models evaluated: Linear Regression, XGBoost, LightGBM, XGB+LGB Ensemble.
Deep-learning models are excluded (retraining per fold is prohibitively slow).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

from features import (load_data, prep_oil, get_nat_hols,
                      engineer_features, add_lags)

DATA_DIR = Path(__file__).parent

FEATURES = [
    "store_nbr", "family", "onpromotion", "next_promo_7",
    "type", "cluster", "dcoilwtico", "oil_l7",
    "is_nat_holiday", "is_reg_holiday", "is_loc_holiday",
    "dow_sin", "dow_cos", "month", "payday_signal", "store_family_mean",
    "lag_16", "lag_17", "lag_18", "lag_21", "lag_28",
    "roll_3", "roll_7", "roll_14",
]

# Walk-forward folds: (train_start, train_end, val_start, val_end)
# Each val window is 16 days — the same horizon as the Kaggle test set.
FOLDS = [
    ("2017-01-01", "2017-04-30", "2017-05-01", "2017-05-16"),
    ("2017-01-01", "2017-05-31", "2017-06-01", "2017-06-16"),
    ("2017-01-01", "2017-06-30", "2017-07-01", "2017-07-16"),
    ("2017-01-01", "2017-07-31", "2017-08-01", "2017-08-16"),
]

# ── 1. LOAD SHARED DATA ───────────────────────────────────────────────────────
print("Loading data...")
train_full, _, stores, oil, holidays = load_data(DATA_DIR)
train_full = train_full[train_full["date"] >= "2017-01-01"].copy()

oil      = prep_oil(oil)
nat_hols = get_nat_hols(holidays)

# ── 2. METRICS HELPER ─────────────────────────────────────────────────────────
def metrics(y_true, y_pred):
    y_pred = np.maximum(0, y_pred)
    rmsle  = np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))
    rmse   = np.sqrt(mean_squared_error(y_true, y_pred))
    mae    = mean_absolute_error(y_true, y_pred)
    r2     = r2_score(y_true, y_pred)
    return dict(RMSLE=rmsle, RMSE=rmse, MAE=mae, R2=r2)

# ── 3. FEATURE ENGINEERING PER FOLD ──────────────────────────────────────────
def build_fold(tr_start, tr_end, vl_start, vl_end):
    """Return (X_train, X_val, y_val) with all features for this fold."""
    fold_train = train_full[
        (train_full["date"] >= tr_start) & (train_full["date"] <= tr_end)
    ].copy()
    fold_val = train_full[
        (train_full["date"] >= vl_start) & (train_full["date"] <= vl_end)
    ].copy()

    # Target encoding computed only from this fold's training data (no leakage)
    target_mean = (
        fold_train.groupby(["store_nbr", "family"])["sales"]
        .mean().reset_index()
        .rename(columns={"sales": "store_family_mean"})
    )

    tr_df = engineer_features(fold_train, stores, oil, target_mean, nat_hols, holidays)
    vl_df = engineer_features(fold_val,   stores, oil, target_mean, nat_hols, holidays)

    # Lags computed on combined window so val rows get lag values from training
    combined = (
        pd.concat([tr_df, vl_df], axis=0)
        .sort_values(["store_nbr", "family", "date"])
    )
    combined = add_lags(combined, lags=[16, 17, 18, 21, 28], rolls=[3, 7, 14])
    combined["log_sales"] = np.log1p(combined["sales"].fillna(0))
    combined = combined.fillna(0)

    split_ts  = pd.Timestamp(tr_end)
    X_tr      = combined[combined["date"] <= split_ts]
    X_vl      = combined[combined["date"] >  split_ts]
    return X_tr, X_vl, X_vl["sales"].values

# ── 4. WALK-FORWARD EVALUATION ────────────────────────────────────────────────
MODEL_NAMES = ["Linear Regression", "XGBoost", "LightGBM", "XGB+LGB (0.3/0.7)"]
all_results = []   # list of dicts: fold, model, metric -> value

SEP = "=" * 70
print(f"\n{SEP}")
print("  WALK-FORWARD CROSS-VALIDATION  (4 folds × 16-day val window)")
print(SEP)

for fold_idx, (tr_start, tr_end, vl_start, vl_end) in enumerate(FOLDS, 1):
    print(f"\nFold {fold_idx}: train {tr_start} → {tr_end} | val {vl_start} → {vl_end}")
    X_tr, X_vl, y_vl = build_fold(tr_start, tr_end, vl_start, vl_end)
    print(f"  Train rows: {len(X_tr):,}  |  Val rows: {len(X_vl):,}")

    fold_rows = []

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_tr[FEATURES], X_tr["log_sales"])
    lr_preds = np.expm1(lr.predict(X_vl[FEATURES]))
    fold_rows.append({"fold": fold_idx, "model": "Linear Regression",
                      **metrics(y_vl, lr_preds)})

    # XGBoost
    xgb = XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42,
        early_stopping_rounds=30, eval_metric="rmse",
    )
    xgb.fit(X_tr[FEATURES], X_tr["log_sales"],
            eval_set=[(X_vl[FEATURES], X_vl["log_sales"])], verbose=False)
    xgb_preds = np.expm1(xgb.predict(X_vl[FEATURES]))
    fold_rows.append({"fold": fold_idx, "model": "XGBoost",
                      **metrics(y_vl, xgb_preds)})

    # LightGBM
    lgb = LGBMRegressor(
        n_estimators=800, learning_rate=0.05, num_leaves=100,
        feature_fraction=0.8, bagging_fraction=0.8, n_jobs=-1,
        random_state=42, verbose=-1,
    )
    lgb.fit(X_tr[FEATURES], X_tr["log_sales"],
            eval_set=[(X_vl[FEATURES], X_vl["log_sales"])], callbacks=[])
    lgb_preds = np.expm1(lgb.predict(X_vl[FEATURES]))
    fold_rows.append({"fold": fold_idx, "model": "LightGBM",
                      **metrics(y_vl, lgb_preds)})

    # Ensemble
    ens_preds = 0.3 * xgb_preds + 0.7 * lgb_preds
    fold_rows.append({"fold": fold_idx, "model": "XGB+LGB (0.3/0.7)",
                      **metrics(y_vl, ens_preds)})

    for row in fold_rows:
        print(f"  {row['model']:<25} RMSLE={row['RMSLE']:.5f}  "
              f"RMSE={row['RMSE']:7.1f}  MAE={row['MAE']:7.1f}  R²={row['R2']:.4f}")

    all_results.extend(fold_rows)

# ── 5. SUMMARY ACROSS FOLDS ───────────────────────────────────────────────────
results_df = pd.DataFrame(all_results)

print(f"\n{SEP}")
print("  SUMMARY  —  mean ± std across 4 folds")
print(SEP)
print(f"\n  {'Model':<25} {'RMSLE mean':>11}  {'±std':>7}  {'RMSE mean':>10}  {'R² mean':>8}")
print(f"  {'-'*25} {'-'*11}  {'-'*7}  {'-'*10}  {'-'*8}")

summary_rows = []
for model in MODEL_NAMES:
    sub  = results_df[results_df["model"] == model]
    row  = {
        "Model":      model,
        "RMSLE_mean": sub["RMSLE"].mean(),
        "RMSLE_std":  sub["RMSLE"].std(),
        "RMSE_mean":  sub["RMSE"].mean(),
        "R2_mean":    sub["R2"].mean(),
    }
    summary_rows.append(row)
    print(f"  {model:<25} {row['RMSLE_mean']:>11.5f}  "
          f"{row['RMSLE_std']:>7.5f}  {row['RMSE_mean']:>10.1f}  {row['R2_mean']:>8.4f}")

summary_df = pd.DataFrame(summary_rows).set_index("Model")

best = summary_df["RMSLE_mean"].idxmin()
most_stable = summary_df["RMSLE_std"].idxmin()
print(f"\n  Best mean RMSLE : {best}")
print(f"  Most consistent : {most_stable}")

# Save CSVs
results_df.to_csv(DATA_DIR / "cv_fold_results.csv", index=False)
summary_df.to_csv(DATA_DIR / "cv_summary.csv")
print(f"\n  Fold results saved to cv_fold_results.csv")
print(f"  Summary    saved to cv_summary.csv")

# ── 6. CHARTS ─────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Walk-Forward Cross-Validation Results", fontsize=13)

colors = plt.cm.tab10.colors
x      = np.arange(len(FOLDS))
width  = 0.2

# Left: RMSLE per fold per model
for i, model in enumerate(MODEL_NAMES):
    sub = results_df[results_df["model"] == model].sort_values("fold")
    ax1.bar(x + i * width, sub["RMSLE"].values, width, label=model,
            color=colors[i], alpha=0.85)

ax1.set_xticks(x + width * 1.5)
ax1.set_xticklabels([f"Fold {i+1}\n({s} →\n{e})"
                     for i, (_, _, s, e) in enumerate(FOLDS)], fontsize=8)
ax1.set_ylabel("RMSLE")
ax1.set_title("RMSLE by Fold & Model")
ax1.legend(fontsize=8)

# Right: mean RMSLE with std error bars
means  = summary_df["RMSLE_mean"].values
stds   = summary_df["RMSLE_std"].values
bars   = ax2.bar(MODEL_NAMES, means, color=colors[:len(MODEL_NAMES)], alpha=0.85)
ax2.errorbar(MODEL_NAMES, means, yerr=stds, fmt="none",
             color="black", capsize=4, linewidth=1.5)
ax2.set_ylabel("Mean RMSLE  (± 1 std)")
ax2.set_title("Mean RMSLE across 4 Folds")
ax2.tick_params(axis="x", rotation=15)
for bar, mean in zip(bars, means):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
             f"{mean:.4f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(DATA_DIR / "cv_results.png", dpi=150)
plt.close()
print("  Chart saved to cv_results.png")

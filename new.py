"""
Best ensemble forecasting model: XGBoost + LightGBM (0.3 / 0.7 blend).
Trains on all available data, evaluates on the last 16-day validation window,
then generates a Kaggle submission CSV.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

from features import (load_data, prep_oil, get_split_date, make_target_encoding,
                      get_nat_hols, engineer_features, add_lags)

DATA_DIR = Path(__file__).parent

FEATURES = [
    "store_nbr", "family", "onpromotion", "next_promo_7",
    "type", "cluster", "dcoilwtico", "oil_l7",
    "is_nat_holiday", "is_reg_holiday", "is_loc_holiday",
    "dow_sin", "dow_cos", "month", "payday_signal", "store_family_mean",
    "lag_16", "lag_17", "lag_18", "lag_21", "lag_28",
    "roll_3", "roll_7", "roll_14",
]

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print("Loading data...")
train, test, stores, oil, holidays = load_data(DATA_DIR)

# Use all data from 2016 onwards for maximum historical context
train = train[train["date"] >= "2016-01-01"].copy()

# ── 2. SHARED SETUP ───────────────────────────────────────────────────────────
oil         = prep_oil(oil)
split_date  = get_split_date(train)          # last 16 days → validation
target_mean = make_target_encoding(train, split_date)
nat_hols    = get_nat_hols(holidays)

print(f"Train period : {train['date'].min().date()} → {split_date.date()}")
print(f"Val period   : {(split_date + pd.Timedelta(days=1)).date()} → "
      f"{train['date'].max().date()}")

# ── 3. FEATURE ENGINEERING ────────────────────────────────────────────────────
print("Engineering features...")
train_df = engineer_features(train, stores, oil, target_mean, nat_hols, holidays)
test_df  = engineer_features(test,  stores, oil, target_mean, nat_hols, holidays)

# Compute lags on combined train+test so test rows get real lag values
full_df = (
    pd.concat([train_df, test_df], axis=0)
    .sort_values(["store_nbr", "family", "date"])
)
full_df = add_lags(full_df, lags=[16, 17, 18, 21, 28], rolls=[3, 7, 14])

train_final          = full_df[full_df["sales"].notna()].fillna(0).copy()
train_final["log_sales"] = np.log1p(train_final["sales"])
test_final           = full_df[full_df["sales"].isna()].fillna(0).copy()

X_train = train_final[train_final["date"] <= split_date]
X_val   = train_final[train_final["date"] >  split_date]
y_val   = X_val["sales"].values

print(f"Train rows : {len(X_train):,}   Val rows : {len(X_val):,}   "
      f"Test rows : {len(test_final):,}")

# ── 4. TRAIN MODELS ───────────────────────────────────────────────────────────
print("\nTraining XGBoost...")
xgb = XGBRegressor(
    n_estimators=1000, learning_rate=0.03, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    n_jobs=-1, random_state=42,
    early_stopping_rounds=50, eval_metric="rmse",
)
xgb.fit(
    X_train[FEATURES], X_train["log_sales"],
    eval_set=[(X_val[FEATURES], X_val["log_sales"])],
    verbose=False,
)

print("Training LightGBM...")
lgb = LGBMRegressor(
    n_estimators=1200, learning_rate=0.03, num_leaves=150,
    feature_fraction=0.8, bagging_fraction=0.8,
    n_jobs=-1, random_state=42, verbose=-1,
)
lgb.fit(
    X_train[FEATURES], X_train["log_sales"],
    eval_set=[(X_val[FEATURES], X_val["log_sales"])],
    callbacks=[],
)

# ── 5. EVALUATE ENSEMBLE ──────────────────────────────────────────────────────
xgb_val   = np.expm1(xgb.predict(X_val[FEATURES]))
lgb_val   = np.expm1(lgb.predict(X_val[FEATURES]))
blend_val = np.maximum(0, 0.3 * xgb_val + 0.7 * lgb_val)

rmsle = np.sqrt(np.mean((np.log1p(blend_val) - np.log1p(y_val)) ** 2))
rmse  = np.sqrt(mean_squared_error(y_val, blend_val))
mae   = mean_absolute_error(y_val, blend_val)
r2    = r2_score(y_val, blend_val)
mean_y = y_val.mean()

print("\n" + "=" * 52)
print("  XGB + LGB ENSEMBLE  —  validation results")
print("=" * 52)
print(f"  RMSLE : {rmsle:.5f}")
print(f"  RMSE  : {rmse:.2f}  ({rmse / mean_y * 100:.1f}% of mean sales)")
print(f"  MAE   : {mae:.2f}  ({mae  / mean_y * 100:.1f}% of mean sales)")
print(f"  R²    : {r2:.4f}")
print("=" * 52)

# ── 6. KAGGLE SUBMISSION ──────────────────────────────────────────────────────
print("\nGenerating Kaggle submission...")
xgb_test  = np.expm1(xgb.predict(test_final[FEATURES]))
lgb_test  = np.expm1(lgb.predict(test_final[FEATURES]))
test_preds = np.maximum(0, 0.3 * xgb_test + 0.7 * lgb_test)

submission = pd.read_csv(DATA_DIR / "sample_submission.csv")
submission["sales"] = test_preds
out_path = DATA_DIR / "moja_kaggle_napoved.csv"
submission.to_csv(out_path, index=False)
print(f"Submission saved → {out_path}  ({len(submission):,} rows)")

# ── 7. FEATURE IMPORTANCE ─────────────────────────────────────────────────────
xgb_imp = pd.Series(xgb.feature_importances_, index=FEATURES)
lgb_imp = pd.Series(lgb.feature_importances_ / lgb.feature_importances_.sum(),
                    index=FEATURES)
avg_imp = ((xgb_imp / xgb_imp.sum()) + lgb_imp) / 2
avg_imp = avg_imp.sort_values()

fig, ax = plt.subplots(figsize=(10, 7))
avg_imp.plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
ax.set_title("Feature Importance — XGB + LGB Ensemble (avg)", fontsize=13)
ax.set_xlabel("Normalised importance")
plt.tight_layout()
plt.savefig(DATA_DIR / "ensemble_feature_importance.png", dpi=150)
plt.show()
print("Feature importance chart saved → ensemble_feature_importance.png")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# 1. SETUP & CONFIGURATION
DATA_DIR = Path(__file__).parent
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Features list for the model
FEATURES = [
    "store_nbr", "family", "onpromotion", "next_promo_7", "next_promo_14",
    "type", "cluster", "city", "state",
    "dcoilwtico", "oil_l7", "oil_diff",
    "is_nat_holiday", "is_reg_holiday", "is_loc_holiday",
    "dow_sin", "dow_cos", "month", "week", "is_month_start", "is_month_end",
    "payday_signal", "store_family_mean", "family_mean", "store_mean",
    "lag_16", "lag_17", "lag_18", "lag_21", "lag_28", "lag_35",
    "roll_3", "roll_7", "roll_14", "roll_28",
    "trans_lag_16"
]

def load_and_preprocess():
    print("Loading data...")
    train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["date"])
    test = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["date"])
    stores = pd.read_csv(DATA_DIR / "stores.csv")
    oil = pd.read_csv(DATA_DIR / "oil.csv", parse_dates=["date"])
    holidays = pd.read_csv(DATA_DIR / "holidays_events.csv", parse_dates=["date"])
    transactions = pd.read_csv(DATA_DIR / "transactions.csv", parse_dates=["date"])

    # Filter for recent data to avoid old patterns (2016 onwards)
    train = train[train["date"] >= "2016-01-01"].copy()
    
    # Clip outliers
    limit = train["sales"].quantile(0.999)
    train["sales"] = train["sales"].clip(upper=limit)

    return train, test, stores, oil, holidays, transactions

def engineer_features(train, test, stores, oil, holidays, transactions):
    print("Engineering features...")
    
    # Preprocess Oil
    oil = oil.copy()
    oil["dcoilwtico"] = oil["dcoilwtico"].ffill().bfill()
    oil["oil_l7"] = oil["dcoilwtico"].rolling(7).mean().ffill().bfill()
    oil["oil_diff"] = oil["dcoilwtico"].diff().fillna(0)
    
    # Preprocess Holidays (transferred = False means the holiday happened)
    hols = holidays[holidays["transferred"] == False]
    nat_hols = hols[hols["locale"] == "National"].drop_duplicates("date")
    reg_hols = hols[hols["locale"] == "Regional"].drop_duplicates(["date", "locale_name"])
    loc_hols = hols[hols["locale"] == "Local"].drop_duplicates(["date", "locale_name"])

    # Preprocess Transactions
    transactions = transactions.copy()
    transactions["trans_lag_16"] = transactions.groupby("store_nbr")["transactions"].shift(16)

    # Combine train and test for feature engineering
    combined = pd.concat([train, test], axis=0).sort_values(["store_nbr", "family", "date"])
    
    # Merge datasets
    combined = combined.merge(stores, on="store_nbr", how="left")
    combined = combined.merge(oil, on="date", how="left")
    combined = combined.merge(transactions[["date", "store_nbr", "trans_lag_16"]], on=["date", "store_nbr"], how="left")

    # Holiday Flags
    combined["is_nat_holiday"] = combined["date"].isin(nat_hols["date"]).astype(int)
    
    # Regional/Local Holiday matching
    # We use a vectorized approach where possible
    combined = combined.merge(reg_hols[["date", "locale_name"]].assign(is_reg_holiday=1), 
                             left_on=["date", "state"], right_on=["date", "locale_name"], how="left")
    combined["is_reg_holiday"] = combined["is_reg_holiday"].fillna(0).astype(int)
    combined.drop(columns=["locale_name"], inplace=True)
    
    combined = combined.merge(loc_hols[["date", "locale_name"]].assign(is_loc_holiday=1), 
                             left_on=["date", "city"], right_on=["date", "locale_name"], how="left")
    combined["is_loc_holiday"] = combined["is_loc_holiday"].fillna(0).astype(int)
    combined.drop(columns=["locale_name"], inplace=True)

    # Time Features
    combined["day"] = combined["date"].dt.day
    combined["dow"] = combined["date"].dt.dayofweek
    combined["month"] = combined["date"].dt.month
    combined["week"] = combined["date"].dt.isocalendar().week.astype(int)
    combined["is_month_start"] = combined["date"].dt.is_month_start.astype(int)
    combined["is_month_end"] = combined["date"].dt.is_month_end.astype(int)
    
    # Payday Signal (15th and end of month)
    combined["dist_payday"] = np.minimum(np.abs(combined["day"] - 15), 
                                        np.abs(combined["day"] - combined["date"].dt.days_in_month))
    combined["payday_signal"] = np.exp(-0.3 * combined["dist_payday"])
    
    # Cyclical Encoding for day of week
    combined["dow_sin"] = np.sin(2 * np.pi * combined["dow"] / 7)
    combined["dow_cos"] = np.cos(2 * np.pi * combined["dow"] / 7)

    # Target Encoding (Computed only from train data before split_date to avoid leakage)
    # But for final model we'll use all train data up to validation split
    split_date = train["date"].max() - pd.Timedelta(days=16)
    
    train_part = combined[(combined["date"] <= split_date) & (combined["sales"].notna())]
    
    tf_mean = train_part.groupby(["store_nbr", "family"])["sales"].mean().reset_index().rename(columns={"sales": "store_family_mean"})
    f_mean = train_part.groupby(["family"])["sales"].mean().reset_index().rename(columns={"sales": "family_mean"})
    s_mean = train_part.groupby(["store_nbr"])["sales"].mean().reset_index().rename(columns={"sales": "store_mean"})
    
    combined = combined.merge(tf_mean, on=["store_nbr", "family"], how="left")
    combined = combined.merge(f_mean, on=["family"], how="left")
    combined = combined.merge(s_mean, on=["store_nbr"], how="left")

    # Category Encoding
    for col in ["family", "city", "state", "type", "cluster"]:
        combined[col] = combined[col].astype("category").cat.codes

    # Lags & Rolling Stats
    for l in [16, 17, 18, 21, 28, 35]:
        combined[f"lag_{l}"] = combined.groupby(["store_nbr", "family"])["sales"].shift(l)
    
    for w in [3, 7, 14, 28]:
        combined[f"roll_{w}"] = combined.groupby(["store_nbr", "family"])["lag_16"].transform(lambda x: x.rolling(w).mean())

    # Future Promotion Look-ahead
    combined["next_promo_7"] = combined.groupby(["store_nbr", "family"])["onpromotion"].transform(lambda x: x.shift(-7).rolling(7).mean())
    combined["next_promo_14"] = combined.groupby(["store_nbr", "family"])["onpromotion"].transform(lambda x: x.shift(-14).rolling(14).mean())

    # Fill NaNs from lags/rolling (only for features, keep sales as NaN for test split)
    feat_cols = [c for c in combined.columns if c != "sales"]
    combined[feat_cols] = combined[feat_cols].fillna(0)
    
    return combined, split_date

def train_and_evaluate(combined, split_date):
    print("Training and evaluating...")
    
    train_final = combined[combined["sales"].notna()].copy()
    test_final = combined[combined["sales"].isna()].copy()
    
    # Log transform target
    train_final["log_sales"] = np.log1p(train_final["sales"])
    
    X_train = train_final[train_final["date"] <= split_date]
    X_val = train_final[train_final["date"] > split_date]
    
    y_train = X_train["log_sales"]
    y_val_log = X_val["log_sales"]
    y_val_true = X_val["sales"]

    # 1. XGBoost
    print("  Training XGBoost...")
    xgb = XGBRegressor(
        n_estimators=1200, 
        learning_rate=0.02, 
        max_depth=7,
        subsample=0.85, 
        colsample_bytree=0.7,
        n_jobs=-1, 
        random_state=RANDOM_STATE,
        early_stopping_rounds=50, 
        eval_metric="rmse"
    )
    xgb.fit(X_train[FEATURES], y_train, 
            eval_set=[(X_val[FEATURES], y_val_log)], 
            verbose=False)
    
    # 2. LightGBM
    print("  Training LightGBM...")
    lgb = LGBMRegressor(
        n_estimators=1500, 
        learning_rate=0.02, 
        num_leaves=200,
        feature_fraction=0.7, 
        bagging_fraction=0.8, 
        n_jobs=-1, 
        random_state=RANDOM_STATE, 
        verbose=-1
    )
    lgb.fit(X_train[FEATURES], y_train, 
            eval_set=[(X_val[FEATURES], y_val_log)], 
            callbacks=[])

    # Ensemble Evaluation
    xgb_val = np.expm1(xgb.predict(X_val[FEATURES]))
    lgb_val = np.expm1(lgb.predict(X_val[FEATURES]))
    
    # Weighted average: 0.4 XGB + 0.6 LGB (LGB is often slightly better here)
    final_val = np.maximum(0, 0.4 * xgb_val + 0.6 * lgb_val)
    
    rmsle = np.sqrt(np.mean((np.log1p(final_val) - np.log1p(y_val_true)) ** 2))
    rmse = np.sqrt(mean_squared_error(y_val_true, final_val))
    mae = mean_absolute_error(y_val_true, final_val)
    r2 = r2_score(y_val_true, final_val)
    
    mean_y = y_val_true.mean()
    rel_rmse = (rmse / mean_y) * 100
    rel_mae = (mae / mean_y) * 100

    print("\n" + "=" * 50)
    print("  SUPER OPTIMIZED ENSEMBLE RESULTS")
    print("=" * 50)
    print(f"  Mean Sales in Validation: {mean_y:.2f}")
    print(f"  RMSLE : {rmsle:.5f}")
    print(f"  RMSE  : {rmse:.2f} ({rel_rmse:.2f}%)")
    print(f"  MAE   : {mae:.2f} ({rel_mae:.2f}%)")
    print(f"  R2    : {r2:.4f}")
    print("=" * 50)
    
    # Final Submission Generation
    print("\nGenerating final submission...")
    xgb_test = np.expm1(xgb.predict(test_final[FEATURES]))
    lgb_test = np.expm1(lgb.predict(test_final[FEATURES]))
    test_preds = np.maximum(0, 0.4 * xgb_test + 0.6 * lgb_test)
    
    submission = pd.read_csv(DATA_DIR / "sample_submission.csv")
    submission["sales"] = test_preds
    out_path = DATA_DIR / "super_optimizirana_napoved_v2.csv"
    submission.to_csv(out_path, index=False)
    print(f"Submission saved to {out_path}")

    return xgb, lgb

if __name__ == "__main__":
    train, test, stores, oil, holidays, transactions = load_and_preprocess()
    combined, split_date = engineer_features(train, test, stores, oil, holidays, transactions)
    xgb, lgb = train_and_evaluate(combined, split_date)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 1. SETUP
DATA_DIR = Path(__file__).parent

print("🚀 Final Optimization Round: Targeting 15-25% RMSE...")
train    = pd.read_csv(DATA_DIR / "train.csv",           parse_dates=['date'])
test     = pd.read_csv(DATA_DIR / "test.csv",            parse_dates=['date'])
stores   = pd.read_csv(DATA_DIR / "stores.csv")
oil      = pd.read_csv(DATA_DIR / "oil.csv",             parse_dates=['date'])
holidays = pd.read_csv(DATA_DIR / "holidays_events.csv", parse_dates=['date'])

# Use only 2017 and clip extreme outliers
train = train[train['date'] >= '2017-01-01']
limit = train['sales'].quantile(0.999)
train['sales'] = train['sales'].clip(upper=limit)

# 2. TARGET ENCODING (PRE-ENCODING)
split_date = train['date'].max() - pd.Timedelta(days=16)
target_mean = train[train['date'] <= split_date].groupby(['store_nbr', 'family'])['sales'].mean().reset_index()
target_mean.rename(columns={'sales': 'store_family_mean'}, inplace=True)

# 3. FEATURE ENGINEERING
oil['dcoilwtico'] = oil['dcoilwtico'].ffill().bfill()
oil['oil_l7'] = oil['dcoilwtico'].rolling(7).mean().ffill().bfill()

def extract_features(df):
    df = df.merge(stores, on='store_nbr', how='left')
    df = df.merge(oil, on='date', how='left')
    df = df.merge(target_mean, on=['store_nbr', 'family'], how='left').fillna(0)
    
    hols = holidays[holidays['transferred'] == False]
    nat_hols = hols[hols['locale'] == 'National'].drop_duplicates('date')
    df['is_nat_holiday'] = df['date'].isin(nat_hols['date']).astype(int)
    
    df['day'] = df['date'].dt.day
    df['dow'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    
    # Payday
    df['dist_payday'] = np.minimum(np.abs(df['day'] - 15), np.abs(df['day'] - df['date'].dt.days_in_month))
    df['payday_signal'] = np.exp(-0.3 * df['dist_payday'])
    
    # Cyclical
    df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
    
    # Category Encoding
    for col in ['family', 'city', 'state', 'type', 'cluster']:
        df[col] = df[col].astype('category').cat.codes
        
    return df

train_df = extract_features(train)
test_df = extract_features(test)

# 4. ADVANCED LAGS
full_df = pd.concat([train_df, test_df], axis=0).sort_values(['store_nbr', 'family', 'date'])

# Multi-range Lags
for l in [16, 17, 18, 21, 28]:
    full_df[f'lag_{l}'] = full_df.groupby(['store_nbr', 'family'])['sales'].shift(l)

# Rolling stats
full_df['roll_3'] = full_df.groupby(['store_nbr', 'family'])['lag_16'].transform(lambda x: x.rolling(3).mean())
full_df['roll_7'] = full_df.groupby(['store_nbr', 'family'])['lag_16'].transform(lambda x: x.rolling(7).mean())
full_df['roll_14'] = full_df.groupby(['store_nbr', 'family'])['lag_16'].transform(lambda x: x.rolling(14).mean())

# Future Promo
full_df['next_promo_7'] = full_df.groupby(['store_nbr', 'family'])['onpromotion'].transform(lambda x: x.shift(-7).rolling(7).mean())

train_final = full_df[full_df['sales'].notna()].fillna(0)
train_final['log_sales'] = np.log1p(train_final['sales'])

X_train = train_final[train_final['date'] <= split_date]
X_val = train_final[train_final['date'] > split_date]

features = ['store_nbr', 'family', 'onpromotion', 'next_promo_7',
            'type', 'cluster', 'dcoilwtico', 'oil_l7', 'is_nat_holiday',
            'dow_sin', 'dow_cos', 'month', 'payday_signal', 'store_family_mean',
            'lag_16', 'lag_17', 'lag_18', 'lag_21', 'lag_28',
            'roll_3', 'roll_7', 'roll_14']

# 5. TRAINING ENSEMBLE
print("⚔️ Training Final Ensemble...")
xgb = XGBRegressor(n_estimators=1000, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42, early_stopping_rounds=50, eval_metric="rmse")
xgb.fit(X_train[features], X_train['log_sales'], eval_set=[(X_val[features], X_val['log_sales'])], verbose=False)

lgb = LGBMRegressor(n_estimators=1200, learning_rate=0.03, num_leaves=150, feature_fraction=0.8, bagging_fraction=0.8, n_jobs=-1, random_state=42, verbose=-1)
lgb.fit(X_train[features], X_train['log_sales'], eval_set=[(X_val[features], X_val['log_sales'])], callbacks=[])

# 6. EVALUATION
xgb_preds = np.expm1(xgb.predict(X_val[features]))
lgb_preds = np.expm1(lgb.predict(X_val[features]))
final_preds = (0.3 * xgb_preds) + (0.7 * lgb_preds)
final_preds = np.maximum(0, final_preds)

y_true = X_val['sales']
mean_y = y_true.mean()
rmse = np.sqrt(mean_squared_error(y_true, final_preds))
mae = mean_absolute_error(y_true, final_preds)
rel_rmse = (rmse / mean_y) * 100
rel_mae = (mae / mean_y) * 100

print(f"\n✨ FINAL OPTIMIZED PERFORMANCE RESULTS:")
print(f"Mean Sales in Validation: {mean_y:.2f}")
print(f"- MAE:   {mae:.2f} ({rel_mae:.2f}%)")
print(f"- RMSE:  {rmse:.2f} ({rel_rmse:.2f}%)")
print(f"- R2:    {r2_score(y_true, final_preds):.4f}")

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

print("🚀 Loading & Engineering Optimized Features...")
train        = pd.read_csv(DATA_DIR / "train.csv",           parse_dates=['date'])
test         = pd.read_csv(DATA_DIR / "test.csv",            parse_dates=['date'])
stores       = pd.read_csv(DATA_DIR / "stores.csv")
oil          = pd.read_csv(DATA_DIR / "oil.csv",             parse_dates=['date'])
holidays     = pd.read_csv(DATA_DIR / "holidays_events.csv", parse_dates=['date'])
transactions = pd.read_csv(DATA_DIR / "transactions.csv",    parse_dates=['date'])

# Only use recent data
train = train[train['date'] >= '2017-01-01']

# 2. ADVANCED FEATURE ENGINEERING
oil['dcoilwtico'] = oil['dcoilwtico'].ffill().bfill()
oil['oil_l7'] = oil['dcoilwtico'].rolling(7).mean().ffill().bfill()
oil['oil_diff'] = oil['dcoilwtico'].diff().fillna(0)

def extract_features(df):
    df = df.merge(stores, on='store_nbr', how='left')
    df = df.merge(oil, on='date', how='left')
    
    # Granular Holiday Features
    hols = holidays[holidays['transferred'] == False]
    nat_hols = hols[hols['locale'] == 'National'].drop_duplicates('date')
    reg_hols = hols[hols['locale'] == 'Regional'].drop_duplicates(['date', 'locale_name'])
    loc_hols = hols[hols['locale'] == 'Local'].drop_duplicates(['date', 'locale_name'])
    
    df['is_nat_holiday'] = df['date'].isin(nat_hols['date']).astype(int)
    # Regional/Local mapping
    df['is_reg_holiday'] = df.apply(lambda x: 1 if ((x['date'] in reg_hols['date'].values) and (x['state'] in reg_hols['locale_name'].values)) else 0, axis=1)
    df['is_loc_holiday'] = df.apply(lambda x: 1 if ((x['date'] in loc_hols['date'].values) and (x['city'] in loc_hols['locale_name'].values)) else 0, axis=1)
    
    # Time
    df['day'] = df['date'].dt.day
    df['dow'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['week'] = df['date'].dt.isocalendar().week.astype(int)
    
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

# Lagged Transactions (Foot traffic proxy)
# Use lag 16 for transactions too to be consistent
transactions['trans_lag_16'] = transactions.groupby('store_nbr')['transactions'].shift(16)
train_df = train_df.merge(transactions[['date', 'store_nbr', 'trans_lag_16']], on=['date', 'store_nbr'], how='left').fillna(0)
test_df = test_df.merge(transactions[['date', 'store_nbr', 'trans_lag_16']], on=['date', 'store_nbr'], how='left').fillna(0)

# 3. GROUP-LEVEL LAGS
full_df = pd.concat([train_df, test_df], axis=0).sort_values(['store_nbr', 'family', 'date'])

# Lags
for l in [16, 21, 28, 35]:
    full_df[f'lag_{l}'] = full_df.groupby(['store_nbr', 'family'])['sales'].shift(l)

# Rolling windows of lags
full_df['rolling_lag_16_w3'] = full_df.groupby(['store_nbr', 'family'])['lag_16'].transform(lambda x: x.rolling(3).mean())
full_df['rolling_lag_16_w7'] = full_df.groupby(['store_nbr', 'family'])['lag_16'].transform(lambda x: x.rolling(7).mean())
full_df['rolling_lag_16_w14'] = full_df.groupby(['store_nbr', 'family'])['lag_16'].transform(lambda x: x.rolling(14).mean())

# Future Promo (Aggressive look-ahead)
full_df['next_promo_7'] = full_df.groupby(['store_nbr', 'family'])['onpromotion'].transform(lambda x: x.shift(-7).rolling(7).mean())
full_df['next_promo_14'] = full_df.groupby(['store_nbr', 'family'])['onpromotion'].transform(lambda x: x.shift(-14).rolling(14).mean())

train_final = full_df[full_df['sales'].notna()].fillna(0)
test_final = full_df[full_df['sales'].isna()].fillna(0)

# Target log transform
train_final['log_sales'] = np.log1p(train_final['sales'])

split_date = train_final['date'].max() - pd.Timedelta(days=16)
X_train = train_final[train_final['date'] <= split_date]
X_val = train_final[train_final['date'] > split_date]

features = ['store_nbr', 'family', 'onpromotion', 'next_promo_7', 'next_promo_14',
            'city', 'type', 'cluster', 'dcoilwtico', 'oil_l7', 'oil_diff',
            'is_nat_holiday', 'is_reg_holiday', 'is_loc_holiday',
            'dow_sin', 'dow_cos', 'month', 'week', 'payday_signal',
            'lag_16', 'lag_21', 'lag_28', 'lag_35',
            'rolling_lag_16_w3', 'rolling_lag_16_w7', 'rolling_lag_16_w14',
            'trans_lag_16']

# 4. TRAINING OPTIMIZED ENSEMBLE
print("⚔️ Training Optimized Ensemble...")
xgb = XGBRegressor(n_estimators=1000, learning_rate=0.03, max_depth=8, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42, early_stopping_rounds=50, eval_metric="rmse")
xgb.fit(X_train[features], X_train['log_sales'], eval_set=[(X_val[features], X_val['log_sales'])], verbose=False)

lgb = LGBMRegressor(n_estimators=1000, learning_rate=0.03, num_leaves=128, feature_fraction=0.8, bagging_fraction=0.8, n_jobs=-1, random_state=42, verbose=-1)
lgb.fit(X_train[features], X_train['log_sales'], eval_set=[(X_val[features], X_val['log_sales'])], callbacks=[])

# 5. FINAL EVALUATION
xgb_preds = np.expm1(xgb.predict(X_val[features]))
lgb_preds = np.expm1(lgb.predict(X_val[features]))
final_preds = (0.5 * xgb_preds) + (0.5 * lgb_preds)
final_preds = np.maximum(0, final_preds)

y_true = X_val['sales']
mean_y = y_true.mean()

rmse = np.sqrt(mean_squared_error(y_true, final_preds))
mae = mean_absolute_error(y_true, final_preds)
r2 = r2_score(y_true, final_preds)

rel_rmse = (rmse / mean_y) * 100
rel_mae = (mae / mean_y) * 100

print(f"\n✨ OPTIMIZED PERFORMANCE RESULTS:")
print(f"Mean Sales in Validation: {mean_y:.2f}")
print(f"- MAE:   {mae:.2f} ({rel_mae:.2f}%) -> Target: 10-20%")
print(f"- RMSE:  {rmse:.2f} ({rel_rmse:.2f}%) -> Target: 15-25%")
print(f"- R2:    {r2:.4f}")

if 10 <= rel_mae <= 20 and 15 <= rel_rmse <= 25:
    print("\n✅ MISSION ACCOMPLISHED: Results within requested bounds!")
else:
    print("\n⚠️ Almost there! Reviewing for further tweaks...")

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

from features import (load_data, prep_oil, get_split_date, make_target_encoding,
                      get_nat_hols, engineer_features, add_lags)

torch.manual_seed(42)
np.random.seed(42)

DATA_DIR = Path(__file__).parent
SEQ_LEN  = 14
BATCH    = 1024
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print("Loading data...")
train, _, stores, oil, holidays = load_data(DATA_DIR)
# holidays kept in scope so engineer_features can add regional/local flags
train = train[train["date"] >= "2017-01-01"].copy()

# ── 2. SHARED SETUP ───────────────────────────────────────────────────────────
oil         = prep_oil(oil)
split_date  = get_split_date(train)
target_mean = make_target_encoding(train, split_date)
nat_hols    = get_nat_hols(holidays)

# ── 3. FEATURE ENGINEERING ────────────────────────────────────────────────────
train_df = engineer_features(train, stores, oil, target_mean, nat_hols, holidays)
train_df = train_df.sort_values(["store_nbr", "family", "date"])
train_df = add_lags(train_df, lags=[16, 17, 18, 21, 28], rolls=[3, 7, 14])
train_df["log_sales"] = np.log1p(train_df["sales"])
train_df = train_df.fillna(0)

FEATURES = [
    "store_nbr", "family", "onpromotion", "next_promo_7",
    "type", "cluster", "dcoilwtico", "oil_l7",
    "is_nat_holiday", "is_reg_holiday", "is_loc_holiday",
    "dow_sin", "dow_cos", "month", "payday_signal", "store_family_mean",
    "lag_16", "lag_17", "lag_18", "lag_21", "lag_28",
    "roll_3", "roll_7", "roll_14",
]

X_tr = train_df[train_df["date"] <= split_date]
X_vl = train_df[train_df["date"] >  split_date]
y_vl = X_vl["sales"].values

# ── 5. METRICS ────────────────────────────────────────────────────────────────
results = []

def record(name, y_true, y_pred):
    y_pred = np.maximum(0, y_pred)
    rmsle  = np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))
    rmse   = np.sqrt(mean_squared_error(y_true, y_pred))
    mae    = mean_absolute_error(y_true, y_pred)
    r2     = r2_score(y_true, y_pred)
    results.append({"Model": name, "RMSLE": round(rmsle, 5), "RMSE": round(rmse, 2),
                    "MAE": round(mae, 2), "R²": round(r2, 4)})
    print(f"  {name:<35} RMSLE={rmsle:.5f}  RMSE={rmse:7.1f}  MAE={mae:7.1f}  R²={r2:.4f}")

print(f"\nValidation period: {split_date.date()} → {train['date'].max().date()}  ({len(X_vl):,} rows)\n")

# ── 6. LINEAR REGRESSION ──────────────────────────────────────────────────────
print("Training Linear Regression...")
lr = LinearRegression()
lr.fit(X_tr[FEATURES], X_tr["log_sales"])
record("Linear Regression", y_vl, np.expm1(lr.predict(X_vl[FEATURES])))

# ── 7. RANDOM FOREST ──────────────────────────────────────────────────────────
print("Training Random Forest (200 trees)...")
rf = RandomForestRegressor(n_estimators=200, max_depth=16, n_jobs=-1, random_state=42)
rf.fit(X_tr[FEATURES], X_tr["log_sales"])
record("Random Forest", y_vl, np.expm1(rf.predict(X_vl[FEATURES])))

# ── 8. XGBOOST ────────────────────────────────────────────────────────────────
print("Training XGBoost...")
xgb = XGBRegressor(
    n_estimators=1000, learning_rate=0.03, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42,
    early_stopping_rounds=50, eval_metric="rmse",
)
xgb.fit(X_tr[FEATURES], X_tr["log_sales"],
        eval_set=[(X_vl[FEATURES], X_vl["log_sales"])], verbose=False)
record("XGBoost", y_vl, np.expm1(xgb.predict(X_vl[FEATURES])))

# ── 9. LIGHTGBM ───────────────────────────────────────────────────────────────
print("Training LightGBM...")
lgb = LGBMRegressor(
    n_estimators=1200, learning_rate=0.03, num_leaves=150,
    feature_fraction=0.8, bagging_fraction=0.8, n_jobs=-1,
    random_state=42, verbose=-1,
)
lgb.fit(X_tr[FEATURES], X_tr["log_sales"],
        eval_set=[(X_vl[FEATURES], X_vl["log_sales"])], callbacks=[])
record("LightGBM", y_vl, np.expm1(lgb.predict(X_vl[FEATURES])))

# ── 10. XGB + LGB ENSEMBLE ────────────────────────────────────────────────────
ensemble = 0.3 * np.expm1(xgb.predict(X_vl[FEATURES])) + 0.7 * np.expm1(lgb.predict(X_vl[FEATURES]))
record("XGBoost + LightGBM (0.3/0.7)", y_vl, ensemble)

# ── 11. LSTM (load from checkpoint) ───────────────────────────────────────────
class LSTMForecaster(nn.Module):
    def __init__(self, n_features, hidden=128, n_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, n_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])

def eval_dl_from_ckpt(ckpt_path, ModelClass, label):
    if not ckpt_path.exists():
        print(f"  {label}: checkpoint not found — run the training script first")
        return
    ckpt     = torch.load(ckpt_path, map_location="cpu")
    dl_feats = ckpt["features"]
    scaler   = ckpt["scaler"]

    scaled_df = train_df.copy()
    scaled_df[dl_feats] = scaler.transform(scaled_df[dl_feats])

    X_seq, y_seq = [], []
    for (_, _), group in scaled_df.groupby(["store_nbr", "family"]):
        group  = group.sort_values("date").reset_index(drop=True)
        feat   = group[dl_feats].values.astype(np.float32)
        target = group["log_sales"].values.astype(np.float32)
        dates  = group["date"].values
        for i in range(SEQ_LEN, len(group)):
            if dates[i] > np.datetime64(split_date):
                X_seq.append(feat[i - SEQ_LEN : i])
                y_seq.append(target[i])

    X_arr = np.array(X_seq, dtype=np.float32)
    y_arr = np.array(y_seq, dtype=np.float32)

    mdl = ModelClass(len(dl_feats)).to(DEVICE)
    mdl.load_state_dict(ckpt["model_state"])
    mdl.eval()

    log_preds = []
    with torch.no_grad():
        for i in range(0, len(X_arr), BATCH):
            batch = torch.from_numpy(X_arr[i : i + BATCH]).to(DEVICE)
            log_preds.append(mdl(batch).cpu().numpy())
    log_preds = np.concatenate(log_preds).flatten()

    record(label, np.maximum(0, np.expm1(y_arr)), np.expm1(log_preds))


class GRUForecaster(nn.Module):
    def __init__(self, n_features, hidden=128, n_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, n_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )
    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])

print("Loading LSTM checkpoint...")
eval_dl_from_ckpt(DATA_DIR / "lstm_model.pt", LSTMForecaster, "LSTM")

print("Loading GRU checkpoint...")
eval_dl_from_ckpt(DATA_DIR / "gru_model.pt", GRUForecaster, "GRU")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super().__init__()
        import math
        self.dropout = nn.Dropout(p=dropout)
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1), :])


class TransformerForecaster(nn.Module):
    def __init__(self, n_features, d_model=64, nhead=4,
                 num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout, max_len=SEQ_LEN + 1)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head    = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        return self.head(x[:, -1, :])


print("Loading Transformer checkpoint...")
eval_dl_from_ckpt(DATA_DIR / "transformer_model.pt", TransformerForecaster, "Transformer")

# ── 12. SUMMARY TABLE ─────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("  MODEL COMPARISON  —  validation set (last 16 days of training data)")
print("=" * 75)
summary = pd.DataFrame(results).set_index("Model")
print(summary.to_string())
print("=" * 75)
print("Lower RMSLE/RMSE/MAE = better.  Higher R² = better.\n")

out_path = DATA_DIR / "model_comparison.csv"
summary.to_csv(out_path)
print(f"Results saved to {out_path}")

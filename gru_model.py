import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

from features import (load_data, prep_oil, get_split_date, make_target_encoding,
                      get_nat_hols, engineer_features, add_lags)

torch.manual_seed(42)
np.random.seed(42)

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN  = 14
BATCH    = 1024
EPOCHS   = 50
LR       = 1e-3
PATIENCE = 8
DATA_DIR = Path(__file__).parent

print(f"Using device: {DEVICE}")

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print("Loading data...")
train, test, stores, oil, holidays = load_data(DATA_DIR)
train = train[train["date"] >= "2017-01-01"].copy()

# ── 2. SHARED SETUP ───────────────────────────────────────────────────────────
oil         = prep_oil(oil)
split_date  = get_split_date(train)
target_mean = make_target_encoding(train, split_date)
nat_hols    = get_nat_hols(holidays)

# ── 3. FEATURE ENGINEERING ────────────────────────────────────────────────────
train_df = engineer_features(train, stores, oil, target_mean, nat_hols, holidays)
train_df = train_df.sort_values(["store_nbr", "family", "date"])
train_df = add_lags(train_df, lags=[16, 21, 28], rolls=[7, 14])
train_df["log_sales"] = np.log1p(train_df["sales"])
train_df = train_df.fillna(0)

FEATURES = [
    "store_nbr", "family", "onpromotion", "next_promo_7",
    "type", "cluster", "dcoilwtico", "oil_l7",
    "is_nat_holiday", "is_reg_holiday", "is_loc_holiday",
    "dow_sin", "dow_cos", "month", "payday_signal", "store_family_mean",
    "lag_16", "lag_21", "lag_28", "roll_7", "roll_14",
]
N_FEATURES = len(FEATURES)

# ── 4. SCALE FEATURES (fit on train only) ─────────────────────────────────────
scaler = StandardScaler()
scaler.fit(train_df[train_df["date"] <= split_date][FEATURES])
train_df[FEATURES] = scaler.transform(train_df[FEATURES])

# ── 5. BUILD SEQUENCES ────────────────────────────────────────────────────────
print(f"Building sequences (window = {SEQ_LEN} days)...")

X_train_seq, y_train_seq = [], []
X_val_seq,   y_val_seq   = [], []

for (_, _), group in train_df.groupby(["store_nbr", "family"]):
    group  = group.sort_values("date").reset_index(drop=True)
    feat   = group[FEATURES].values.astype(np.float32)
    target = group["log_sales"].values.astype(np.float32)
    dates  = group["date"].values

    for i in range(SEQ_LEN, len(group)):
        seq = feat[i - SEQ_LEN : i]
        lbl = target[i]
        if dates[i] <= np.datetime64(split_date):
            X_train_seq.append(seq)
            y_train_seq.append(lbl)
        else:
            X_val_seq.append(seq)
            y_val_seq.append(lbl)

X_train_seq = np.array(X_train_seq, dtype=np.float32)
y_train_seq = np.array(y_train_seq, dtype=np.float32)
X_val_seq   = np.array(X_val_seq,   dtype=np.float32)
y_val_seq   = np.array(y_val_seq,   dtype=np.float32)

print(f"  Train sequences : {X_train_seq.shape}")
print(f"  Val sequences   : {X_val_seq.shape}")

# ── 6. PYTORCH DATASET & DATALOADER ──────────────────────────────────────────
class SalesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(SalesDataset(X_train_seq, y_train_seq), batch_size=BATCH, shuffle=True,  num_workers=0)
val_loader   = DataLoader(SalesDataset(X_val_seq,   y_val_seq),   batch_size=BATCH, shuffle=False, num_workers=0)

# ── 7. GRU MODEL ──────────────────────────────────────────────────────────────
class GRUForecaster(nn.Module):
    def __init__(self, n_features, hidden=128, n_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features, hidden_size=hidden,
            num_layers=n_layers, batch_first=True, dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])

model     = GRUForecaster(N_FEATURES).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4, min_lr=1e-5)
criterion = nn.MSELoss()

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── 8. TRAINING LOOP ──────────────────────────────────────────────────────────
print("\nTraining GRU...")
train_losses, val_losses = [], []
best_val_loss = float("inf")
best_weights  = None
no_improve    = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    running = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running += loss.item() * len(xb)
    train_loss = running / len(train_loader.dataset)

    model.eval()
    running = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            running += criterion(model(xb), yb).item() * len(xb)
    val_loss = running / len(val_loader.dataset)

    scheduler.step(val_loss)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:>3} | train MSE {train_loss:.5f} | val MSE {val_loss:.5f} | lr {optimizer.param_groups[0]['lr']:.2e}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        no_improve    = 0
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}.")
            break

model.load_state_dict(best_weights)

# ── 9. EVALUATE ───────────────────────────────────────────────────────────────
model.eval()
all_preds = []
with torch.no_grad():
    for xb, _ in val_loader:
        all_preds.append(model(xb.to(DEVICE)).cpu().numpy())

log_preds  = np.concatenate(all_preds).flatten()
preds      = np.maximum(0, np.expm1(log_preds))
y_true     = np.maximum(0, np.expm1(y_val_seq))
mean_sales = y_true.mean()

rmsle = np.sqrt(np.mean((np.log1p(preds) - np.log1p(y_true)) ** 2))
rmse  = np.sqrt(mean_squared_error(y_true, preds))
mae   = mean_absolute_error(y_true, preds)
r2    = r2_score(y_true, preds)

print("\n" + "=" * 50)
print("  GRU RESULTS")
print("=" * 50)
print(f"  RMSLE : {rmsle:.5f}")
print(f"  RMSE  : {rmse:.2f}  ({rmse / mean_sales * 100:.1f}% of mean sales)")
print(f"  MAE   : {mae:.2f}  ({mae  / mean_sales * 100:.1f}% of mean sales)")
print(f"  R2    : {r2:.4f}")

# ── 10. TRAINING CURVE ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(train_losses, label="Train MSE")
ax.plot(val_losses,   label="Val MSE")
ax.set_title("GRU - Training & Validation Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE (log-space)")
ax.legend()
plt.tight_layout()
plt.savefig(DATA_DIR / "gru_training_curve.png", dpi=150)
plt.show()
print("Training curve saved to gru_training_curve.png")

# ── 11. SAVE MODEL ────────────────────────────────────────────────────────────
torch.save({"model_state": best_weights, "scaler": scaler, "features": FEATURES}, DATA_DIR / "gru_model.pt")
print("Model saved to gru_model.pt")

# ── 12. GENERATE KAGGLE SUBMISSION ────────────────────────────────────────────
print("\nGenerating Kaggle submission...")
test_raw = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["date"])

train_inf              = engineer_features(train, stores, oil, target_mean, nat_hols, holidays)
train_inf["log_sales"] = np.log1p(train_inf["sales"])
train_inf["is_test"]   = 0

test_inf           = engineer_features(test_raw, stores, oil, target_mean, nat_hols, holidays)
test_inf["is_test"] = 1

combined = (
    pd.concat([train_inf, test_inf], axis=0)
    .sort_values(["store_nbr", "family", "date"])
    .reset_index(drop=True)
)
combined = add_lags(combined, lags=[16, 21, 28], rolls=[7, 14])
combined = combined.fillna(0)
combined[FEATURES] = scaler.transform(combined[FEATURES])

X_test_seq = []
test_keys  = []

for (snbr, fam), group in combined.groupby(["store_nbr", "family"]):
    group = group.sort_values("date").reset_index(drop=True)
    feat  = group[FEATURES].values.astype(np.float32)
    for i in range(SEQ_LEN, len(group)):
        if group.loc[i, "is_test"] == 1:
            X_test_seq.append(feat[i - SEQ_LEN : i])
            test_keys.append((snbr, fam, group.loc[i, "date"]))

X_test_arr = np.array(X_test_seq, dtype=np.float32)

model.eval()
test_log_preds = []
with torch.no_grad():
    for i in range(0, len(X_test_arr), BATCH):
        batch = torch.from_numpy(X_test_arr[i : i + BATCH]).to(DEVICE)
        test_log_preds.append(model(batch).cpu().numpy())

test_log_preds = np.concatenate(test_log_preds).flatten()
test_preds     = np.maximum(0, np.expm1(test_log_preds))

pred_df = pd.DataFrame(test_keys, columns=["store_nbr", "family", "date"])
pred_df["sales"] = test_preds

submission = (
    test_raw[["id", "date", "store_nbr", "family"]]
    .merge(pred_df, on=["date", "store_nbr", "family"], how="left")
    .fillna(0)[["id", "sales"]]
)

out_path = DATA_DIR / "gru_submission.csv"
submission.to_csv(out_path, index=False)
print(f"Submission saved to {out_path}  ({len(submission)} rows)")

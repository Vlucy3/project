"""
Derives actionable business insights from retail sales data and model outputs.
Outputs: formatted console report + 4 charts saved to disk.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")

from features import (load_data, prep_oil, get_split_date, make_target_encoding,
                      get_nat_hols, engineer_features, add_lags)

DATA_DIR = Path(__file__).parent
report_lines = []

def section(title):
    sep = "=" * 62
    block = f"\n{sep}\n  {title}\n{sep}"
    print(block)
    report_lines.append(block)

def line(text=""):
    print(text)
    report_lines.append(text)

# ── 1. LOAD & PREPARE ─────────────────────────────────────────────────────────
print("Loading data...")
train, test, stores, oil, holidays = load_data(DATA_DIR)
oil = prep_oil(oil)

# Enrich train with auxiliary tables
df = train.copy()
df = df.merge(stores, on="store_nbr", how="left")
df = df.merge(oil[["date", "dcoilwtico"]], on="date", how="left")

df["day"]        = df["date"].dt.day
df["dow"]        = df["date"].dt.dayofweek
df["dow_name"]   = df["date"].dt.day_name()
df["month"]      = df["date"].dt.month
df["month_name"] = df["date"].dt.strftime("%b")
df["year"]       = df["date"].dt.year
df["is_weekend"] = (df["dow"] >= 5).astype(int)

# Payday proximity: 1st & 15th  +/- 1 day
df["dist_15"]    = np.abs(df["day"] - 15)
df["dist_end"]   = np.abs(df["day"] - df["date"].dt.days_in_month)
df["dist_payday"]= np.minimum(df["dist_15"], df["dist_end"])
df["is_payday"]  = (df["dist_payday"] <= 1).astype(int)

# National holidays
hols = holidays[holidays["transferred"] == False]
nat  = hols[hols["locale"] == "National"].drop_duplicates("date")[["date", "description"]]
nat  = nat.rename(columns={"description": "holiday_name"})
df   = df.merge(nat, on="date", how="left")
df["is_holiday"] = df["holiday_name"].notna().astype(int)

# ── 2. INSIGHT: PRODUCT FAMILY PERFORMANCE ────────────────────────────────────
section("1. PRODUCT FAMILY PERFORMANCE  (avg daily sales per store)")

family_grp = df.groupby("family")["sales"]
fam_avg    = family_grp.mean().sort_values(ascending=False)
fam_zero   = (family_grp.apply(lambda x: (x == 0).mean()) * 100).reindex(fam_avg.index)

line(f"  {'Family':<35} {'Avg Sales':>10}  {'Zero-sale days':>14}")
line(f"  {'-'*35} {'-'*10}  {'-'*14}")
for fam in fam_avg.index:
    line(f"  {fam:<35} {fam_avg[fam]:>10.1f}  {fam_zero[fam]:>13.1f}%")

top3  = fam_avg.head(3).index.tolist()
bot3  = fam_avg.tail(3).index.tolist()
line(f"\n  Top revenue families : {', '.join(top3)}")
line(f"  Low-volume families  : {', '.join(bot3)}")
line("  Recommendation: focus inventory buffer on top families; review SKU")
line("  viability for families with >70% zero-sale days.")

# ── 3. INSIGHT: PROMOTION LIFT ────────────────────────────────────────────────
section("2. PROMOTION LIFT BY PRODUCT FAMILY")

promo_df   = df[df["sales"] > 0].copy()
promo_df["has_promo"] = (promo_df["onpromotion"] > 0)

promo_grp  = promo_df.groupby(["family", "has_promo"])["sales"].mean().unstack()
promo_grp.columns = ["no_promo", "with_promo"]
promo_grp  = promo_grp.dropna()
promo_grp["lift_pct"] = (promo_grp["with_promo"] - promo_grp["no_promo"]) / promo_grp["no_promo"] * 100
promo_grp  = promo_grp.sort_values("lift_pct", ascending=False)

line(f"\n  {'Family':<35} {'No Promo':>9}  {'With Promo':>10}  {'Lift':>7}")
line(f"  {'-'*35} {'-'*9}  {'-'*10}  {'-'*7}")
for fam, row in promo_grp.iterrows():
    line(f"  {fam:<35} {row['no_promo']:>9.1f}  {row['with_promo']:>10.1f}  {row['lift_pct']:>+6.1f}%")

best_promo = promo_grp["lift_pct"].idxmax()
line(f"\n  Highest promotion ROI: {best_promo} ({promo_grp.loc[best_promo,'lift_pct']:+.1f}%)")
line("  Recommendation: concentrate promotional spend on high-lift families.")
line("  Families with negative lift may be over-promoted (cannibalisation).")

# ── 4. INSIGHT: HOLIDAY IMPACT ────────────────────────────────────────────────
section("3. HOLIDAY SALES IMPACT (national holidays)")

hol_avg     = df[df["is_holiday"] == 1].groupby("holiday_name")["sales"].mean()
no_hol_avg  = df[df["is_holiday"] == 0]["sales"].mean()
hol_lift    = ((hol_avg - no_hol_avg) / no_hol_avg * 100).sort_values(ascending=False)

line(f"\n  Baseline avg sales (non-holiday): {no_hol_avg:.1f}")
line(f"\n  {'Holiday':<45} {'Lift vs baseline':>16}")
line(f"  {'-'*45} {'-'*16}")
for hol, lift in hol_lift.items():
    line(f"  {hol:<45} {lift:>+15.1f}%")

line("\n  Recommendation: pre-position inventory 2-3 days before high-lift holidays.")
line("  Negative-lift holidays may indicate store closures — verify with ops.")

# ── 5. INSIGHT: SEASONALITY ───────────────────────────────────────────────────
section("4. SEASONALITY PATTERNS")

dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
dow_avg   = df.groupby("dow_name")["sales"].mean().reindex(dow_order)
line("\n  Day-of-week average sales:")
for day, avg in dow_avg.items():
    bar = "█" * int(avg / dow_avg.max() * 30)
    line(f"  {day:<12} {avg:>7.1f}  {bar}")

weekend_lift = (dow_avg[["Saturday","Sunday"]].mean() / dow_avg[["Monday","Tuesday","Wednesday","Thursday","Friday"]].mean() - 1) * 100
line(f"\n  Weekend vs weekday sales lift: {weekend_lift:+.1f}%")

month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
month_avg   = df.groupby("month_name")["sales"].mean().reindex(month_order)
peak_month  = month_avg.idxmax()
trough_month= month_avg.idxmin()
line(f"\n  Peak month: {peak_month} ({month_avg[peak_month]:.1f})  |  Trough: {trough_month} ({month_avg[trough_month]:.1f})")
line(f"  Peak-to-trough ratio: {month_avg[peak_month]/month_avg[trough_month]:.2f}x")

payday_avg    = df[df["is_payday"] == 1]["sales"].mean()
non_payday_avg= df[df["is_payday"] == 0]["sales"].mean()
payday_lift   = (payday_avg / non_payday_avg - 1) * 100
line(f"\n  Payday effect (±1 day of 1st/15th): {payday_lift:+.1f}% sales lift")
line("  Recommendation: increase replenishment cycle frequency around paydays.")

# ── 6. INSIGHT: STORE TYPE PERFORMANCE ───────────────────────────────────────
section("5. STORE TYPE & CLUSTER PERFORMANCE")

type_avg    = df.groupby("type")["sales"].mean().sort_values(ascending=False)
cluster_avg = df.groupby("cluster")["sales"].mean().sort_values(ascending=False)

line("\n  Average sales by store type:")
for t, avg in type_avg.items():
    line(f"    Type {t}: {avg:.1f}")

line(f"\n  Best cluster: {cluster_avg.idxmax()} ({cluster_avg.max():.1f})  |  "
     f"Weakest: {cluster_avg.idxmin()} ({cluster_avg.min():.1f})")
line(f"  Performance spread across clusters: {cluster_avg.max()/cluster_avg.min():.2f}x")
line("  Recommendation: use high-performing clusters as benchmarks for")
line("  promotional planning and assortment decisions in weaker clusters.")

# ── 7. INSIGHT: FEATURE IMPORTANCE FROM RF ────────────────────────────────────
section("6. WHAT DRIVES SALES MOST  (Random Forest feature importance)")

split_date  = get_split_date(train[train["date"] >= "2017-01-01"])
target_mean = make_target_encoding(train[train["date"] >= "2017-01-01"], split_date)
nat_hols_df = get_nat_hols(holidays)

rf_df = engineer_features(train[train["date"] >= "2017-01-01"].copy(),
                          stores, oil, target_mean, nat_hols_df, holidays)
rf_df = rf_df.sort_values(["store_nbr", "family", "date"])
rf_df = add_lags(rf_df, lags=[16, 21, 28], rolls=[7, 14])
rf_df["log_sales"] = np.log1p(rf_df["sales"])
rf_df = rf_df.fillna(0)

RF_FEATURES = [
    "store_nbr", "family", "onpromotion", "next_promo_7",
    "type", "cluster", "dcoilwtico", "oil_l7",
    "is_nat_holiday", "is_reg_holiday", "is_loc_holiday",
    "dow_sin", "dow_cos", "month", "payday_signal", "store_family_mean",
    "lag_16", "lag_21", "lag_28", "roll_7", "roll_14",
]
FEATURE_LABELS = {
    "store_family_mean": "Historical avg (store×family)",
    "lag_16": "Sales 16 days ago",
    "lag_21": "Sales 21 days ago",
    "lag_28": "Sales 28 days ago",
    "roll_7": "7-day rolling avg",
    "roll_14": "14-day rolling avg",
    "onpromotion": "Items on promotion (current)",
    "next_promo_7": "Upcoming promotions (7-day)",
    "store_nbr": "Store identity",
    "family": "Product family",
    "cluster": "Store cluster",
    "type": "Store type",
    "month": "Month",
    "payday_signal": "Payday proximity",
    "dow_sin": "Day of week (sin)",
    "dow_cos": "Day of week (cos)",
    "dcoilwtico": "Oil price",
    "oil_l7": "Oil price (7-day avg)",
    "is_nat_holiday": "National holiday",
    "is_reg_holiday": "Regional holiday",
    "is_loc_holiday": "Local holiday",
}

X_tr = rf_df[rf_df["date"] <= split_date]

print("\n  Training RF for importance (100 trees, may take ~30s)...")
rf = RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42)
rf.fit(X_tr[RF_FEATURES], X_tr["log_sales"])

imp = pd.Series(rf.feature_importances_, index=RF_FEATURES).sort_values(ascending=False)
line(f"\n  {'Driver':<40} {'Importance':>10}")
line(f"  {'-'*40} {'-'*10}")
for feat, val in imp.items():
    label = FEATURE_LABELS.get(feat, feat)
    line(f"  {label:<40} {val:>10.4f}")

line("\n  Top 3 takeaways:")
line("  1. Past sales (lags + rolling avg) dominate — strong autocorrelation.")
line("     Recommendation: monitor stock-outs carefully; a zero today propagates")
line("     into underforecasting 2-4 weeks later.")
line("  2. Promotional signals rank high — both current and upcoming promotions.")
line("     Recommendation: share promotion calendars with supply chain at least")
line("     2 weeks in advance.")
line("  3. Oil price contributes despite not being a direct demand driver —")
line("     likely a macro proxy. Recommendation: track oil trends as an early")
line("     warning signal for consumer spending shifts in Ecuador.")

# ── 8. CHARTS ─────────────────────────────────────────────────────────────────
print("\nGenerating charts...")

# Chart 1: Promotion lift by family (top 15)
fig, ax = plt.subplots(figsize=(10, 6))
top_lift = promo_grp["lift_pct"].sort_values().tail(15)
colors   = ["#d73027" if v < 0 else "#1a9850" for v in top_lift]
top_lift.plot(kind="barh", ax=ax, color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Promotion Sales Lift by Product Family (top 15)", fontsize=13)
ax.set_xlabel("Sales lift vs non-promotion days (%)")
plt.tight_layout()
plt.savefig(DATA_DIR / "insight_promo_lift.png", dpi=150)
plt.close()

# Chart 2: Monthly seasonality
fig, ax = plt.subplots(figsize=(10, 4))
month_avg.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
ax.set_title("Average Daily Sales by Month", fontsize=13)
ax.set_xlabel("")
ax.set_ylabel("Avg Sales")
ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig(DATA_DIR / "insight_seasonality.png", dpi=150)
plt.close()

# Chart 3: Feature importance
fig, ax = plt.subplots(figsize=(10, 6))
imp_plot = imp.sort_values().tail(15)
imp_plot.index = [FEATURE_LABELS.get(f, f) for f in imp_plot.index]
imp_plot.plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Top 15 Sales Drivers (Random Forest Importance)", fontsize=13)
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(DATA_DIR / "insight_feature_importance.png", dpi=150)
plt.close()

# Chart 4: Day-of-week pattern
fig, ax = plt.subplots(figsize=(8, 4))
dow_avg.plot(kind="bar", ax=ax, color="coral", edgecolor="white")
ax.set_title("Average Daily Sales by Day of Week", fontsize=13)
ax.set_xlabel("")
ax.set_ylabel("Avg Sales")
ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig(DATA_DIR / "insight_dow_pattern.png", dpi=150)
plt.close()

print("Charts saved: insight_promo_lift.png, insight_seasonality.png,")
print("              insight_feature_importance.png, insight_dow_pattern.png")

# ── 9. SAVE TEXT REPORT ───────────────────────────────────────────────────────
report_path = DATA_DIR / "business_insights_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("RETAIL SALES FORECASTING — BUSINESS INSIGHTS REPORT\n")
    f.write(f"Generated from: {len(train):,} training rows\n")
    f.write(f"Date range: {train['date'].min().date()} → {train['date'].max().date()}\n")
    f.write("\n".join(report_lines))

print(f"\nFull report saved to {report_path}")

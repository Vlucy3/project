import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import base64

# --- CONFIG ---
st.set_page_config(page_title="Retail Sales Forecaster", layout="wide")

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

def engineer_features(train, test, stores, oil, holidays, transactions):
    # Simplified version of your existing engineering logic
    oil = oil.copy()
    oil["dcoilwtico"] = oil["dcoilwtico"].ffill().bfill()
    oil["oil_l7"] = oil["dcoilwtico"].rolling(7).mean().ffill().bfill()
    oil["oil_diff"] = oil["dcoilwtico"].diff().fillna(0)
    
    hols = holidays[holidays["transferred"] == False]
    nat_hols = hols[hols["locale"] == "National"].drop_duplicates("date")
    reg_hols = hols[hols["locale"] == "Regional"].drop_duplicates(["date", "locale_name"])
    loc_hols = hols[hols["locale"] == "Local"].drop_duplicates(["date", "locale_name"])

    transactions = transactions.copy()
    transactions["trans_lag_16"] = transactions.groupby("store_nbr")["transactions"].shift(16)

    combined = pd.concat([train, test], axis=0).sort_values(["store_nbr", "family", "date"])
    combined = combined.merge(stores, on="store_nbr", how="left")
    combined = combined.merge(oil, on="date", how="left")
    combined = combined.merge(transactions[["date", "store_nbr", "trans_lag_16"]], on=["date", "store_nbr"], how="left")

    combined["is_nat_holiday"] = combined["date"].isin(nat_hols["date"]).astype(int)
    
    # Matching regional/local (simplified for memory/speed in app)
    combined = combined.merge(reg_hols[["date", "locale_name"]].assign(is_reg_holiday=1), 
                             left_on=["date", "state"], right_on=["date", "locale_name"], how="left")
    combined["is_reg_holiday"] = combined["is_reg_holiday"].fillna(0).astype(int)
    
    combined = combined.merge(loc_hols[["date", "locale_name"]].assign(is_loc_holiday=1), 
                             left_on=["date", "city"], right_on=["date", "locale_name"], how="left")
    combined["is_loc_holiday"] = combined["is_loc_holiday"].fillna(0).astype(int)

    combined["day"] = combined["date"].dt.day
    combined["dow"] = combined["date"].dt.dayofweek
    combined["month"] = combined["date"].dt.month
    combined["week"] = combined["date"].dt.isocalendar().week.astype(int)
    combined["is_month_start"] = combined["date"].dt.is_month_start.astype(int)
    combined["is_month_end"] = combined["date"].dt.is_month_end.astype(int)
    
    combined["dist_payday"] = np.minimum(np.abs(combined["day"] - 15), 
                                        np.abs(combined["day"] - combined["date"].dt.days_in_month))
    combined["payday_signal"] = np.exp(-0.3 * combined["dist_payday"])
    combined["dow_sin"] = np.sin(2 * np.pi * combined["dow"] / 7)
    combined["dow_cos"] = np.cos(2 * np.pi * combined["dow"] / 7)

    # Target Encoding
    split_date = train["date"].max() - pd.Timedelta(days=16)
    train_part = combined[(combined["date"] <= split_date) & (combined["sales"].notna())]
    
    tf_mean = train_part.groupby(["store_nbr", "family"])["sales"].mean().reset_index().rename(columns={"sales": "store_family_mean"})
    f_mean = train_part.groupby(["family"])["sales"].mean().reset_index().rename(columns={"sales": "family_mean"})
    s_mean = train_part.groupby(["store_nbr"])["sales"].mean().reset_index().rename(columns={"sales": "store_mean"})
    
    combined = combined.merge(tf_mean, on=["store_nbr", "family"], how="left")
    combined = combined.merge(f_mean, on=["family"], how="left")
    combined = combined.merge(s_mean, on=["store_nbr"], how="left")

    for col in ["family", "city", "state", "type", "cluster"]:
        combined[col] = combined[col].astype("category").cat.codes

    for l in [16, 17, 18, 21, 28, 35]:
        combined[f"lag_{l}"] = combined.groupby(["store_nbr", "family"])["sales"].shift(l)
    
    for w in [3, 7, 14, 28]:
        combined[f"roll_{w}"] = combined.groupby(["store_nbr", "family"])["lag_16"].transform(lambda x: x.rolling(w).mean())

    combined["next_promo_7"] = combined.groupby(["store_nbr", "family"])["onpromotion"].transform(lambda x: x.shift(-7).rolling(7).mean())
    combined["next_promo_14"] = combined.groupby(["store_nbr", "family"])["onpromotion"].transform(lambda x: x.shift(-14).rolling(14).mean())

    feat_cols = [c for c in combined.columns if c != "sales"]
    combined[feat_cols] = combined[feat_cols].fillna(0)
    
    return combined, split_date

def main():
    st.title("🛍️ Retail Sales Forecasting Dashboard")
    st.markdown("""
    This application uses a **Super Optimized Ensemble (XGBoost + LightGBM)** to predict supermarket sales.
    Upload your datasets to begin.
    """)

    # --- SIDEBAR ---
    st.sidebar.header("Data Upload")
    train_file = st.sidebar.file_uploader("train.csv", type="csv")
    test_file = st.sidebar.file_uploader("test.csv", type="csv")
    stores_file = st.sidebar.file_uploader("stores.csv", type="csv")
    oil_file = st.sidebar.file_uploader("oil.csv", type="csv")
    hols_file = st.sidebar.file_uploader("holidays_events.csv", type="csv")
    trans_file = st.sidebar.file_uploader("transactions.csv", type="csv")

    if all([train_file, test_file, stores_file, oil_file, hols_file, trans_file]):
        if st.sidebar.button("🚀 Run Forecast"):
            with st.spinner("Processing data and training models..."):
                # Load
                train = pd.read_csv(train_file, parse_dates=["date"])
                test = pd.read_csv(test_file, parse_dates=["date"])
                stores = pd.read_csv(stores_file)
                oil = pd.read_csv(oil_file, parse_dates=["date"])
                holidays = pd.read_csv(hols_file, parse_dates=["date"])
                transactions = pd.read_csv(trans_file, parse_dates=["date"])

                # Pre-process (2017+ for speed in UI)
                train = train[train["date"] >= "2017-01-01"].copy()
                
                # Engineer
                combined, split_date = engineer_features(train, test, stores, oil, holidays, transactions)
                
                train_final = combined[combined["sales"].notna()].copy()
                test_final = combined[combined["sales"].isna()].copy()
                train_final["log_sales"] = np.log1p(train_final["sales"])
                
                X_train = train_final[train_final["date"] <= split_date]
                X_val = train_final[train_final["date"] > split_date]
                y_train = X_train["log_sales"]
                y_val_log = X_val["log_sales"]
                y_val_true = X_val["sales"]

                # Train
                st.write("### Model Training")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Training XGBoost...")
                    xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, n_jobs=-1)
                    xgb.fit(X_train[FEATURES], y_train, eval_set=[(X_val[FEATURES], y_val_log)], verbose=False)
                    st.success("XGBoost Ready!")
                
                with col2:
                    st.write("Training LightGBM...")
                    lgb = LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=128, n_jobs=-1, verbose=-1)
                    lgb.fit(X_train[FEATURES], y_train, eval_set=[(X_val[FEATURES], y_val_log)], callbacks=[])
                    st.success("LightGBM Ready!")

                # Evaluate
                xgb_val = np.expm1(xgb.predict(X_val[FEATURES]))
                lgb_val = np.expm1(lgb.predict(X_val[FEATURES]))
                final_val = np.maximum(0, 0.4 * xgb_val + 0.6 * lgb_val)
                
                rmsle = np.sqrt(np.mean((np.log1p(final_val) - np.log1p(y_val_true)) ** 2))
                r2 = r2_score(y_val_true, final_val)

                # Results
                st.write("### Performance Metrics")
                m1, m2 = st.columns(2)
                m1.metric("RMSLE", f"{rmsle:.5f}")
                m2.metric("R² Score", f"{r2:.4f}")

                # Visualization
                st.write("### Forecast Visualization (Sample: Store 1)")
                fig, ax = plt.subplots(figsize=(10, 4))
                
                # Filter for sample viz
                sample_val = X_val[X_val["store_nbr"] == 1].copy()
                sample_val["pred"] = final_val[X_val["store_nbr"] == 1]
                
                sample_val = sample_val.groupby("date")[["sales", "pred"]].sum().reset_index()
                ax.plot(sample_val["date"], sample_val["sales"], label="Actual", marker='o')
                ax.plot(sample_val["date"], sample_val["pred"], label="Forecast", marker='x', linestyle='--')
                ax.set_title("Store 1 - Aggregate Daily Sales Forecast")
                ax.legend()
                st.pyplot(fig)

                # Predictions
                st.write("### Final Predictions")
                xgb_test = np.expm1(xgb.predict(test_final[FEATURES]))
                lgb_test = np.expm1(lgb.predict(test_final[FEATURES]))
                test_preds = np.maximum(0, 0.4 * xgb_test + 0.6 * lgb_test)
                
                res_df = test.copy()
                res_df["sales"] = test_preds
                st.dataframe(res_df.head(10))

                # Download
                csv = res_df[["id", "sales"]].to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="forecast_results.csv">Download Prediction CSV</a>'
                st.markdown(href, unsafe_allow_html=True)

    else:
        st.info("Please upload all 6 required CSV files in the sidebar to run the forecaster.")
        st.image("https://www.streamlit.io/images/brand/streamlit-logo-primary-colormark-darktext.png", width=200)

if __name__ == "__main__":
    main()

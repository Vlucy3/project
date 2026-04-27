"""Shared data loading and feature engineering for all forecasting scripts."""

import pandas as pd
import numpy as np
from pathlib import Path


def load_data(data_dir: Path):
    """Load all CSVs. Returns (train, test, stores, oil, holidays)."""
    train    = pd.read_csv(data_dir / "train.csv",           parse_dates=["date"])
    test     = pd.read_csv(data_dir / "test.csv",            parse_dates=["date"])
    stores   = pd.read_csv(data_dir / "stores.csv")
    oil      = pd.read_csv(data_dir / "oil.csv",             parse_dates=["date"])
    holidays = pd.read_csv(data_dir / "holidays_events.csv", parse_dates=["date"])
    return train, test, stores, oil, holidays


def prep_oil(oil: pd.DataFrame) -> pd.DataFrame:
    """Fill gaps and add 7-day rolling mean."""
    oil = oil.copy()
    oil["dcoilwtico"] = oil["dcoilwtico"].ffill().bfill()
    oil["oil_l7"]     = oil["dcoilwtico"].rolling(7).mean().ffill().bfill()
    return oil


def get_split_date(train: pd.DataFrame, val_days: int = 16) -> pd.Timestamp:
    return train["date"].max() - pd.Timedelta(days=val_days)


def make_target_encoding(train: pd.DataFrame, split_date: pd.Timestamp) -> pd.DataFrame:
    """Store x family mean sales computed on training portion only (no leakage)."""
    return (
        train[train["date"] <= split_date]
        .groupby(["store_nbr", "family"])["sales"]
        .mean().reset_index()
        .rename(columns={"sales": "store_family_mean"})
    )


def get_nat_hols(holidays: pd.DataFrame) -> pd.DataFrame:
    return (
        holidays[(holidays["transferred"] == False) & (holidays["locale"] == "National")]
        .drop_duplicates("date")[["date"]]
    )


def engineer_features(df: pd.DataFrame, stores: pd.DataFrame, oil: pd.DataFrame,
                      target_mean: pd.DataFrame, nat_hols: pd.DataFrame,
                      holidays: pd.DataFrame = None) -> pd.DataFrame:
    """
    Merge auxiliary tables and add time/calendar/payday/holiday features.

    Pass the full holidays DataFrame to also add is_reg_holiday and
    is_loc_holiday (matched on each store's state and city respectively).
    Omitting it keeps the original national-holiday-only behaviour.
    """
    df = df.copy()
    df = df.merge(stores,      on="store_nbr", how="left")
    df = df.merge(oil,         on="date",      how="left")
    df = df.merge(target_mean, on=["store_nbr", "family"], how="left").fillna(0)

    df["is_nat_holiday"] = df["date"].isin(nat_hols["date"]).astype(int)

    # Regional holidays: match on date + store's state
    # Local holidays:    match on date + store's city
    # Must be done before label-encoding wipes the string columns.
    if holidays is not None:
        hols = holidays[holidays["transferred"] == False]

        reg = (
            hols[hols["locale"] == "Regional"][["date", "locale_name"]]
            .drop_duplicates()
            .rename(columns={"locale_name": "state"})
            .assign(is_reg_holiday=1)
        )
        df = df.merge(reg, on=["date", "state"], how="left")
        df["is_reg_holiday"] = df["is_reg_holiday"].fillna(0).astype(int)

        loc = (
            hols[hols["locale"] == "Local"][["date", "locale_name"]]
            .drop_duplicates()
            .rename(columns={"locale_name": "city"})
            .assign(is_loc_holiday=1)
        )
        df = df.merge(loc, on=["date", "city"], how="left")
        df["is_loc_holiday"] = df["is_loc_holiday"].fillna(0).astype(int)

    df["day"]            = df["date"].dt.day
    df["dow"]            = df["date"].dt.dayofweek
    df["month"]          = df["date"].dt.month
    df["dist_payday"]    = np.minimum(
        np.abs(df["day"] - 15),
        np.abs(df["day"] - df["date"].dt.days_in_month)
    )
    df["payday_signal"]  = np.exp(-0.3 * df["dist_payday"])
    df["dow_sin"]        = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"]        = np.cos(2 * np.pi * df["dow"] / 7)

    for col in ["family", "city", "state", "type", "cluster"]:
        df[col] = df[col].astype("category").cat.codes

    return df


def add_lags(df: pd.DataFrame, lags: list, rolls: list,
             promo_horizon: int = 7) -> pd.DataFrame:
    """
    Add per-(store, family) lag features, rolling means of the smallest lag,
    and a forward promotion signal. Expects df already sorted by
    [store_nbr, family, date].
    """
    df      = df.copy()
    base    = f"lag_{min(lags)}"

    for lag in lags:
        df[f"lag_{lag}"] = df.groupby(["store_nbr", "family"])["sales"].shift(lag)

    for w in rolls:
        df[f"roll_{w}"] = (
            df.groupby(["store_nbr", "family"])[base]
            .transform(lambda x: x.rolling(w).mean())
        )

    df["next_promo_7"] = (
        df.groupby(["store_nbr", "family"])["onpromotion"]
        .transform(lambda x: x.shift(-promo_horizon).rolling(promo_horizon).mean())
    )

    return df

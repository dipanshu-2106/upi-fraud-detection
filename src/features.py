"""Stage 2: Feature engineering for UPI transactions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
EPOCH = pd.Timestamp("1970-01-01")
FIRST_TXN_GAP = 86400.0  # seconds; used when a user has no earlier transaction
FEATURE_COLS = [
    "amount",
    "hour",
    "amount_zscore_vs_user",
    "txn_count_last_hour",
    "is_new_device",
    "is_new_city",
    "time_since_last_txn",
]
def load_transactions():
    """Read transactions.csv with the timestamp parsed as datetime."""
    return pd.read_csv(DATA_DIR / "transactions.csv", parse_dates=["timestamp"])
def build_user_profiles(df):
    """Normal behaviour of every user: amount mean/std, usual device, home city."""
    g = df.groupby("user_id")
    profiles = pd.DataFrame({
        "amount_mean": g["amount"].mean(),
        "amount_std": g["amount"].std(),
        "usual_device": g["device_id"].agg(lambda s: s.mode().iloc[0]),
        "home_city": g["city"].agg(lambda s: s.mode().iloc[0]),
    })
    # avoid divide-by-zero for users with 1 transaction or no spread
    profiles["amount_std"] = profiles["amount_std"].fillna(1.0).clip(lower=1.0)
    return profiles
def save_profiles(profiles):
    """Save profiles as JSON so the API can load them later."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / "user_profiles.json"
    with open(path, "w") as f:
        json.dump(profiles.to_dict(orient="index"), f)
    return path
def compute_features(df, profiles):
    """Return a DataFrame with the 7 model features (same row order as df)."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # attach each row's user profile (aligned to df rows)
    p = profiles.reindex(df["user_id"].values)
    p.index = df.index
    feats = pd.DataFrame(index=df.index)
    feats["amount"] = df["amount"]
    feats["hour"] = df["timestamp"].dt.hour
    feats["amount_zscore_vs_user"] = (df["amount"] - p["amount_mean"]) / p["amount_std"]
    feats["is_new_device"] = (df["device_id"] != p["usual_device"]).astype(int)
    feats["is_new_city"] = (df["city"] != p["home_city"]).astype(int)
    feats["txn_count_last_hour"] = 0
    feats["time_since_last_txn"] = FIRST_TXN_GAP
    # per-user time based features
    ordered = df.sort_values(["user_id", "timestamp"], kind="stable")
    for _, grp in ordered.groupby("user_id", sort=False):
        secs = (grp["timestamp"] - EPOCH).dt.total_seconds().to_numpy()
        n = len(secs)
        # transactions of this user in the last 60 minutes (including this one)
        start = np.searchsorted(secs, secs - 3600, side="right")
        counts = np.arange(n) - start + 1
        # seconds since this user's previous transaction
        gaps = np.empty(n)
        gaps[0] = FIRST_TXN_GAP
        gaps[1:] = np.diff(secs)
        feats.loc[grp.index, "txn_count_last_hour"] = counts
        feats.loc[grp.index, "time_since_last_txn"] = gaps
    return feats[FEATURE_COLS]
def main():
    df = load_transactions()
    profiles = build_user_profiles(df)
    path = save_profiles(profiles)
    feats = compute_features(df, profiles)
    feats.to_csv(DATA_DIR / "features.csv", index=False)
    print(f"Saved features to {DATA_DIR / 'features.csv'}")
    print(f"Saved user profiles to {path}")
    print(feats.head())
    print(feats.describe().round(2).T)
if __name__ == "__main__":
    main()

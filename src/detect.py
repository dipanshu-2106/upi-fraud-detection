"""Stage 3: Three anomaly detection methods + ensemble."""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from src.features import (
    DATA_DIR,
    FEATURE_COLS,
    MODELS_DIR,
    build_user_profiles,
    compute_features,
    load_transactions,
    save_profiles,
)
def flag_iqr(df):
    """Method 1: flag if amount > Q3 + 1.5 * IQR of the same user."""
    q = df.groupby("user_id")["amount"].quantile([0.25, 0.75]).unstack()
    q.columns = ["q1", "q3"]
    upper = q["q3"] + 1.5 * (q["q3"] - q["q1"])
    limit = df["user_id"].map(upper)
    return (df["amount"] > limit).astype(int)
def run_isolation_forest(X):
    """Method 2: Isolation Forest on engineered features (no labels used)."""
    model = IsolationForest(n_estimators=200, contamination=0.02, random_state=42)
    model.fit(X)
    flags = (model.predict(X) == -1).astype(int)   # -1 means anomaly
    scores = -model.score_samples(X)               # higher = more suspicious
    return model, flags, scores
def flag_rolling_zscore(df, window=10, threshold=3.0):
    """Method 3: compare amount with the user's rolling mean/std of past spending."""
    d = df[["user_id", "timestamp", "amount"]].sort_values(
        ["user_id", "timestamp"], kind="stable"
    )
    grp = d.groupby("user_id")["amount"]
    # shift(1) so the current transaction is not part of its own baseline
    prev_mean = grp.transform(lambda s: s.shift(1).rolling(window, min_periods=3).mean())
    prev_std = grp.transform(lambda s: s.shift(1).rolling(window, min_periods=3).std())
    z = (d["amount"] - prev_mean) / prev_std.replace(0, np.nan)
    flags = (z > threshold).astype(int)
    return flags.reindex(df.index)
def main():
    df = load_transactions()
    profiles = build_user_profiles(df)
    save_profiles(profiles)
    X = compute_features(df, profiles)   # is_fraud is NOT used anywhere here
    res = df.copy()
    res["flag_iqr"] = flag_iqr(df)
    model, flags, scores = run_isolation_forest(X)
    res["flag_iforest"] = flags
    res["iforest_score"] = scores.round(4)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "feature_cols": FEATURE_COLS},
        MODELS_DIR / "isolation_forest.joblib",
    )
    res["flag_timeseries"] = flag_rolling_zscore(df)
    # Ensemble: flag if at least 2 of the 3 methods flag it
    votes = res[["flag_iqr", "flag_iforest", "flag_timeseries"]].sum(axis=1)
    res["flag_ensemble"] = (votes >= 2).astype(int)
    res.to_csv(DATA_DIR / "results.csv", index=False)
    print(f"Saved results to {DATA_DIR / 'results.csv'}")
    print(f"Saved model to {MODELS_DIR / 'isolation_forest.joblib'}")
    print("\nTransactions flagged by each method (out of %d):" % len(res))
    for col in ["flag_iqr", "flag_iforest", "flag_timeseries", "flag_ensemble"]:
        print(f"  {col:16s}: {int(res[col].sum())}")
if __name__ == "__main__":
    main()

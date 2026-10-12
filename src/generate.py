"""Stage 1: Generate synthetic UPI transactions."""
from pathlib import Path
import numpy as np
import pandas as pd
SEED = 42
N_TXN = 10_000
N_USERS = 500
N_DAYS = 30
FRAUD_RATE = 0.02
N_BURSTS = 8
START_DATE = pd.Timestamp("2025-01-01")
CITIES = [
    "Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Kolkata",
    "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Bhopal", "Indore",
]
TXN_TYPES = ["P2P", "P2M", "BILL", "RECHARGE"]
TXN_PROBS = [0.35, 0.45, 0.12, 0.08]
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
def build_users(rng):
    """Create a normal behaviour profile for every user."""
    users = []
    for i in range(N_USERS):
        start = int(rng.integers(7, 13))      # active from 7..12
        length = int(rng.integers(10, 14))    # active for 10..13 hours
        users.append({
            "user_id": f"U{i:04d}",
            "mu": float(rng.normal(6.0, 0.6)),        # log-normal mean (exp(6) ~ Rs 400)
            "sigma": float(rng.uniform(0.4, 0.8)),    # log-normal spread
            "active_start": start,
            "active_end": min(start + length, 23),
            "home_city": str(rng.choice(CITIES)),
            "device_id": f"D{i:04d}",
            "receivers": [f"R{int(r):04d}" for r in rng.integers(0, 300, size=6)],
        })
    return users
def other_city(rng, home):
    """Pick any city that is not the home city."""
    return str(rng.choice([c for c in CITIES if c != home]))
def make_row(user, ts, amount, receiver, device, city, txn_type, is_fraud):
    """Build one transaction as a dictionary."""
    return {
        "user_id": user["user_id"],
        "amount": round(float(amount), 2),
        "timestamp": ts,
        "receiver_id": receiver,
        "device_id": device,
        "city": city,
        "txn_type": txn_type,
        "is_fraud": int(is_fraud),
    }
def normal_rows(rng, users, n):
    """Normal transactions that follow each user's own pattern."""
    rows = []
    for _ in range(n):
        u = users[int(rng.integers(0, N_USERS))]
        hour = int(rng.integers(u["active_start"], u["active_end"] + 1))
        ts = START_DATE + pd.Timedelta(
            days=int(rng.integers(0, N_DAYS)),
            hours=hour,
            minutes=int(rng.integers(0, 60)),
            seconds=int(rng.integers(0, 60)),
        )
        amount = np.clip(rng.lognormal(u["mu"], u["sigma"]), 1, 200000)
        if rng.random() < 0.8:
            receiver = str(rng.choice(u["receivers"]))
        else:
            receiver = f"R{int(rng.integers(0, 300)):04d}"
        # about 1% of normal transactions happen while travelling
        city = other_city(rng, u["home_city"]) if rng.random() < 0.01 else u["home_city"]
        txn_type = str(rng.choice(TXN_TYPES, p=TXN_PROBS))
        rows.append(make_row(u, ts, amount, receiver, u["device_id"], city, txn_type, 0))
    return rows
def burst_rows(rng, users):
    """Fraud bursts: 5-7 quick transfers inside one minute from a new device."""
    rows = []
    for b in range(N_BURSTS):
        u = users[int(rng.integers(0, N_USERS))]
        base = START_DATE + pd.Timedelta(
            days=int(rng.integers(0, N_DAYS)),
            hours=int(rng.integers(0, 24)),
            minutes=int(rng.integers(0, 60)),
        )
        size = int(rng.integers(5, 8))
        device = f"D_NEW_B{b:02d}"
        city = other_city(rng, u["home_city"])
        offsets = np.sort(rng.integers(0, 55, size=size))
        for off in offsets:
            ts = base + pd.Timedelta(seconds=int(off))
            amount = np.exp(u["mu"]) * rng.uniform(2, 6)
            receiver = f"R{int(rng.integers(300, 400)):04d}"
            rows.append(make_row(u, ts, amount, receiver, device, city, "P2P", 1))
    return rows
def single_fraud_rows(rng, users, n):
    """Single fraud transactions with a mix of 2-3 suspicious signals."""
    rows = []
    for k in range(n):
        u = users[int(rng.integers(0, N_USERS))]
        # signals: 0 = big amount, 1 = odd hour, 2 = new device, 3 = new city
        signals = set(rng.choice(4, size=int(rng.integers(2, 4)), replace=False).tolist())
        if 0 in signals:
            amount = np.exp(u["mu"]) * rng.uniform(8, 25)
        else:
            amount = np.clip(rng.lognormal(u["mu"], u["sigma"]), 1, 200000)
        if 1 in signals:
            hour = int(rng.integers(2, 5))  # 2, 3 or 4 AM
        else:
            hour = int(rng.integers(u["active_start"], u["active_end"] + 1))
        ts = START_DATE + pd.Timedelta(
            days=int(rng.integers(0, N_DAYS)),
            hours=hour,
            minutes=int(rng.integers(0, 60)),
            seconds=int(rng.integers(0, 60)),
        )
        device = f"D_NEW_{k:03d}" if 2 in signals else u["device_id"]
        city = other_city(rng, u["home_city"]) if 3 in signals else u["home_city"]
        receiver = f"R{int(rng.integers(300, 400)):04d}"
        rows.append(make_row(u, ts, amount, receiver, device, city, "P2P", 1))
    return rows
def generate():
    """Build the full dataset and return it as a DataFrame."""
    rng = np.random.default_rng(SEED)
    users = build_users(rng)
    n_fraud = int(N_TXN * FRAUD_RATE)          # 200
    bursts = burst_rows(rng, users)
    n_single = n_fraud - len(bursts)
    singles = single_fraud_rows(rng, users, n_single)
    normals = normal_rows(rng, users, N_TXN - n_fraud)
    df = pd.DataFrame(normals + singles + bursts)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.insert(0, "txn_id", [f"T{i + 1:06d}" for i in range(len(df))])
    return df
def main():
    df = generate()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "transactions.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} transactions to {out_path}")
    print(f"Users: {df['user_id'].nunique()} | Fraud: {df['is_fraud'].sum()} "
          f"({df['is_fraud'].mean() * 100:.2f}%)")
    print(df.head())
if __name__ == "__main__":
    main()

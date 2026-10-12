"""Stage 6: send live fake UPI transactions to the API."""
import json
import os
import random
import time
from pathlib import Path
import requests
ROOT = Path(__file__).resolve().parent
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
CITIES = [
    "Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Kolkata",
    "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Bhopal", "Indore",
]
TXN_TYPES = ["P2P", "P2M", "BILL", "RECHARGE"]
SUSPICIOUS_RATE = 0.10  # about 10% of transactions look suspicious
def load_profiles():
    """Load user profiles saved by the feature stage."""
    path = ROOT / "models" / "user_profiles.json"
    if not path.exists():
        raise SystemExit("models/user_profiles.json not found. Run 'python run_all.py' first.")
    with open(path) as f:
        return json.load(f)
def normal_txn(user_id, p):
    """A transaction that follows the user's normal pattern."""
    return {
        "user_id": user_id,
        "amount": round(max(10.0, p["amount_mean"] * random.uniform(0.3, 1.5)), 2),
        "receiver_id": f"R{random.randint(0, 299):04d}",
        "device_id": p["usual_device"],
        "city": p["home_city"],
        "txn_type": random.choice(TXN_TYPES),
    }
def suspicious_txn(user_id, p):
    """A transaction with 2-3 suspicious signals."""
    txn = normal_txn(user_id, p)
    signals = random.sample(["amount", "device", "city"], k=random.choice([2, 3]))
    if "amount" in signals:
        txn["amount"] = round(p["amount_mean"] * random.uniform(8, 25), 2)
    if "device" in signals:
        txn["device_id"] = f"D_NEW_{random.randint(100, 999)}"
    if "city" in signals:
        txn["city"] = random.choice([c for c in CITIES if c != p["home_city"]])
    txn["receiver_id"] = f"R{random.randint(300, 399):04d}"
    txn["txn_type"] = "P2P"
    return txn
def main():
    profiles = load_profiles()
    users = sorted(profiles.keys())
    sent = 0
    flagged = 0
    print(f"Sending live transactions to {API_URL} (press Ctrl+C to stop)\n")
    try:
        while True:
            user_id = random.choice(users)
            p = profiles[user_id]
            is_sus = random.random() < SUSPICIOUS_RATE
            txn = suspicious_txn(user_id, p) if is_sus else normal_txn(user_id, p)
            try:
                resp = requests.post(f"{API_URL}/check", json=txn, timeout=5)
            except requests.RequestException:
                print("Cannot reach the API. Is 'uvicorn api.main:app --port 8000' running? Retrying...")
                time.sleep(3)
                continue
            if resp.status_code != 200:
                print(f"API error {resp.status_code}: {resp.text}")
            else:
                out = resp.json()
                sent += 1
                flagged += int(out["fraud"])
                verdict = "FRAUD " if out["fraud"] else "OK    "
                sent_as = "suspicious" if is_sus else "normal"
                reasons = ", ".join(out["reasons"]) if out["reasons"] else "-"
                print(f"{verdict}| sent as {sent_as:10s} | {user_id} | Rs {txn['amount']:>10.2f} "
                      f"| {txn['city']:10s} | score {out['score']:.3f} | {reasons}")
            time.sleep(random.uniform(1, 2))
    except KeyboardInterrupt:
        print(f"\nStopped. Sent {sent} transactions, {flagged} flagged as fraud.")
if __name__ == "__main__":
    main()

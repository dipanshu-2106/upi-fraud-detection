"""Optional background demo traffic: the API sends fake transactions to itself."""
import random
import threading
import time
import requests
from simulate import load_profiles, normal_txn, suspicious_txn
def run(base_url):
    time.sleep(8)   # wait until the server is up
    try:
        profiles = load_profiles()
    except SystemExit:
        print("[demo] user_profiles.json not found - demo traffic is off")
        return
    users = sorted(profiles)
    print("[demo] background demo traffic started")
    while True:
        try:
            uid = random.choice(users)
            p = profiles[uid]
            txn = suspicious_txn(uid, p) if random.random() < 0.10 else normal_txn(uid, p)
            requests.post(base_url + "/check", json=txn, timeout=10)
        except requests.RequestException:
            time.sleep(3)
        except Exception as exc:
            print(f"[demo] error: {exc}")
            time.sleep(5)
        time.sleep(random.uniform(2, 4))
def start(base_url):
    """Start the demo traffic in a background thread."""
    threading.Thread(target=run, args=(base_url,), daemon=True).start()

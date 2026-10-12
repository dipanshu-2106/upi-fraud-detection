"""Stage 5: FastAPI service for real-time UPI fraud checks (risk levels + alerts)."""
import json
import os
import sys
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import joblib
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.features import DATA_DIR, FIRST_TXN_GAP, MODELS_DIR  # noqa: E402
# risk level rules (score is "higher = more suspicious")
RISK_MARGIN_HIGH = 0.10     # fraud + score >= threshold + 0.10  -> HIGH
RISK_MARGIN_MEDIUM = 0.05   # score >= threshold - 0.05          -> MEDIUM (borderline)
ALERT_LOG = DATA_DIR / "alerts.log"
# ---- in-memory state ----
STATE = {
    "model": None,
    "feature_cols": None,
    "profiles": {},
    "default_mean": 600.0,
    "default_std": 900.0,
    "threshold": 0.5,
}
HISTORY = {}                  # user_id -> deque of (timestamp, amount)
RECENT = deque(maxlen=200)    # last 200 results
ALERTS = deque(maxlen=50)     # last 50 HIGH risk alerts
COUNTER = {"n": 0, "alerts": 0}
TELEGRAM = {"last_sent": 0.0}
LOCK = threading.Lock()
@asynccontextmanager
async def lifespan(app):
    """Load model and user profiles once when the server starts."""
    model_path = MODELS_DIR / "isolation_forest.joblib"
    profile_path = MODELS_DIR / "user_profiles.json"
    if not model_path.exists() or not profile_path.exists():
        raise RuntimeError("Model files not found. Run 'python run_all.py' first.")
    bundle = joblib.load(model_path)
    STATE["model"] = bundle["model"]
    STATE["feature_cols"] = bundle["feature_cols"]
    # score above this value means the model flags the transaction as fraud
    STATE["threshold"] = float(-bundle["model"].offset_)
    with open(profile_path) as f:
        profiles = json.load(f)
    STATE["profiles"] = profiles
    if profiles:
        n = len(profiles)
        STATE["default_mean"] = sum(p["amount_mean"] for p in profiles.values()) / n
        STATE["default_std"] = sum(p["amount_std"] for p in profiles.values()) / n
    if os.environ.get("DEMO_SIMULATOR", "").lower() == "on":
        from api.demo_sim import start as start_demo
        start_demo("http://127.0.0.1:" + os.environ.get("PORT", "8000"))
    yield
app = FastAPI(title="UPI Fraud Detection API", version="2.0", lifespan=lifespan)
class Transaction(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=50)
    amount: float = Field(..., gt=0, le=10_000_000)
    receiver_id: str = Field(..., min_length=1, max_length=50)
    device_id: str = Field(..., min_length=1, max_length=50)
    city: str = Field(..., min_length=1, max_length=50)
    txn_type: str = Field("P2P", min_length=1, max_length=20)
    timestamp: Optional[datetime] = None  # if missing, current time is used
@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    """Return a short, clear message for bad input."""
    details = []
    for err in exc.errors():
        loc = [str(x) for x in err.get("loc", []) if x != "body"]
        details.append({
            "field": ".".join(loc) or "body",
            "message": str(err.get("msg", "invalid value")),
        })
    return JSONResponse(status_code=422, content={"error": "Invalid input", "details": details})
def build_features(txn, ts):
    """Make the same 7 features as training, using live history."""
    prof = STATE["profiles"].get(txn.user_id)
    if prof:
        mean = prof["amount_mean"]
        std = max(prof["amount_std"], 1.0)
        new_device = int(txn.device_id != prof["usual_device"])
        new_city = int(txn.city != prof["home_city"])
    else:
        # unknown user: no profile, so we cannot say device/city is new
        mean = STATE["default_mean"]
        std = max(STATE["default_std"], 1.0)
        new_device = 0
        new_city = 0
    hist = HISTORY.get(txn.user_id, [])
    window_start = ts - timedelta(hours=1)
    count = 1 + sum(1 for t, _ in hist if window_start < t <= ts)
    earlier = [t for t, _ in hist if t <= ts]
    gap = (ts - max(earlier)).total_seconds() if earlier else FIRST_TXN_GAP
    return {
        "amount": txn.amount,
        "hour": ts.hour,
        "amount_zscore_vs_user": (txn.amount - mean) / std,
        "txn_count_last_hour": count,
        "is_new_device": new_device,
        "is_new_city": new_city,
        "time_since_last_txn": gap,
    }
def build_reasons(f):
    """Plain-language reasons for suspicious signals."""
    reasons = []
    if f["amount_zscore_vs_user"] > 3:
        reasons.append("amount much higher than usual")
    if 0 <= f["hour"] <= 5:
        reasons.append("odd hour (late night)")
    if f["is_new_device"]:
        reasons.append("new device")
    if f["is_new_city"]:
        reasons.append("new city")
    if f["txn_count_last_hour"] >= 4:
        reasons.append("too many transactions in the last hour")
    return reasons
def risk_level(is_fraud, score):
    """LOW / MEDIUM / HIGH based on the model score and its fraud threshold."""
    thr = STATE["threshold"]
    if is_fraud and score >= thr + RISK_MARGIN_HIGH:
        return "HIGH"
    if is_fraud or score >= thr - RISK_MARGIN_MEDIUM:
        return "MEDIUM"
    return "LOW"
def alert_text(e):
    """Readable alert message."""
    reasons = ", ".join(e["reasons"]) if e["reasons"] else "-"
    return (
        "HIGH RISK UPI transaction\n"
        f"User: {e['user_id']}\n"
        f"Amount: Rs {e['amount']:,.2f}\n"
        f"City: {e['city']}\n"
        f"Device: {e['device_id']}\n"
        f"Score: {e['score']}\n"
        f"Reasons: {reasons}\n"
        f"Time: {e['time']}"
    )
def send_telegram(token, chat_id, text):
    """Send one Telegram message (runs in a background thread)."""
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": text},
            timeout=5,
        )
        if r.status_code != 200:
            print(f"[alert] Telegram error {r.status_code}: {r.text[:150]}")
    except requests.RequestException as exc:
        print(f"[alert] Telegram not reachable: {exc}")
def send_alert(entry):
    """Alert channels: memory, log file, and optional Telegram."""
    text = alert_text(entry)
    with LOCK:
        ALERTS.append(entry)
        COUNTER["alerts"] += 1
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(ALERT_LOG, "a", encoding="utf-8") as f:
            f.write(text.replace("\n", " | ") + "\n")
    except OSError:
        pass
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    now = time.time()
    if now - TELEGRAM["last_sent"] < 2:   # small gap so Telegram does not block us
        return
    TELEGRAM["last_sent"] = now
    threading.Thread(target=send_telegram, args=(token, chat_id, text), daemon=True).start()
@app.get("/")
def root():
    return {
        "message": "UPI Fraud Detection API is running",
        "docs": "/docs",
        "health": "/health",
        "recent": "/recent",
        "alerts": "/alerts",
    }
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": STATE["model"] is not None,
        "threshold": round(STATE["threshold"], 4),
        "users_with_history": len(HISTORY),
        "transactions_checked": COUNTER["n"],
        "alerts_raised": COUNTER["alerts"],
        "telegram_enabled": bool(
            os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID")
        ),
    }
@app.post("/check")
def check(txn: Transaction):
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    ts = txn.timestamp or datetime.now()
    if ts.tzinfo is not None:
        ts = ts.replace(tzinfo=None)
    with LOCK:
        feats = build_features(txn, ts)
        X = pd.DataFrame([feats], columns=STATE["feature_cols"])
        try:
            model = STATE["model"]
            is_fraud = bool(model.predict(X)[0] == -1)
            score = float(-model.score_samples(X)[0])   # higher = more suspicious
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Model error: {exc}")
        risk = risk_level(is_fraud, score)
        reasons = build_reasons(feats)
        if is_fraud and not reasons:
            reasons.append("unusual mix of signals (flagged by model)")
        # update live history for this user
        HISTORY.setdefault(txn.user_id, deque(maxlen=100)).append((ts, txn.amount))
        COUNTER["n"] += 1
        entry = {
            "txn_id": f"LIVE{COUNTER['n']:06d}",
            "time": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "user_id": txn.user_id,
            "amount": round(txn.amount, 2),
            "city": txn.city,
            "device_id": txn.device_id,
            "receiver_id": txn.receiver_id,
            "txn_type": txn.txn_type,
            "fraud": is_fraud,
            "score": round(score, 4),
            "risk": risk,
            "reasons": reasons,
        }
        RECENT.append(entry)
    if risk == "HIGH":
        send_alert(entry)
    return {"fraud": is_fraud, "score": round(score, 4), "risk": risk, "reasons": reasons}
@app.get("/recent")
def recent(limit: int = Query(200, ge=1, le=200)):
    """Last results, newest first."""
    with LOCK:
        items = list(RECENT)
    return list(reversed(items))[:limit]
@app.get("/alerts")
def alerts(limit: int = Query(50, ge=1, le=50)):
    """Last HIGH risk alerts, newest first."""
    with LOCK:
        items = list(ALERTS)
    return list(reversed(items))[:limit]

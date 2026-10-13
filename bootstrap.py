"""Starts everything the dashboard needs when no API is running (for Streamlit Community Cloud)."""
import os
import sys
import threading
import time
from pathlib import Path
import requests
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
API_PORT = 8000
def api_alive(api_url, timeout=1.5):
    """True if an API answers on api_url."""
    try:
        return requests.get(api_url + "/health", timeout=timeout).status_code == 200
    except requests.RequestException:
        return False
def ensure_files():
    """Create data, model and charts if they do not exist yet."""
    needed = [
        ROOT / "models" / "isolation_forest.joblib",
        ROOT / "models" / "user_profiles.json",
        ROOT / "data" / "comparison.csv",
    ]
    if all(p.exists() for p in needed):
        return
    from src import detect, evaluate, generate
    generate.main()
    detect.main()
    evaluate.main()
def ensure_running(api_url):
    """Use a running API if there is one, else start one inside this process."""
    if api_alive(api_url):
        return "external"
    # only start an embedded API when the dashboard points to this machine
    if "127.0.0.1" not in api_url and "localhost" not in api_url:
        return "remote-offline"
    ensure_files()
    os.environ["DEMO_SIMULATOR"] = "on"
    os.environ["PORT"] = str(API_PORT)
    import uvicorn
    from api.main import app as fastapi_app
    config = uvicorn.Config(fastapi_app, host="127.0.0.1", port=API_PORT, log_level="warning")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(90):
        if api_alive(api_url, timeout=1):
            return "embedded"
        time.sleep(1)
    return "embedded-not-ready"

"""Stage 6: Streamlit dashboard - risk levels, filters, trend charts, alerts."""
import html
import os
from pathlib import Path
import pandas as pd
import requests
import streamlit as st
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CHARTS_DIR = DATA_DIR / "charts"
CITIES = [
    "Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Kolkata",
    "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Bhopal", "Indore",
]
RISK_ORDER = ["LOW", "MEDIUM", "HIGH"]
RISK_COLORS = {"LOW": "#2ecc71", "MEDIUM": "#f39c12", "HIGH": "#e74c3c"}
ROW_STYLE = {
    "LOW": "background-color: rgba(46, 204, 113, 0.22)",
    "MEDIUM": "background-color: rgba(243, 156, 18, 0.30)",
    "HIGH": "background-color: rgba(231, 76, 60, 0.40)",
}
BUCKETS = {"15 seconds": 15, "30 seconds": 30, "1 minute": 60}
st.set_page_config(page_title="UPI Fraud Detection", page_icon="\U0001F6E1\uFE0F", layout="wide")
CSS = """
<style>
.hero {padding: 1.3rem 1.6rem; border-radius: 16px; margin-bottom: 1rem; color: white;
       background: linear-gradient(135deg, #1e3c72 0%, #2a5298 45%, #7b2ff7 100%);}
.hero h1 {margin: 0; font-size: 2.1rem; color: white; padding: 0;}
.hero p {margin: 0.35rem 0 0 0; opacity: 0.88; font-size: 1rem;}
.kpi {padding: 0.9rem 1rem; border-radius: 12px; background: rgba(128, 128, 128, 0.12);
      border-left: 6px solid #6c5ce7;}
.kpi .label {font-size: 0.75rem; opacity: 0.75; text-transform: uppercase; letter-spacing: 0.06em;}
.kpi .value {font-size: 1.8rem; font-weight: 700; line-height: 1.25;}
.badge {display: inline-block; padding: 2px 12px; border-radius: 999px; font-weight: 600;
        font-size: 0.8rem; color: white; margin-right: 6px;}
.alert-card {padding: 0.8rem 1rem; border-radius: 12px; margin-bottom: 0.6rem;
             background: rgba(231, 76, 60, 0.14); border-left: 6px solid #e74c3c;}
.alert-card .top {font-weight: 700; font-size: 1.02rem;}
.alert-card .meta {opacity: 0.85; font-size: 0.88rem; margin-top: 2px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    '<div class="hero"><h1>&#128737;&#65039; UPI Fraud Detection</h1>'
    "<p>Real-time monitoring with risk levels, live trends and instant alerts</p></div>",
    unsafe_allow_html=True,
)
def fetch(path):
    """GET from the API; returns None if the API is not reachable."""
    try:
        r = requests.get(f"{API_URL}{path}", timeout=3)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None
def kpi(label, value, color):
    return (
        f'<div class="kpi" style="border-left-color:{color}">'
        f'<div class="label">{label}</div><div class="value">{value}</div></div>'
    )
def badge(level):
    return f'<span class="badge" style="background:{RISK_COLORS[level]}">{level}</span>'
# ---------------- sidebar: filters ----------------
with st.sidebar:
    st.header("Filters")
    st.multiselect("Risk level", RISK_ORDER, default=RISK_ORDER, key="f_risk")
    st.multiselect("City (empty = all cities)", CITIES, key="f_city")
    st.slider("Amount (Rs)", 0, 200000, (0, 200000), step=500, key="f_amount")
    st.text_input("User ID contains", key="f_user", placeholder="e.g. U0123")
    st.checkbox("Show only fraud", key="f_fraud")
    st.selectbox("Trend bucket", list(BUCKETS), index=1, key="f_bucket")
    st.divider()
    health = fetch("/health")
    if health:
        st.success("API online")
        tele = "on" if health.get("telegram_enabled") else "off"
        st.caption(f"Model threshold: {health.get('threshold')}  |  Telegram alerts: {tele}")
    else:
        st.error("API offline - start: uvicorn api.main:app --port 8000")
    st.caption("Filters apply to the table and charts. KPI cards show all data.")
def to_frame(items):
    """API items -> DataFrame with helper columns."""
    df = pd.DataFrame(items)
    df["ts"] = pd.to_datetime(df["time"])
    df["reasons_text"] = df["reasons"].apply(lambda r: ", ".join(r) if r else "-")
    return df
def apply_filters(df):
    ss = st.session_state
    out = df[df["risk"].isin(ss.get("f_risk", RISK_ORDER))]
    cities = ss.get("f_city", [])
    if cities:
        out = out[out["city"].isin(cities)]
    lo, hi = ss.get("f_amount", (0, 200000))
    out = out[(out["amount"] >= lo) & (out["amount"] <= hi)]
    query = str(ss.get("f_user", "")).strip().upper()
    if query:
        out = out[out["user_id"].str.upper().str.contains(query, regex=False)]
    if ss.get("f_fraud", False):
        out = out[out["fraud"]]
    return out
def color_row(row):
    return [ROW_STYLE.get(row["risk"], "")] * len(row)
def trend_frame(df, seconds):
    """Count transactions per time bucket and risk level."""
    d = df.copy()
    d["bucket"] = d["ts"].dt.floor(pd.Timedelta(seconds=seconds))
    t = d.groupby(["bucket", "risk"]).size().unstack(fill_value=0)
    for level in RISK_ORDER:
        if level not in t.columns:
            t[level] = 0
    return t[RISK_ORDER]
def notify_new_high(df):
    """Show a toast popup when a new HIGH risk transaction arrives."""
    high = df[df["risk"] == "HIGH"]
    newest_id = high.iloc[0]["txn_id"] if not high.empty else ""
    if "seen_high" not in st.session_state:
        st.session_state["seen_high"] = newest_id   # first load: do not pop up old ones
        return
    if newest_id and newest_id != st.session_state["seen_high"]:
        row = high.iloc[0]
        st.toast(f"HIGH risk: {row['user_id']} - Rs {row['amount']:,.0f}", icon="\U0001F6A8")
        st.session_state["seen_high"] = newest_id
@st.fragment(run_every=2)
def live_view():
    """Refreshes every 2 seconds: KPI cards, charts, table."""
    items = fetch("/recent")
    if items is None:
        st.warning("Cannot reach the API. Start it with: uvicorn api.main:app --port 8000")
        return
    if not items:
        st.info("No transactions yet. Start the simulator: python simulate.py")
        return
    df = to_frame(items)
    notify_new_high(df)
    total = len(df)
    fraud = int(df["fraud"].sum())
    high = int((df["risk"] == "HIGH").sum())
    flagged_amount = float(df.loc[df["fraud"], "amount"].sum())
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(kpi("Total (last 200)", total, "#3498db"), unsafe_allow_html=True)
    c2.markdown(kpi("Fraud flagged", fraud, "#e74c3c"), unsafe_allow_html=True)
    c3.markdown(kpi("Fraud %", f"{fraud / total * 100:.1f}%", "#f39c12"), unsafe_allow_html=True)
    c4.markdown(kpi("High risk", high, "#c0392b"), unsafe_allow_html=True)
    c5.markdown(kpi("Amount flagged", f"Rs {flagged_amount:,.0f}", "#8e44ad"), unsafe_allow_html=True)
    flt = apply_filters(df)
    st.write("")
    st.caption(f"Showing {len(flt)} of {total} transactions after filters")
    if flt.empty:
        st.info("No transactions match the current filters.")
        return
    left, right = st.columns(2)
    with left:
        st.markdown("##### Transactions by risk level over time")
        seconds = BUCKETS[st.session_state.get("f_bucket", "30 seconds")]
        st.bar_chart(trend_frame(flt, seconds), color=[RISK_COLORS[r] for r in RISK_ORDER])
    with right:
        st.markdown("##### Anomaly score trend (latest 80)")
        scores = flt.iloc[::-1]["score"].tail(80).reset_index(drop=True)
        st.line_chart(scores, color="#6c5ce7")
    st.markdown(
        "##### Live transactions &nbsp; " + "".join(badge(level) for level in RISK_ORDER),
        unsafe_allow_html=True,
    )
    view = flt[["time", "user_id", "amount", "city", "device_id", "risk", "score", "reasons_text"]]
    view = view.rename(columns={"reasons_text": "reasons"})
    styled = view.style.apply(color_row, axis=1).format({"amount": "Rs {:,.2f}", "score": "{:.3f}"})
    st.dataframe(
        styled,
        hide_index=True,
        height=430,
        column_config={"reasons": st.column_config.TextColumn("reasons", width="large")},
    )
@st.fragment(run_every=2)
def alerts_view():
    """Latest HIGH risk alerts."""
    alerts = fetch("/alerts")
    if alerts is None:
        st.warning("Cannot reach the API.")
        return
    st.caption("HIGH risk alerts are also saved in data/alerts.log and sent to Telegram if configured.")
    if not alerts:
        st.info("No HIGH risk alerts yet.")
        return
    st.markdown(f"**{len(alerts)} recent HIGH risk alerts** (newest first)")
    for a in alerts[:15]:
        reasons = ", ".join(a.get("reasons", [])) or "-"
        card = (
            '<div class="alert-card">'
            f'<div class="top">&#128680; {html.escape(str(a["user_id"]))} - Rs {a["amount"]:,.2f}</div>'
            f'<div class="meta">{html.escape(str(a["time"]))} | {html.escape(str(a["city"]))} | '
            f'device {html.escape(str(a["device_id"]))} | score {a["score"]}</div>'
            f'<div class="meta">Reasons: {html.escape(reasons)}</div>'
            "</div>"
        )
        st.markdown(card, unsafe_allow_html=True)
tab_live, tab_alerts, tab_compare = st.tabs(["Live feed", "Alerts", "Method comparison"])
with tab_live:
    live_view()
with tab_alerts:
    alerts_view()
with tab_compare:
    comp_path = DATA_DIR / "comparison.csv"
    if comp_path.exists():
        st.subheader("Precision / Recall / F1 of each method")
        st.dataframe(pd.read_csv(comp_path), hide_index=True)
    else:
        st.info("comparison.csv not found. Run: python run_all.py")
    charts = [
        ("Method comparison", "comparison.png"),
        ("Amount distribution: fraud vs normal", "amount_distribution.png"),
        ("Fraud by hour", "fraud_by_hour.png"),
    ]
    for title, fname in charts:
        path = CHARTS_DIR / fname
        if path.exists():
            st.subheader(title)
            st.image(str(path))
        else:
            st.info(f"{fname} not found. Run: python run_all.py")

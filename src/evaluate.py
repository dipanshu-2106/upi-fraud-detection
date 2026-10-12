"""Stage 4: Evaluate the 3 methods and the ensemble."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # save charts to files, do not open windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHARTS_DIR = DATA_DIR / "charts"
METHODS = {
    "flag_iqr": "IQR (per user)",
    "flag_iforest": "Isolation Forest",
    "flag_timeseries": "Rolling Z-score",
    "flag_ensemble": "Ensemble (2 of 3)",
}
def evaluate_methods(res):
    """Precision, recall, F1 and confusion matrix for every method."""
    y_true = res["is_fraud"]
    rows = []
    for col, name in METHODS.items():
        y_pred = res[col]
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        rows.append({
            "method": name,
            "flagged": int(y_pred.sum()),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
            "f1": round(f1_score(y_true, y_pred, zero_division=0), 3),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        })
    return pd.DataFrame(rows)
def markdown_table(table):
    """Turn the comparison table into a markdown table (used for README)."""
    cols = ["method", "flagged", "precision", "recall", "f1", "tp", "fp", "fn", "tn"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in table[cols].iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)
def chart_comparison(table):
    """Bar chart: precision, recall and F1 of every method."""
    metrics = ["precision", "recall", "f1"]
    labels = ["Precision", "Recall", "F1"]
    x = np.arange(len(table))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (m, lab) in enumerate(zip(metrics, labels)):
        bars = ax.bar(x + (i - 1) * width, table[m], width, label=lab)
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(table["method"])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Method comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "comparison.png", dpi=120)
    plt.close(fig)
def chart_amount_distribution(res):
    """Histogram of amount: fraud vs normal (log scale on x)."""
    normal = res.loc[res["is_fraud"] == 0, "amount"]
    fraud = res.loc[res["is_fraud"] == 1, "amount"]
    bins = np.logspace(np.log10(res["amount"].min()), np.log10(res["amount"].max()), 40)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(normal, bins=bins, alpha=0.6, density=True, label="Normal", color="green")
    ax.hist(fraud, bins=bins, alpha=0.6, density=True, label="Fraud", color="red")
    ax.set_xscale("log")
    ax.set_xlabel("Amount (Rs, log scale)")
    ax.set_ylabel("Density")
    ax.set_title("Amount distribution: fraud vs normal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "amount_distribution.png", dpi=120)
    plt.close(fig)
def chart_fraud_by_hour(res):
    """Bar chart: how many fraud transactions happen in each hour."""
    hours = res.loc[res["is_fraud"] == 1, "timestamp"].dt.hour
    counts = hours.value_counts().reindex(range(24), fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(counts.index, counts.values, color="red", alpha=0.8)
    ax.set_xticks(range(24))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Number of fraud transactions")
    ax.set_title("Fraud by hour")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "fraud_by_hour.png", dpi=120)
    plt.close(fig)
def print_conclusion(table):
    """Short plain-language conclusion."""
    best = table.loc[table["f1"].idxmax()]
    top_p = table.loc[table["precision"].idxmax()]
    top_r = table.loc[table["recall"].idxmax()]
    print("\nConclusion (simple words):")
    print(f"- Best overall (highest F1): {best['method']} (F1 = {best['f1']}).")
    print(f"- Fewest false alarms (highest precision): {top_p['method']} ({top_p['precision']}).")
    print(f"- Catches most fraud (highest recall): {top_r['method']} ({top_r['recall']}).")
    print("- IQR and Z-score are simple and easy to explain, but they look at amount only.")
    print("- Isolation Forest is better when fraud shows in many signals together")
    print("  (amount, hour, device, city, speed of transactions).")
    print("- Ensemble needs 2 methods to agree, so compare its precision and recall")
    print("  with the single methods in the table to see the trade-off.")
    print("- Remember: data is synthetic, so real-world accuracy will be lower.")
def main():
    res = pd.read_csv(DATA_DIR / "results.csv", parse_dates=["timestamp"])
    table = evaluate_methods(res)
    table.to_csv(DATA_DIR / "comparison.csv", index=False)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    chart_comparison(table)
    chart_amount_distribution(res)
    chart_fraud_by_hour(res)
    print("Comparison table:")
    print(table.to_string(index=False))
    for _, r in table.iterrows():
        print(f"\nConfusion matrix: {r['method']}")
        print("               Pred normal  Pred fraud")
        print(f"  Real normal  {r['tn']:>10}  {r['fp']:>10}")
        print(f"  Real fraud   {r['fn']:>10}  {r['tp']:>10}")
    print_conclusion(table)
    print(f"\nSaved: {DATA_DIR / 'comparison.csv'}")
    print(f"Saved charts in: {CHARTS_DIR}")
    return table
if __name__ == "__main__":
    main()

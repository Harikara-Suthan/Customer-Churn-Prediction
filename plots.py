"""Matplotlib figures shared by the dashboard and evaluate.py.

Figures are built with matplotlib's Figure class (no pyplot), so they work
the same in Streamlit and in scripts.
"""
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from preprocessing import TARGET

STAY_COLOR = "#4c78a8"
CHURN_COLOR = "#d9742b"
GRID = "#dddddd"


def _new_axes(width=6.0, height=3.6):
    fig = Figure(figsize=(width, height), layout="constrained")
    ax = fig.subplots()
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax


def churn_distribution(df: pd.DataFrame) -> Figure:
    counts = df[TARGET].value_counts().reindex([0, 1], fill_value=0)
    fig, ax = _new_axes(4.5, 3.2)
    bars = ax.bar(["Stayed", "Churned"], counts.values, color=[STAY_COLOR, CHURN_COLOR])
    for bar, n in zip(bars, counts.values):
        share = n / counts.sum() if counts.sum() else 0
        ax.text(bar.get_x() + bar.get_width() / 2, n, f"{n:,}\n({share:.1%})",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Customers")
    ax.set_ylim(0, counts.max() * 1.2 if counts.max() else 1)
    ax.set_title("Overall churn distribution")
    return fig


def churn_rate_by_category(df: pd.DataFrame, column: str) -> Figure:
    stats = df.groupby(column)[TARGET].agg(["mean", "count"]).sort_values("mean")
    fig, ax = _new_axes(6.0, 0.9 + 0.55 * len(stats))
    ax.barh(stats.index.astype(str), stats["mean"] * 100, color=CHURN_COLOR)
    overall = df[TARGET].mean() * 100
    ax.axvline(overall, color="#555555", linestyle="--", linewidth=1)
    ax.text(overall, len(stats) - 0.45, f" overall {overall:.1f}%", fontsize=8, color="#555555")
    for i, (rate, n) in enumerate(zip(stats["mean"] * 100, stats["count"])):
        ax.text(rate, i, f" {rate:.1f}%  (n={n:,})", va="center", fontsize=8)
    ax.set_xlim(0, max(stats["mean"].max() * 100 * 1.35, 5))
    ax.set_xlabel("Churn rate (%)")
    ax.set_title(f"Churn rate by {column}")
    return fig


def churn_rate_by_tenure_group(df: pd.DataFrame) -> Figure:
    bins = [-1, 12, 24, 36, 48, 60, np.inf]
    labels = ["0-12", "13-24", "25-36", "37-48", "49-60", "61+"]
    group = pd.cut(df["tenure"], bins=bins, labels=labels)
    stats = df.groupby(group, observed=True)[TARGET].agg(["mean", "count"])
    fig, ax = _new_axes(6.0, 3.4)
    ax.bar(stats.index.astype(str), stats["mean"] * 100, color=CHURN_COLOR)
    for i, (rate, n) in enumerate(zip(stats["mean"] * 100, stats["count"])):
        ax.text(i, rate, f"{rate:.1f}%\nn={n:,}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(stats["mean"].max() * 100 * 1.3, 5))
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Churn rate (%)")
    ax.set_title("Churn rate by tenure")
    return fig


def numeric_hist_by_churn(df: pd.DataFrame, column: str) -> Figure:
    fig, ax = _new_axes(6.0, 3.4)
    values = df[column].dropna()
    bins = np.histogram_bin_edges(values, bins=25)
    for flag, label, color in [(0, "Stayed", STAY_COLOR), (1, "Churned", CHURN_COLOR)]:
        subset = df.loc[df[TARGET] == flag, column].dropna()
        ax.hist(subset, bins=bins, alpha=0.6, color=color, label=label, density=True)
    ax.set_xlabel(column)
    ax.set_ylabel("Share of customers (density)")
    ax.set_title(f"{column} distribution by churn")
    ax.legend(frameon=False)
    return fig


def confusion_matrix_fig(cm: np.ndarray) -> Figure:
    fig = Figure(figsize=(4.2, 3.8), layout="constrained")
    ax = fig.subplots()
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax, shrink=0.8)
    names = ["Stayed", "Churned"]
    ax.set_xticks([0, 1], names)
    ax.set_yticks([0, 1], names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (test set)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    return fig


def roc_curve_fig(fpr, tpr, auc: float) -> Figure:
    fig, ax = _new_axes(4.4, 3.8)
    ax.plot(fpr, tpr, color=CHURN_COLOR, label=f"Model (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#888888", linestyle="--", label="Random guess")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curve (test set)")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(color=GRID, linewidth=0.5)
    return fig


def importance_bar(importance: pd.DataFrame, title: str, xlabel: str) -> Figure:
    """Horizontal bars for a table with columns 'feature' and 'value'."""
    data = importance.iloc[::-1]
    fig, ax = _new_axes(6.0, 0.8 + 0.4 * len(data))
    ax.barh(data["feature"], data["value"], color=STAY_COLOR)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    return fig


def effect_bar(effects: pd.DataFrame) -> Figure:
    """Signed bars: positive (orange) raises the churn estimate, negative (blue) lowers it."""
    data = effects.iloc[::-1]
    labels = [f"{f} = {v}" for f, v in zip(data["feature"], data["customer_value"])]
    values = data["effect_on_probability"] * 100
    fig, ax = _new_axes(6.0, 0.8 + 0.42 * len(data))
    ax.barh(labels, values, color=[CHURN_COLOR if v > 0 else STAY_COLOR for v in values])
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_xlabel("Change in churn probability vs. a typical customer (percentage points)")
    return fig

"""
Stage 2: Exploratory Data Analysis.

Produces the year-month core inflation heatmap, the CPI-components-vs-repo-rate
chart, and correlation + yearly summary tables.

Run standalone:
    python -m src.eda
"""

import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df_cpi = pd.read_csv(config.CLEANED_CPI_FILE)
    df_cpi["date"] = pd.to_datetime(df_cpi["date"])
    df_mpc = pd.read_csv(config.PROCESSED_CPI_MPC_FILE)
    df_mpc["date"] = pd.to_datetime(df_mpc["date"])
    return df_cpi, df_mpc


def plot_core_cpi_heatmap(df_cpi: pd.DataFrame, out_path=None) -> None:
    """Year x month heatmap of core CPI YoY inflation."""
    out_path = out_path or config.OUTPUTS_DIR / "01_core_cpi_heatmap.png"
    df_cpi = df_cpi.copy()
    df_cpi["year"] = df_cpi["date"].dt.year
    df_cpi["month"] = df_cpi["date"].dt.month

    heat = df_cpi.pivot_table(index="year", columns="month", values="cpi_core_yoy", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        heat, cmap="RdYlGn_r", center=4, annot=True, fmt=".1f",
        linewidths=0.3, ax=ax, cbar_kws={"label": "Core CPI YoY %"},
    )
    ax.set_title(
        "India Core CPI Inflation (%) — Monthly, 2015–2025\n"
        "Green = Below 4% target band | Red = Elevated",
        fontsize=12, pad=10,
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)


def plot_cpi_vs_repo_rate(df_cpi: pd.DataFrame, df_mpc: pd.DataFrame, out_path=None) -> None:
    """Two-panel chart: CPI component lines on top, repo rate steps + hike/cut
    markers on the bottom."""
    out_path = out_path or config.OUTPUTS_DIR / "02_cpi_vs_repo_rate.png"

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    ax1 = axes[0]
    ax1.plot(df_cpi["date"], df_cpi["cpi_general_yoy"], color="#2c3e50", linewidth=2.5, label="Headline CPI")
    ax1.plot(df_cpi["date"], df_cpi["cpi_core_yoy"], color="#e74c3c", linewidth=2, linestyle="--", label="Core (demand-side)")
    ax1.plot(df_cpi["date"], df_cpi["cpi_food_yoy"], color="#27ae60", linewidth=1.5, alpha=0.8, label="Food (supply-side)")
    ax1.plot(df_cpi["date"], df_cpi["cpi_fuel_yoy"], color="#f39c12", linewidth=1.5, alpha=0.8, label="Fuel")
    ax1.axhline(y=4, color="grey", linestyle=":", linewidth=1, label="RBI 4% target")
    ax1.axhline(y=6, color="red", linestyle=":", linewidth=1, alpha=0.5, label="Upper tolerance band")
    ax1.legend(fontsize=9)
    ax1.set_ylabel("YoY Inflation %")
    ax1.set_title("India CPI Components vs. RBI Monetary Policy Repo Rate")
    ax1.grid(True, alpha=0.2)

    ax2 = axes[1]
    df_cpi_sorted = df_cpi.sort_values("date").copy()
    df_mpc_sorted = df_mpc.sort_values("date")
    df_cpi_sorted["repo_rate"] = (
        df_cpi_sorted["date"].map(dict(zip(df_mpc_sorted["date"], df_mpc_sorted["repo_rate"]))).ffill()
    )
    ax2.step(df_cpi_sorted["date"], df_cpi_sorted["repo_rate"], color="#8e44ad", linewidth=2, label="Repo Rate %")

    hikes = df_mpc[df_mpc["decision"] == "hike"]
    cuts = df_mpc[df_mpc["decision"] == "cut"]
    ax2.scatter(hikes["date"], hikes["repo_rate"], color="red", zorder=5, s=60, label="Rate Hike", marker="^")
    ax2.scatter(cuts["date"], cuts["repo_rate"], color="green", zorder=5, s=60, label="Rate Cut", marker="v")
    ax2.set_ylabel("Repo Rate %")
    ax2.set_xlabel("Date")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)


def compute_correlations(df_cpi: pd.DataFrame) -> pd.DataFrame:
    """Correlation of each supply/demand component with headline CPI.
    Saved to CSV — the original notebook only printed these to console."""
    components = {
        "Food (supply)": "cpi_food_yoy",
        "Fuel (supply)": "cpi_fuel_yoy",
        "Core (demand)": "cpi_core_yoy",
    }
    rows = []
    for name, col in components.items():
        valid = df_cpi[["cpi_general_yoy", col]].dropna()
        corr = valid["cpi_general_yoy"].corr(valid[col])
        rows.append({"component": name, "correlation_with_headline": round(corr, 3)})
        logger.info("Correlation (%s vs headline CPI): %.3f", name, corr)

    corr_df = pd.DataFrame(rows)
    corr_df.to_csv(config.OUTPUTS_DIR / "cpi_component_correlations.csv", index=False)
    return corr_df


def compute_yearly_summary(df_cpi: pd.DataFrame) -> pd.DataFrame:
    """Year-wise average inflation by component, saved to CSV."""
    df_cpi = df_cpi.copy()
    df_cpi["year"] = df_cpi["date"].dt.year
    yearly = df_cpi.groupby("year")[
        ["cpi_general_yoy", "cpi_food_yoy", "cpi_fuel_yoy", "cpi_core_yoy"]
    ].mean().round(2)
    yearly.to_csv(config.OUTPUTS_DIR / "yearly_cpi_components.csv")
    logger.info("Saved yearly_cpi_components.csv (%d years)", len(yearly))
    return yearly


def run() -> None:
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df_cpi, df_mpc = load_data()

    plot_core_cpi_heatmap(df_cpi)
    plot_cpi_vs_repo_rate(df_cpi, df_mpc)
    compute_correlations(df_cpi)
    compute_yearly_summary(df_cpi)

    logger.info("Stage 2 complete.")


if __name__ == "__main__":
    run()
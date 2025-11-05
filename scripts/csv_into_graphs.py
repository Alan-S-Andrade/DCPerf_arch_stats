#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

INPUT_DATA_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
PLOT_DIR = INPUT_DATA_DIR.parent / "plots"

def load_perf_csvs(data_dir: Path):
    csv_files = sorted(data_dir.glob("*.csv"))
    data = {}
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            df.columns = [c.strip() for c in df.columns]
            df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
            workload = f.stem
            data[workload] = df.set_index("Counter")["Value"].to_dict()
        except Exception as e:
            print(f"Error parsing {f.name}: {e}")
    return pd.DataFrame(data).T  # Workloads as rows, counters as columns


def compute_metrics(df: pd.DataFrame):
    df = df.copy()

    # IPC: Instructions per total cycles
    if "instructions" in df.columns and "cycles" in df.columns:
        df["IPC"] = df["instructions"] / df["cycles"]

    # User vs Kernel cycle ratio
    if "cycles:u" in df.columns and "cycles:k" in df.columns:
        total_cycles = df["cycles:u"] + df["cycles:k"]
        df["User_Cycle_Pct"] = df["cycles:u"] / total_cycles * 100
        df["Kernel_Cycle_Pct"] = df["cycles:k"] / total_cycles * 100

        # Weighted IPC contributions (for stacked IPC bars)
        df["User_IPC"] = df["IPC"] * (df["User_Cycle_Pct"] / 100.0)
        df["Kernel_IPC"] = df["IPC"] * (df["Kernel_Cycle_Pct"] / 100.0)

    # MPKI metrics: misses per 1K instructions
    for miss_event in ["L1-dcache-load-misses", "LLC-load-misses", "iTLB-load-misses", "L1-icache-load-misses"]:
        if miss_event in df.columns and "instructions" in df.columns:
            df[f"{miss_event}_MPKI"] = 1000 * df[miss_event] / df["instructions"]

    # Branch misprediction rate
    if "BR_MISP_RETIRED.ALL_BRANCHES" in df.columns and "BR_INST_RETIRED.ALL_BRANCHES" in df.columns:
        df["Branch_Mispredict_Rate"] = df["BR_MISP_RETIRED.ALL_BRANCHES"] / df["BR_INST_RETIRED.ALL_BRANCHES"] * 100

    return df


def plot_grouped_bars(df: pd.DataFrame, metrics: list, title: str, ylabel: str, outfile: Path):
    """Plot grouped bar chart for given metrics across workloads."""
    sub = df[metrics].dropna(axis=0, how="all")
    if sub.empty:
        print(f"Skipping {title}: no data available.")
        return
    ax = sub.plot(kind="bar", figsize=(12, 6), width=0.8)
    plt.title(title, fontsize=14)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=30, ha="right")
    plt.legend(fontsize=9)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()

def plot_ipc(df: pd.DataFrame, outfile: Path):
    """Plot IPC with stacked user/kernel breakdown"""
    if not {"User_IPC", "Kernel_IPC"}.issubset(df.columns):
        print("Skipping IPC stacked plot: missing user/kernel cycle data.")
        return

    sub = df[["User_IPC", "Kernel_IPC"]].dropna()
    ax = sub.plot(
        kind="bar",
        stacked=True,
        figsize=(12, 6),
        color=["#F28603", "#0915F6"],
        width=0.8,
    )

    plt.title("Instructions Per Cycle (IPC) with User/Kernel Breakdown", fontsize=14)
    plt.ylabel("IPC", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    plt.legend(["User", "Kernel"], fontsize=10, loc="upper right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # total IPC and user/kernel %
    for i, workload in enumerate(sub.index):
        total_ipc = sub.loc[workload].sum()
        user_pct = df.loc[workload, "User_Cycle_Pct"]
        kern_pct = df.loc[workload, "Kernel_Cycle_Pct"]
        plt.text(i, total_ipc + total_ipc * 0.02, f"{total_ipc:.2f}", ha="center", va="bottom", fontsize=9)
        plt.text(i, sub.loc[workload, "User_IPC"]/2, f"{user_pct:.0f}%", ha="center", va="center", color="white", fontsize=8)
        plt.text(i, sub.loc[workload, "User_IPC"] + sub.loc[workload, "Kernel_IPC"]/2,
                 f"{kern_pct:.0f}%", ha="center", va="center", color="white", fontsize=8)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


def find_columns(df, prefix):
    """return all columns starting with a prefix like 'BR_MISP_RETIRED.' """
    return [c for c in df.columns if c.startswith(prefix)]

if __name__ == "__main__":
    df = load_perf_csvs(INPUT_DATA_DIR)
    df = compute_metrics(df)

    PLOT_DIR.mkdir(exist_ok=True)

   # === IPC (user/kernel) ===
    plot_ipc(df, PLOT_DIR / "ipc_stacked.png")

    # === MPKI ===
    mpki_metrics = [c for c in df.columns if c.endswith("_MPKI")]
    if mpki_metrics:
        plot_grouped_bars(df, mpki_metrics, "Cache Misses Per Kilo Instructions", "MPKI", PLOT_DIR / "mpki.png")

    # === Branch Misprediction Rate ===
    if "Branch_Mispredict_Rate" in df.columns:
        plot_grouped_bars(df, ["Branch_Mispredict_Rate"], "Branch Misprediction Rate", "Rate (%)", PLOT_DIR / "branch_mispredict_rate.png")

    # === Expanded branch misprediction breakdown ===
    mispred_cols = find_columns(df, "BR_MISP_RETIRED.")
    if mispred_cols:
        plot_grouped_bars(
            df,
            mispred_cols,
            "Branch Mispredictions Breakdown",
            "Event Count",
            PLOT_DIR / "branch_mispredict_breakdown.png"
        )

    # === Expanded branches retired breakdown ===
    retired_cols = find_columns(df, "BR_INST_RETIRED.")
    if retired_cols:
        plot_grouped_bars(
            df,
            retired_cols,
            "Branches Retired Breakdown",
            "Event Count",
            PLOT_DIR / "branch_retired_breakdown.png"
        )

    # === BACLEARS ===
    if "BACLEARS.ANY" in df.columns:
        plot_grouped_bars(df, ["BACLEARS.ANY"], "Branch Clears (BACLEARS.ANY)", "Count", PLOT_DIR / "baclears_any.png")

    print(f"Plots stored at: {PLOT_DIR.resolve()}")

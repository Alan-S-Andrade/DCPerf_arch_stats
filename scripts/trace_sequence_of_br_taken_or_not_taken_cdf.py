#!/usr/bin/env python3
"""
Compute CDFs of consecutive taken and not-taken conditional branch run lengths across multiple traces.

Usage:
    python3 branch_run_length_cdf.py trace1.data trace2.data ...

Output:
    ../plots/taken_run_length_cdf.png
    ../plots/not_taken_run_length_cdf.png
"""

import re
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("../user_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Regex to parse perf-like disassembly lines
LINE_RE = re.compile(
    r"^\s*\S+\s+\d+\s+\[\d+\]\s+[\d\.]+:\s+([0-9A-Fa-f]+)\s+(.+?)\s*\(([^)]+)\)"
)

def parse_trace(path: Path):
    """Parse input file and return a list of (VA, mnemonic)."""
    instrs = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = LINE_RE.match(line)
            if not m:
                continue
            va = int(m.group(1), 16)
            asm = m.group(2).strip()
            if not asm:
                continue
            mnemonic = asm.split()[0].lower()
            instrs.append((va, mnemonic))
    return instrs

def is_conditional_branch(mnemonic: str) -> bool:
    """True if the instruction is a conditional branch (jz, jne, jge and not callq or retq)."""
    return mnemonic.startswith("j") and mnemonic not in {"jmp", "jmpq", "call", "callq", "ret", "retq"}

def classify_branches(instrs):
    """Return list of tuples (VA, taken_flag) for conditional branches."""
    branches = []
    for i, (va, mnemonic) in enumerate(instrs[:-1]):
        if not is_conditional_branch(mnemonic):
            continue
        next_va = instrs[i + 1][0]
        # heuristic: if the next VA is not sequential (offset > 16), branch was taken
        taken = abs(next_va - va) > 16
        branches.append((va, taken))
    return branches

def compute_runs(branches):
    """Compute consecutive run lengths for taken and not-taken."""
    if not branches:
        return [], []
    taken_runs, not_taken_runs = [], []
    current_taken = branches[0][1]
    length = 1
    for _, taken in branches[1:]:
        if taken == current_taken:
            length += 1
        else:
            if current_taken:
                taken_runs.append(length)
            else:
                not_taken_runs.append(length)
            current_taken = taken
            length = 1
    # finalize
    if current_taken:
        taken_runs.append(length)
    else:
        not_taken_runs.append(length)
    return taken_runs, not_taken_runs

def compute_cdf(data):
    """Return sorted x and cumulative y values."""
    if not data:
        return [], []
    data = sorted(data)
    n = len(data)
    x = data
    y = [i / n for i in range(1, n + 1)]
    return x, y

def plot_cdfs(run_dict, title, out_path):
    plt.figure(figsize=(9, 6))
    for label, runs in run_dict.items():
        if not runs:
            continue
        x, y = compute_cdf(runs)
        plt.plot(x, y, label=label, linewidth=2)
    plt.xlabel("Consecutive Conditional Branch Run Length")
    plt.ylabel("CDF")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xscale("log")  # log helps for long tails
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Wrote {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Compute taken/not-taken conditional branch run length CDFs")
    ap.add_argument("traces", nargs="+", help="input trace files")
    args = ap.parse_args()

    taken_dict, not_taken_dict = {}, {}

    for trace_path in args.traces:
        path = Path(trace_path)
        if not path.exists():
            print(f"[WARN] {path} not found, skipping")
            continue
        instrs = parse_trace(path)
        branches = classify_branches(instrs)
        taken_runs, not_taken_runs = compute_runs(branches)
        taken_dict[path.stem] = taken_runs
        not_taken_dict[path.stem] = not_taken_runs
        print(f"{path.name}: {len(instrs)} instructions, {len(branches)} conditional branches, "
              f"{len(taken_runs)} taken-runs, {len(not_taken_runs)} not-taken-runs")

    if not taken_dict:
        print("No data found")
        return

    plot_cdfs(taken_dict, "CDF of Consecutive Taken Conditional Branch Run Lengths", OUT_DIR / "taken_run_length_cdf.png")
    plot_cdfs(not_taken_dict, "CDF of Consecutive Not-Taken Conditional Branch Run Lengths", OUT_DIR / "not_taken_run_length_cdf.png")

if __name__ == "__main__":
    main()
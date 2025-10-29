#!/usr/bin/env python3
"""
Compute CDFs of absolute jump/branch target distances across applications instruction traces

Usage:
    python3 trace_jump_cdf_abs_log.py django.data feedsim.data mediawiki.data
"""

import re
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

OUT_DIR = Path("../user_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LINE_RE = re.compile(
    r"^\s*\S+\s+\d+\s+\[\d+\]\s+[\d\.]+:\s+([0-9A-Fa-f]+)\s+(.+?)\s*\(([^)]+)\)"
)
SYM_OFF_RE = re.compile(r"^(?P<sym>.+?)\s*(?:@plt)?\s*\+\s*0x(?P<off>[0-9A-Fa-f]+)\s*$")

def normalize_symbol(sym: str) -> str:
    return re.sub(r'@plt(?:\+0x[0-9A-Fa-f]+)?', '', sym.strip())

class Instr:
    def __init__(self, va, sym, off, obj):
        self.va = va
        self.sym = sym
        self.off = off
        self.obj = obj

def parse_line(line: str):
    m = LINE_RE.match(line)
    if not m:
        return None
    va = int(m.group(1), 16)
    sympart = m.group(2).strip()
    obj = m.group(3).strip()
    mo = SYM_OFF_RE.search(sympart)
    off = None
    sym = sympart
    if mo:
        sym = mo.group("sym").strip()
        off = int(mo.group("off"), 16)
    else:
        m2 = re.search(r'\+0x([0-9A-Fa-f]+)', sympart)
        if m2:
            off = int(m2.group(1), 16)
            sym = re.sub(r'\+0x[0-9A-Fa-f]+', '', sympart).strip()
    return Instr(va, normalize_symbol(sym), off, obj)

def read_trace(path: Path):
    instrs = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            i = parse_line(line)
            if i:
                instrs.append(i)
    return instrs

def group_blocks(instrs):
    """Returns list of basic blocks (start_va, end_va)"""
    if not instrs:
        return []
    blocks = []
    curr = instrs[0]
    start_va = curr.va
    end_va = curr.va
    for inst in instrs[1:]:
        sequential = False
        if (inst.obj == curr.obj) and (inst.sym == curr.sym) and (curr.off is not None and inst.off is not None):
            if inst.va - curr.va == inst.off - curr.off and inst.va - curr.va >= 0:
                sequential = True
        if sequential:
            end_va = inst.va
        else:
            blocks.append((start_va, end_va))
            start_va = inst.va
            end_va = inst.va
        curr = inst
    blocks.append((start_va, end_va))
    return blocks

def compute_jump_sizes(blocks):
    """Compute absolute jump distances between consecutive blocks"""
    jumps = []
    for i in range(len(blocks)-1):
        curr_end = blocks[i][1]
        next_start = blocks[i+1][0]
        jump = abs(next_start - curr_end)
        jumps.append(jump)
    return jumps

def compute_cdf(data):
    data = sorted(data)
    n = len(data)
    x = data
    y = [i/n for i in range(1, n+1)]
    return x, y

def plot_cdfs(jump_dict):
    # --- Full linear CDF ---
    plt.figure(figsize=(9,6))
    for label, jumps in jump_dict.items():
        if not jumps:
            continue
        x, y = compute_cdf(jumps)
        plt.plot(x, y, label=label, linewidth=2)
    plt.xlabel("Absolute Jump Size (VA difference)")
    plt.ylabel("CDF")
    plt.title("CDF of Jump/Branch Target Sizes Across Applications")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    full_path = OUT_DIR / "jump_cdf_abs_full.png"
    plt.savefig(full_path, dpi=150)
    plt.close()
    print(f"Wrote {full_path}")

    # --- Linear CDF zoomed upper tail ---
    plt.figure(figsize=(9,6))
    for label, jumps in jump_dict.items():
        if not jumps:
            continue
        x, y = compute_cdf(jumps)
        plt.plot(x, y, label=label, linewidth=2)
    plt.xlabel("Absolute Jump Size (VA difference)")
    plt.ylabel("CDF")
    plt.title("CDF of Jump/Branch Target Sizes Across Applications (zoomed)")
    plt.ylim(0.7, 1.0)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    zoom_path = OUT_DIR / "jump_cdf_abs_zoomed.png"
    plt.savefig(zoom_path, dpi=150)
    plt.close()
    print(f"Wrote {zoom_path}")

    # --- CDF with log-scale x-axis ---
    plt.figure(figsize=(9,6))
    for label, jumps in jump_dict.items():
        if not jumps:
            continue
        x, y = compute_cdf(jumps)
        plt.plot(x, y, label=label, linewidth=2)
    plt.xlabel("Absolute Jump Size (VA difference, log scale)")
    plt.ylabel("CDF")
    plt.title("CDF of Jump/Branch Target Sizes Across Applications (log-x)")
    plt.xscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    logx_path = OUT_DIR / "jump_cdf_abs_logx.png"
    plt.savefig(logx_path, dpi=150)
    plt.close()
    print(f"Wrote {logx_path}")

    # --- Log-transformed CDF ---
    plt.figure(figsize=(9,6))
    for label, jumps in jump_dict.items():
        if not jumps:
            continue
        log_jumps = [np.log10(j) for j in jumps if j > 0]
        x, y = compute_cdf(log_jumps)
        plt.plot(x, y, label=label, linewidth=2)
    plt.xlabel("Log10(Absolute Jump Size)")
    plt.ylabel("CDF")
    plt.title("CDF of Log10 Jump/Branch Target Sizes Across Applications")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    log_path = OUT_DIR / "jump_cdf_abs_log.png"
    plt.savefig(log_path, dpi=150)
    plt.close()
    print(f"Wrote {log_path}")

def main():
    ap = argparse.ArgumentParser(description="Compute CDFs of absolute jump distances")
    ap.add_argument("traces", nargs="+", help="trace files")
    args = ap.parse_args()

    jump_dict = {}
    for trace_path in args.traces:
        path = Path(trace_path)
        if not path.exists():
            print(f"{path} not found")
            continue
        instrs = read_trace(path)
        blocks = group_blocks(instrs)
        jumps = compute_jump_sizes(blocks)
        jump_dict[path.stem] = jumps
        print(f"{path.name}: {len(instrs)} instructions → {len(blocks)} blocks → {len(jumps)} jumps")

    if jump_dict:
        plot_cdfs(jump_dict)
    else:
        print("No valid jumps to plot")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Parse instruction trace files, group into basic blocks, and plot all their CDFs on one plot.

Usage:python3 trace_into_block_length_cdf.py django.data feedsim.data mediawiki.data video_transcode.data taobench.data
"""

import re
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("../user_plots")

# --- regex parsers ---
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
    instructions = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            i = parse_line(line)
            if i:
                instructions.append(i)
    return instructions

def group_blocks(instructions):
    if not instructions:
        return []
    blocks = []
    curr = instructions[0]
    count = 1
    for inst in instructions[1:]:
        sequential = False
        if (inst.obj == curr.obj) and (inst.sym == curr.sym) and (curr.off is not None and inst.off is not None):
            if inst.va - curr.va == inst.off - curr.off and inst.va - curr.va >= 0: # we have the same virtual address offset as the previouys instruction (compare address and symbol) => sequential
                sequential = True
        if sequential:
            count += 1
        else:
            blocks.append(count)
            count = 1
        curr = inst
    blocks.append(count)
    return blocks

def compute_cdf(data):
    data = sorted(data)
    n = len(data)
    x = data
    y = [i / n for i in range(1, n + 1)]
    return x, y

def plot_cdfs(block_dict):
    # --- Full CDF ---
    plt.figure(figsize=(9, 6))
    for label, sizes in block_dict.items():
        if not sizes:
            continue
        x, y = compute_cdf(sizes)
        plt.plot(x, y, label=label, linewidth=2)
    # plt.xscale("log")
    plt.xlabel("Basic Block Size (instructions, log scale)")
    plt.ylabel("CDF")
    plt.title("CDF of Basic Block Sizes across User-Space Execution") # change to kernel
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()
    full_path = OUT_DIR / "block_cdf_full.png"
    plt.savefig(full_path, dpi=150)
    plt.close()
    print(f"Wrote {full_path}")

    # --- Zoomed upper tail ---
    plt.figure(figsize=(9, 6))
    for label, sizes in block_dict.items():
        if not sizes:
            continue
        x, y = compute_cdf(sizes)
        plt.plot(x, y, label=label, linewidth=2)
    # plt.xscale("log")
    plt.xlabel("Basic Block Size (instructions, log scale)")
    plt.ylabel("CDF")
    plt.title("CDF of Basic Block Sizes across User-Space Execution (zoomed)") # change to kernel    
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.ylim(0.6, 1.0)
    plt.yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    plt.tight_layout()
    zoom_path = OUT_DIR / "block_cdf_zoomed.png"
    plt.savefig(zoom_path, dpi=150)
    plt.close()
    print(f"Wrote {zoom_path}")

def main():
    ap = argparse.ArgumentParser(description="Plot CDF of basic block sizes across apps")
    ap.add_argument("traces", nargs="+", help="trace file paths")
    args = ap.parse_args()

    block_dict = {}
    for trace_path in args.traces:
        path = Path(trace_path)
        if not path.exists():
            print(f"{path} not found")
            return
        instrs = read_trace(path)
        blocks = group_blocks(instrs)
        block_dict[path.stem] = blocks
        print(f"{path.name}: {len(instrs)} instructions → {len(blocks)} blocks")

    if not block_dict:
        print("No traces parsed")
        return

    plot_cdfs(block_dict)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Compute:
  1) CDFs of absolute jump/branch target distances across traces
  2) Scatter plots of jump size vs. preceding block length
  3) For a SINGLE app (--familyApp NAME):
       - Multi-curve CDF of instruction-family counts per basic block
       - Stacked bar of average per-block family composition
  4) NEW: CDFs of Read-After-Write (RAW) dependency distances across traces

Usage:
  python3 trace_jump_cdf_abs_log.py -i traces/ -o results/ [--kernelSpace] [--familyApp NAME]

Input lines (perf script disassembly style), e.g.:
   hhvmworker 1637979 [001] 450361.144668968:      7c6a49b64cc0 pcre_exec+0x0 (/usr/lib/x86_64-linux-gnu/libpcre.so.3.13.3)         nop %edi, %edx
"""

import re
import csv
from collections import Counter, defaultdict
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from typing import Tuple, Set, Dict, List, Optional

# -----------------------------
# Parsing & classification
# -----------------------------

OUT_DIR = Path("wdl_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

class Instruction:
    def __init__(self, process, pid, cpu, timestamp, address, function, binary, instruction):
        self.process = process
        self.pid = pid
        self.cpu = cpu
        self.timestamp = timestamp
        self.address = address           # hex string without 0x
        self.function = function
        self.binary = binary
        self.instruction = instruction

def write_percentile_table(prefix: str, data_dict: Dict[str, List[int]], output_dir: Path):
    """
    Write percentile tables (10th .. 100th percentile) for each key in data_dict.
    Saves: <output_dir>/<prefix>_<key>_percentiles.csv
    """
    percentiles = [i / 10.0 for i in range(1, 11)]  # 0.1, 0.2, ... 1.0

    for key, values in data_dict.items():
        if not values:
            continue

        arr = np.sort(np.asarray(values))
        n = len(arr)
        if n == 0:
            continue

        pct_vals = []
        for p in percentiles:
            idx = int(np.ceil(p * n)) - 1
            idx = max(0, min(idx, n - 1))
            pct_vals.append(arr[idx])

        path = output_dir / f"{prefix}_{key}_percentiles.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Percentile", "Value"])
            for p, v in zip(percentiles, pct_vals):
                w.writerow([p, v])

        print(f"Wrote {path}")

def parse_line(line) -> Optional[Instruction]:
    if "instruction trace error" in line:
        return None
    line = line.strip()
    if not line:
        return None

    pattern = (
        r"^(\S+)\s+"              # process
        r"(\d+)\s+"               # PID
        r"\[(\d+)\]\s+"           # CPU
        r"([\d\.]+):\s+"          # timestamp
        r"([0-9a-fA-F]+)\s+"      # address (hex, no 0x)
        r"(.+?)\s+\(([^)]+)\)\s+" # function and binary
        r"(.+)$"                  # instruction text
    )
    m = re.match(pattern, line)
    if not m:
        return None

    process, pid, cpu, timestamp, address, function, binary, instruction = m.groups()
    return Instruction(process, pid, cpu, timestamp, address, function.strip(), binary.strip(), instruction.strip())

def classify_instruction(instr: str) -> str:
    """
    Heuristic categorization into families:
      branch, load, store, move, arith, logic, mem, other
    """
    s = instr.strip()
    if not s:
        return "other"
    op = s.split()[0].lower()

    # Branchy
    if op.startswith("j") or op in {"call", "ret", "jmp"}:
        return "branch"

    # Memory touch detection (parentheses in AT&T syntax)
    touches_mem = "(" in s and ")" in s

    # Specific load/store for mov*
    if op.startswith("mov") and touches_mem:
        # Split on comma to get src,dst
        parts = [p.strip() for p in s.split(",", 1)]
        if len(parts) == 2:
            src, dst = parts
            if "(" in src and "(" not in dst:
                return "load"
            if "(" in dst and "(" not in src:
                return "store"
        # If ambiguous, treat as move
        return "move"

    # Other clear memory ops (simple heuristic)
    if op in {"push", "pop", "stos", "lods", "cmps", "scas"}:
        return "mem"

    if touches_mem:
        return "mem"

    # Moves
    if op.startswith("mov") or op == "lea":
        return "move"

    # Arithmetic
    if op.startswith(("add", "sub", "adc", "sbb", "mul", "imul", "div", "idiv", "inc", "dec", "neg")) or \
       op.startswith(("sar","sal","shl","shr","rol","ror")):
        return "arith"

    # Logic / compare
    if op.startswith(("and","or","xor","cmp","test","not")):
        return "logic"

    return "other"

def parse_trace(filename):
    instructions = []
    family_counts = Counter()
    with open(filename, "r") as f:
        for line in f:
            ins = parse_line(line)
            if not ins:
                continue
            instructions.append(ins)
            fam = classify_instruction(ins.instruction)
            family_counts[fam] += 1
    return instructions, family_counts

# -----------------------------
# Block building & jumps
# -----------------------------

PAGE_SEQ_LIMIT = 8  # consider <=8B sequential advance in VA as same basic block (AT&T encoding varies)

def group_blocks(instructions):
    """
    Groups sequential instructions into "basic blocks" by VA monotonic small deltas.
    Each block tuple: (start_va_hex, end_va_hex, block_len, lib_basename, pid)
    """
    if not instructions:
        return []

    blocks = []
    curr = instructions[0]
    start_va = curr.address
    end_va = curr.address
    block_len = 1
    curr_lib = curr.binary.split('/')[-1].rstrip(')')
    curr_pid = curr.pid

    for instr in instructions[1:]:
        next_va = int(instr.address, 16)
        curr_va = int(curr.address, 16)
        delta = abs(next_va - curr_va)

        same_flow = (0 < delta <= PAGE_SEQ_LIMIT)
        if same_flow and instr.pid == curr_pid and (instr.binary.split('/')[-1].rstrip(')') == curr_lib):
            # continue current block when pid/lib consistent
            end_va = instr.address
            block_len += 1
        else:
            # close current
            blocks.append((start_va, end_va, block_len, curr_lib, curr_pid))
            # start new block
            start_va = instr.address
            end_va = instr.address
            block_len = 1
            curr_lib = instr.binary.split('/')[-1].rstrip(')')
            curr_pid = instr.pid

        curr = instr

    blocks.append((start_va, end_va, block_len, curr_lib, curr_pid))
    return blocks

def is_conditional_branch(mnemonic: str) -> bool:
    """True if the instruction is a conditional branch (jz, jne, jge and not callq or retq)."""
    return mnemonic.startswith("j") and mnemonic not in {"jmp", "jmpq", "call", "callq", "ret", "retq"}

def classify_branches(instrs):
    """Return list of tuples (VA, taken_flag) for conditional branches."""
    branches = []
    n = len(instrs)
    if n < 2:
        return branches

    for i in range(n - 1):
        ins = instrs[i]
        next_ins = instrs[i + 1]

        # extract VA and mnemonic
        try:
            va = int(ins.address, 16)
        except:
            continue

        # mnemonic = first token of instruction string
        parts = ins.instruction.split()
        if not parts:
            continue
        mnemonic = parts[0].lower()

        if not is_conditional_branch(mnemonic):
            continue

        # next VA
        try:
            next_va = int(next_ins.address, 16)
        except:
            continue

        # heuristic for branch taken/not taken
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

def plot_cdfs_taken_not_taken(run_dict, title, out_path):
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

def compute_jump_sizes_with_blocklens(blocks):
    """
    Only count jumps where next block shares same pid AND same library (avoid cross-process/lib jumps).
    Returns jump_sizes, block_lens, pairs[(block_len, jump_size)]
    """
    jump_sizes, block_lens, pairs = [], [], []
    for i in range(len(blocks) - 1):
        src = blocks[i]
        dst = blocks[i + 1]
        if src[3] != dst[3] or src[4] != dst[4]:
            continue
        src_len = src[2]
        src_end_va = int(src[1], 16)
        dst_start_va = int(dst[0], 16)
        diff = abs(dst_start_va - src_end_va)
        jump_sizes.append(diff)
        block_lens.append(src_len)
        pairs.append((src_len, diff))
    return jump_sizes, block_lens, pairs

def print_large_jumps(jump_sizes, threshold=0x100):
    large = [(i, j) for i, j in enumerate(jump_sizes) if j > threshold]
    if not large:
        return []
    print(f"\n[+] Found {len(large)} large jumps (> 0x{threshold:x}):")
    for idx, dist in large[:50]:
        print(f"  Jump #{idx}: deltaVA = {dist} ({hex(dist)})")
    return large

# -----------------------------
# Family counts per block (single app)
# -----------------------------

def family_counts_per_block(instructions, blocks):
    """
    For the given instruction list and its blocks, return:
      - per_block_counts: list[Counter(family->count)] aligned with blocks
      - family_lists: dict[family] -> list of integer counts (same order as blocks)
    """
    # Build a fast index of instruction -> family
    fam_by_idx = []
    for ins in instructions:
        fam_by_idx.append(classify_instruction(ins.instruction))

    per_block_counts = []
    lengths = [b[2] for b in blocks]
    start = 0
    for blen in lengths:
        end = start + blen
        c = Counter()
        for i in range(start, end):
            c[fam_by_idx[i]] += 1
        per_block_counts.append(c)
        start = end

    family_lists = defaultdict(list)
    all_fams = sorted({f for c in per_block_counts for f in c})
    for c in per_block_counts:
        for f in all_fams:
            family_lists[f].append(c.get(f, 0))

    return per_block_counts, family_lists

# -----------------------------
# Math helpers & plotting
# -----------------------------

def compute_cdf(data):
    data = np.sort(np.asarray(data))
    n = len(data)
    if n == 0:
        return np.array([]), np.array([])
    y = np.arange(1, n + 1) / n
    return data, y

def plot_jump_cdfs(jump_dict, output_dir, kernel_space):
    # Linear CDF
    plt.figure(figsize=(9, 6))
    for label, jumps in jump_dict.items():
        if not jumps:
            continue
        x, y = compute_cdf(jumps)
        plt.plot(x, y, label=label, linewidth=2)
    plt.xlabel("Absolute Jump Size (VA difference)")
    plt.ylabel("CDF")
    plt.title(f"CDF of Jump/Branch Target Sizes ({'Kernel' if kernel_space else 'User'}-Space)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    path = output_dir / "jump_cdf_abs_full.png"
    plt.savefig(path, dpi=150)
    print(f"Wrote {path}")
    plt.close()

    # Log-X CDF
    plt.figure(figsize=(9, 6))
    for label, jumps in jump_dict.items():
        if not jumps:
            continue
        x, y = compute_cdf(jumps)
        plt.plot(x, y, label=label, linewidth=2)
    plt.xscale("log")
    plt.xlabel("Absolute Jump Size (log scale)")
    plt.ylabel("CDF")
    plt.title(f"CDF of Jump/Branch Target Sizes (log-x, {'Kernel' if kernel_space else 'User'}-Space)")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    path = output_dir / "jump_cdf_abs_logx.png"
    plt.savefig(path, dpi=150)
    print(f"Wrote {path}")
    plt.close()

    # Zoomed tail
    plt.figure(figsize=(9, 6))
    for label, jumps in jump_dict.items():
        if not jumps:
            continue
        x, y = compute_cdf(jumps)
        plt.plot(x, y, label=label, linewidth=2)
    plt.xscale("log")
    plt.ylim(0.8, 1.0)
    plt.yticks([0.8, 0.9, 1.0])
    plt.xlabel("Absolute Jump Size (log scale)")
    plt.ylabel("CDF")
    plt.title(f"Zoomed Upper Tail CDF ({'Kernel' if kernel_space else 'User'}-Space)")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    path = output_dir / "jump_cdf_abs_zoomed.png"
    plt.savefig(path, dpi=150)
    print(f"Wrote {path}")
    plt.close()

# def plot_jump_vs_blocklen_scatter(pairs_by_trace, output_dir, kernel_space):
#     # Combined
#     plt.figure(figsize=(9, 6))
#     for label, pairs in pairs_by_trace.items():
#         if not pairs:
#             continue
#         xs = [p[0] for p in pairs]
#         ys = [p[1] for p in pairs]
#         plt.scatter(xs, ys, s=12, alpha=0.4, label=label)
#     plt.yscale("log")
#     plt.xscale("log")
#     plt.xlabel("Block Length (sequential instruction count)")
#     plt.ylabel("Next Jump Size (abs VA difference, log)")
#     plt.title(f"Jump Size vs. Block Length ({'Kernel' if kernel_space else 'User'}-Space)")
#     plt.grid(True, which="both", linestyle="--", alpha=0.5)
#     plt.legend(markerscale=2, fontsize="small")
#     plt.tight_layout()
#     out = output_dir / "jump_vs_blocklen_all_traces.png"
#     plt.savefig(out, dpi=150)
#     print(f"Wrote {out}")
#     plt.close()

#     # Per-trace
#     for label, pairs in pairs_by_trace.items():
#         if not pairs:
#             continue
#         xs = [p[0] for p in pairs]
#         ys = [p[1] for p in pairs]
#         plt.figure(figsize=(8, 5))
#         plt.scatter(xs, ys, s=14, alpha=0.6)
#         plt.yscale("log")
#         plt.xscale("log")
#         plt.xlabel("Block Length (sequential instruction count)")
#         plt.ylabel("Next Jump Size (abs VA difference, log)")
#         plt.title(f"Jump Size vs. Block Length — {label}")
#         plt.grid(True, which="both", linestyle="--", alpha=0.5)
#         plt.tight_layout()
#         out = output_dir / f"jump_vs_blocklen_{label}.png"
#         plt.savefig(out, dpi=150)
#         print(f"Wrote {out}")
#         plt.close()

# ---- New: Family CDFs (single app) ----

def plot_family_cdfs_single_app(app_label, family_lists, output_dir):
    """
    family_lists: dict[family] -> list[count_per_block]
    Plots multi-curve CDF where each curve is the CDF of that family's counts per basic block.
    """
    plt.figure(figsize=(9, 6))
    for family, counts in sorted(family_lists.items()):
        if not counts:
            continue
        x, y = compute_cdf(counts)
        plt.plot(x, y, linewidth=2, label=family)
    plt.xlabel("Per-Block Count")
    plt.ylabel("CDF")
    plt.title(f"Instruction type Counts per Block — CDFs ({app_label})")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    out = output_dir / f"{app_label}_family_counts_per_block_cdfs.png"
    plt.savefig(out, dpi=150)
    print(f"Wrote {out}")
    plt.close()

# ---- New: Stacked bar for average per-block composition (single app) ----

def plot_family_stacked_bar_single_app(app_label, family_lists, output_dir):
    """
    Build a single stacked bar where each segment is the mean fraction of that family per block.
    """
    families = sorted(family_lists.keys())
    num_blocks = len(next(iter(family_lists.values()))) if families else 0
    if num_blocks == 0:
        print("No blocks for stacked bar.")
        return

    per_family_avg_fraction = {f: 0.0 for f in families}

    for b in range(num_blocks):
        total = sum(family_lists[f][b] for f in families)
        if total == 0:
            continue
        for f in families:
            per_family_avg_fraction[f] += family_lists[f][b] / total

    for f in families:
        per_family_avg_fraction[f] /= max(1, num_blocks)

    plt.figure(figsize=(7, 5))
    bottom = 0.0
    x = [0]
    for f in families:
        h = per_family_avg_fraction[f]
        plt.bar(x, [h], bottom=bottom, label=f, width=0.5)
        bottom += h

    plt.xticks([0], [app_label])
    plt.ylim(0, 1)
    plt.ylabel("Average fraction per block")
    plt.title(f"Average Instruction type distribution per Block — {app_label}")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    plt.tight_layout()
    out = output_dir / f"{app_label}_family_composition_stacked_bar.png"
    plt.savefig(out, dpi=150)
    print(f"Wrote {out}")
    plt.close()

def write_family_counts_csv_single_app(app_label, per_block_counts, output_dir):
    families = sorted({f for c in per_block_counts for f in c})
    path = output_dir / f"{app_label}_family_counts_per_block.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Block_Index"] + families)
        for i, c in enumerate(per_block_counts):
            w.writerow([i] + [c.get(f, 0) for f in families])
    print(f"Wrote {path}")

# -----------------------------
# CSV helpers
# -----------------------------

def write_pairs_csv(pairs_by_trace, output_dir):
    combined_rows = []
    for label, pairs in pairs_by_trace.items():
        if not pairs:
            continue
        per_path = output_dir / f"jump_vs_blocklen_{label}.csv"
        with open(per_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Block_Index", "Block_Len", "Next_Jump_Size"])
            for i, (blen, jsize) in enumerate(pairs):
                w.writerow([i, blen, jsize])
                combined_rows.append([label, i, blen, jsize])
        print(f"Wrote {per_path}")

    if combined_rows:
        all_path = output_dir / "jump_vs_blocklen_all.csv"
        with open(all_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Trace", "Block_Index", "Block_Len", "Next_Jump_Size"])
            w.writerows(combined_rows)
        print(f"Wrote {all_path}")


def write_frequency_distributions_csv(datasets: Dict[str, List[float]], output_dir: Path):
    """
    Write a single CSV containing percentile values (P10, P20, ..., P100) for each dataset.

    The CSV header will be: "block size:", P10, P20, P30, P40, P50, P60, P70, P80, P90, P100
    Each row: label, P10_value, P20_value, ..., P100_value

    The values are actual percentiles without normalization.
    """
    if not datasets:
        print("No datasets provided for frequency distributions.")
        return {}

    header_percentiles = [f"P{i}" for i in range(10, 101, 10)]  # P10, P20, ..., P100
    percentile_values = list(range(10, 101, 10))  # [10, 20, ..., 100]

    out_path = output_dir / "frequency_distributions.csv"
    freqs_out = {}
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        # First cell label to match requested format
        w.writerow(["block size:"] + header_percentiles)

        for label, values in sorted(datasets.items()):
            if not values:
                freqs = [0.0 for _ in range(len(percentile_values))]
            else:
                freqs = np.percentile(np.array(values), percentile_values)

            freqs_out[label] = freqs
            # Write row: label + percentile values with no decimal places (actual counts)
            w.writerow([label] + [int(x) for x in freqs])

    print(f"Wrote combined frequency distributions CSV: {out_path}")
    return freqs_out

# ============================================================
# NEW SECTION: RAW dependency distance computation & plotting
# ============================================================

# --- Register canonicalization ---

_alias_groups = [
    ("rax", ["rax","eax","ax","al","ah"]),
    ("rbx", ["rbx","ebx","bx","bl","bh"]),
    ("rcx", ["rcx","ecx","cx","cl","ch"]),
    ("rdx", ["rdx","edx","dx","dl","dh"]),
    ("rsi", ["rsi","esi","si","sil"]),
    ("rdi", ["rdi","edi","di","dil"]),
    ("rbp", ["rbp","ebp","bp","bpl"]),
    ("rsp", ["rsp","esp","sp","spl"]),
    ("rip", ["rip","eip","ip"]),
]
_alias_map: Dict[str,str] = {}
for base, names in _alias_groups:
    for n in names:
        _alias_map[n] = base
for i in range(8,16):
    base = f"r{i}"
    _alias_map[base] = base
    _alias_map[f"e{i}"] = base
    _alias_map[f"{base}d"] = base
    _alias_map[f"{base}w"] = base
    _alias_map[f"{base}b"] = base

def canonical_reg(reg: str) -> Optional[str]:
    """
    Return canonical architectural register name for RAW tracking.
    Keeps xmm/ymm/zmm/k as-is (lowercased).
    Returns None for weird or segment/debug regs we choose to ignore.
    """
    if not reg:
        return None
    r = reg.lower().lstrip('%')
    # vector/scalar/special we keep:
    if r.startswith(("xmm","ymm","zmm","k")):
        return r
    # gp regs
    if r in _alias_map:
        return _alias_map[r]
    # common segment/base/index regs we generally ignore for RAW (fs/gs)
    if r in {"cs","ds","es","fs","gs","ss"}:
        return None
    # mmx
    if r.startswith("mm"):
        return r
    # mask CR/DR/Txx
    if r.startswith(("cr","dr","tr")):
        return None
    # default: return the thing (conservative)
    return r

_reg_re = re.compile(r"%[a-zA-Z][a-zA-Z0-9]*")

def extract_registers(text: str) -> List[str]:
    return [m.group(0) for m in _reg_re.finditer(text)]

def split_operands(instr_text: str) -> Tuple[str, List[str]]:
    """
    Return opcode and operand list (naive comma split; good enough for AT&T + perf script).
    """
    s = instr_text.strip()
    if not s:
        return "", []
    parts = s.split(None, 1)
    if len(parts) == 1:
        return parts[0].lower(), []
    op = parts[0].lower()
    ops = [o.strip() for o in parts[1].split(",")]
    return op, ops

def dest_is_register(dst: str) -> Optional[str]:
    """
    If the destination operand is a register (not memory), return its canonical name.
    """
    if "(" in dst and ")" in dst:
        return None
    regs = _reg_re.findall(dst)
    if not regs:
        return None
    # If multiple (rare), take the last token as the dest reg
    return canonical_reg(regs[-1])

def regs_in_operand_as_reads(opnd: str) -> Set[str]:
    """
    All registers appearing in an operand are *read* for addressing/source purposes.
    """
    regs = set()
    for r in extract_registers(opnd):
        cr = canonical_reg(r)
        if cr:
            regs.add(cr)
    return regs

def reads_writes_for_instruction(instr_text: str) -> Tuple[Set[str], Set[str]]:
    """
    Return (reads, writes) sets of canonical registers for one instruction in AT&T syntax.
    Heuristic but covers common cases.
    """
    op, ops = split_operands(instr_text)
    op = op.lower()
    reads: Set[str] = set()
    writes: Set[str] = set()

    # zero-operand
    if not ops:
        # treat as no reg read/write (e.g., nop, cpuid handled poorly; okay)
        return reads, writes

    # helpers
    def rw_binop():
        # typical two-operand: src, dst
        for o in ops:
            reads.update(regs_in_operand_as_reads(o))
        d = dest_is_register(ops[-1])
        if d:
            writes.add(d)

    def rw_unop_writes():
        # single-operand operations that read+write the operand
        reads.update(regs_in_operand_as_reads(ops[0]))
        d = dest_is_register(ops[0])
        if d:
            writes.add(d)

    # Categories
    if op.startswith("cmov"):
        # cmovcc src, dst : read src (+flags, ignored), write dst
        for o in ops:
            reads.update(regs_in_operand_as_reads(o))
        d = dest_is_register(ops[-1])
        if d:
            writes.add(d)
        return reads, writes

    if op in {"cmp","test"}:
        for o in ops:
            reads.update(regs_in_operand_as_reads(o))
        return reads, writes

    if op.startswith("mov"):
        # mov src, dst
        for o in ops:
            reads.update(regs_in_operand_as_reads(o))
        d = dest_is_register(ops[-1])
        if d:
            writes.add(d)
        return reads, writes

    if op == "lea":
        # lea src, dst : reads addressing regs in src, writes dst
        reads.update(regs_in_operand_as_reads(ops[0]))
        d = dest_is_register(ops[-1])
        if d:
            writes.add(d)
        return reads, writes

    if op in {"add","sub","adc","sbb","and","or","xor","xadd"} or \
       op in {"shl","shr","sal","sar","rol","ror"} or \
       op in {"imul"}:
        rw_binop()
        return reads, writes

    if op in {"inc","dec","neg","not"}:
        rw_unop_writes()
        return reads, writes

    if op == "push":
        # reads operand (+rsp), writes rsp
        reads.update(regs_in_operand_as_reads(ops[0]))
        reads.add("rsp")
        writes.add("rsp")
        return reads, writes

    if op == "pop":
        # reads rsp, writes dst (+rsp)
        reads.add("rsp")
        d = dest_is_register(ops[0])
        if d:
            writes.add(d)
        writes.add("rsp")
        return reads, writes

    if op in {"call","callq"}:
        # possible indirect register read; rsp read+write
        for o in ops:
            reads.update(regs_in_operand_as_reads(o))
        reads.add("rsp")
        writes.add("rsp")
        # rip write ignored
        return reads, writes

    if op in {"ret","retq"}:
        reads.add("rsp")
        writes.add("rsp")
        return reads, writes

    if op.startswith("j"):  # jmp/jcc
        # handle jmp *%reg as read
        for o in ops:
            reads.update(regs_in_operand_as_reads(o))
        return reads, writes

    # default: conservative two-operand attempt or read-only
    if len(ops) == 2:
        rw_binop()
    else:
        for o in ops:
            reads.update(regs_in_operand_as_reads(o))
    return reads, writes

def raw_dependency_distances(instructions: List[Instruction]) -> List[int]:
    """
    Compute RAW distances (in number of intervening dynamic instructions)
    per trace stream. Distance for a read of register R is i - last_write[R],
    if a previous write to R exists. Resets tracking across PID/binary changes.
    """
    dists: List[int] = []
    last_write: Dict[str, int] = {}
    prev_pid = None
    prev_lib = None

    for i, ins in enumerate(instructions):
        # Reset when context changes (avoid cross-PID/lib linking)
        curr_pid = ins.pid
        curr_lib = ins.binary
        if prev_pid is None:
            prev_pid, prev_lib = curr_pid, curr_lib
        elif curr_pid != prev_pid or curr_lib != prev_lib:
            last_write.clear()
            prev_pid, prev_lib = curr_pid, curr_lib

        reads, writes = reads_writes_for_instruction(ins.instruction)

        # record distances for each read that has a prior write
        for r in reads:
            if r in last_write:
                d = i - last_write[r]
                if d > 0:
                    dists.append(d)

        # update last write indices
        for w in writes:
            last_write[w] = i

    return dists

def plot_raw_cdfs(raw_dict: Dict[str, List[int]], output_dir: Path):
    """
    Plot CDF of RAW dependency distances for each trace.
    """
    if not raw_dict:
        print("No RAW distances to plot.")
        return

    # Linear x
    plt.figure(figsize=(9, 6))
    for label, dists in raw_dict.items():
        if not dists:
            continue
        x, y = compute_cdf(dists)
        plt.plot(x, y, linewidth=2, label=label)
    plt.xlabel("RAW Dependency Distance (instructions)")
    plt.ylabel("CDF")
    plt.title("CDF of Read-After-Write (RAW) Dependency Distances")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    out = output_dir / "raw_dep_cdf_full.png"
    plt.savefig(out, dpi=150)
    print(f"Wrote {out}")
    plt.close()

    # Log-x
    plt.figure(figsize=(9, 6))
    for label, dists in raw_dict.items():
        if not dists:
            continue
        x, y = compute_cdf(dists)
        plt.plot(x, y, linewidth=2, label=label)
    plt.xscale("log")
    plt.xlabel("RAW Dependency Distance (log scale)")
    plt.ylabel("CDF")
    plt.title("CDF of Read-After-Write (RAW) Dependency Distances — log-x")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    out = output_dir / "raw_dep_cdf_logx.png"
    plt.savefig(out, dpi=150)
    print(f"Wrote {out}")
    plt.close()

    # Zoomed tail
    plt.figure(figsize=(9, 6))
    for label, dists in raw_dict.items():
        if not dists:
            continue
        x, y = compute_cdf(dists)
        plt.plot(x, y, linewidth=2, label=label)
    plt.xscale("log")
    plt.ylim(0.8, 1.0)
    plt.yticks([0.8, 0.9, 1.0])
    plt.xlabel("RAW Dependency Distance (log scale)")
    plt.ylabel("CDF")
    plt.title("RAW Dependency Distance — Zoomed Upper Tail")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    out = output_dir / "raw_dep_cdf_zoomed.png"
    plt.savefig(out, dpi=150)
    print(f"Wrote {out}")
    plt.close()

def plot_cdfs_block_len(block_dict):
    # --- Full CDF ---
    plt.figure(figsize=(9, 6))
    for label, sizes in block_dict.items():
        if not sizes:
            continue
        x, y = compute_cdf(sizes)
        plt.plot(x, y, label=label, linewidth=2)
    # plt.xscale("log")
    plt.xlabel("Basic Block Size (instructions)")
    plt.ylabel("CDF")
    plt.title("CDF of Basic Block Sizes across User-Space Execution") # change to kernel
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()
    full_path = OUT_DIR / "block_cdf_full.png"
    plt.savefig(full_path, dpi=150)
    plt.close()
    print(f"Wrote {full_path}")

    # --- Full CDF log ---
    plt.figure(figsize=(9, 6))
    for label, sizes in block_dict.items():
        if not sizes:
            continue
        x, y = compute_cdf(sizes)
        plt.plot(x, y, label=label, linewidth=2)
    plt.xscale("log")
    plt.xlabel("Basic Block Size (instructions, log scale)")
    plt.ylabel("CDF")
    plt.title("CDF of Basic Block Sizes across User-Space Execution") # change to kernel
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()
    full_path = OUT_DIR / "block_cdf_full_log.png"
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
    plt.xscale("log")
    plt.xlabel("Basic Block Size (instructions, log scale)")
    plt.ylabel("CDF")
    plt.title("CDF of Basic Block Sizes across User-Space Execution (zoomed)") # change to kernel    
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.ylim(0.8, 1.0)
    plt.yticks([0.8, 0.9, 1.0])
    plt.tight_layout()
    zoom_path = OUT_DIR / "block_cdf_zoomed.png"
    plt.savefig(zoom_path, dpi=150)
    plt.close()
    print(f"Wrote {zoom_path}")


def write_raw_csv(raw_dict: Dict[str, List[int]], output_dir: Path):
    for label, dists in raw_dict.items():
        path = output_dir / f"raw_dep_distances_{label}.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Index", "RAW_Distance"])
            for i, d in enumerate(dists):
                w.writerow([i, d])
        print(f"Wrote {path}")

# -----------------------------
# Main
# -----------------------------

def write_cdf_percentiles_csv(jump_dict: Dict[str, List[int]], output_dir: Path):
    """
    For each application/trace, compute CDF percentiles (0.1 .. 1.0)
    using the same values plotted in the CDF curves.
    Writes one CSV per app.
    """
    percentiles = [i / 10.0 for i in range(1, 11)]  # 0.1, 0.2, ... 1.0

    for label, jumps in jump_dict.items():
        if not jumps:
            continue

        data = np.sort(np.asarray(jumps))
        n = len(data)
        if n == 0:
            continue

        # compute percentile values
        pct_vals = []
        for p in percentiles:
            # percentile index using CDF convention (i/n)
            idx = int(np.ceil(p * n)) - 1
            idx = max(0, min(idx, n - 1))
            pct_vals.append(data[idx])

        out = output_dir / f"{label}_jump_cdf_percentiles.csv"
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Percentile", "Value"])
            for p, v in zip(percentiles, pct_vals):
                w.writerow([p, v])

        print(f"Wrote percentile CDF CSV for {label}: {out}")

def main():
    ap = argparse.ArgumentParser(description="Jump CDFs, scatter plots, single-app family analytics, and RAW dependency distances")
    ap.add_argument("-i", "--inputPath", required=True, help="Input trace directory")
    ap.add_argument("-o", "--outputPath", required=True, help="Output directory")
    ap.add_argument("-k", "--kernelSpace", action="store_true", help="Kernel-space trace flag")
    ap.add_argument("--familyApp", help="Single app label (filename stem) to plot family CDFs & stacked bar")
    ap.add_argument("--rawCSV", action="store_true", help="Also write per-trace RAW distance CSVs")
    args = ap.parse_args()

    input_dir = Path(args.inputPath)
    output_dir = Path(args.outputPath)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Pass 1: parse all traces, build blocks, compute jump stats ---
    jump_dict: Dict[str, List[int]] = {}
    pairs_by_trace: Dict[str, List[Tuple[int,int]]] = {}
    all_large_jumps = []

    # For RAW distances
    raw_dict: Dict[str, List[int]] = {}

    # Keep parsed content for single-app processing
    single_app_instrs = None
    single_app_blocks = None
    block_dict = {}
    taken_dict = {}
    not_taken_dict = {}

    for trace_path in sorted(input_dir.iterdir()):
        if not trace_path.is_file():
            continue
        label = trace_path.stem
        instrs, _fam_counts_total = parse_trace(trace_path)
        blocks = group_blocks(instrs)
        # accumulate per-block family counts across all traces
        try:
            per_block_counts_trace, family_lists_trace = family_counts_per_block(instrs, blocks)
        except Exception:
            per_block_counts_trace, family_lists_trace = [], {}

        # initialize global accumulator on first use
        if 'global_family_lists' not in locals():
            global_family_lists = defaultdict(list)

        for fam, lst in family_lists_trace.items():
            global_family_lists[fam].extend(lst)
        block_dict[trace_path.stem] = [b[2] for b in blocks]

        jumps, blens, pairs = compute_jump_sizes_with_blocklens(blocks)
        jump_dict[label] = jumps
        pairs_by_trace[label] = pairs

        branches = classify_branches(instrs)
        taken_runs, not_taken_runs = compute_runs(branches)
        taken_dict[trace_path.stem] = taken_runs
        not_taken_dict[trace_path.stem] = not_taken_runs

        # RAW distances for this stream (resets when PID/lib changes internally)
        raw_d = raw_dependency_distances(instrs)
        raw_dict[label] = raw_d

        print(f"{trace_path.name}: {len(instrs)} instructions, {len(blocks)} blocks, {len(jumps)} filtered jumps, {len(raw_d)} RAW reads")

        large = print_large_jumps(jumps, threshold=10000)
        all_large_jumps.extend([(label, j) for _, j in large])

        if args.familyApp and label == args.familyApp:
            single_app_instrs = instrs
            single_app_blocks = blocks

    # block len / taken/not-taken / jump / raw / family datasets
    # Instead of plotting, build a combined dataset map and write a single CSV of frequency distributions.
    combined_datasets: Dict[str, List[float]] = {}

    # block sizes
    for label, sizes in block_dict.items():
        print(label)
        print(np.percentile(np.array(sizes), [10,20,30,40,50,60,70,80,90,100]))
        combined_datasets[f"block::{label}"] = sizes

    # taken / not-taken run lengths
    for label, sizes in taken_dict.items():
        combined_datasets[f"taken_runs::{label}"] = sizes
    for label, sizes in not_taken_dict.items():
        combined_datasets[f"not_taken_runs::{label}"] = sizes

    # jumps
    for label, js in jump_dict.items():
        combined_datasets[f"jumps::{label}"] = js

    # RAW dependency distances
    for label, rd in raw_dict.items():
        combined_datasets[f"raw::{label}"] = rd

    # If familyApp requested, include per-family counts as separate rows
    # Include aggregated family lists across all traces
    if 'global_family_lists' in locals():
        for fam, lst in global_family_lists.items():
            combined_datasets[f"family::{fam}"] = lst

    # Also include single-app family breakdown if requested
    if args.familyApp and single_app_instrs and single_app_blocks:
        per_block_counts, family_lists = family_counts_per_block(single_app_instrs, single_app_blocks)
        for fam, lst in family_lists.items():
            combined_datasets[f"family::{args.familyApp}::{fam}"] = lst

    # Write single combined CSV of frequency distributions and get computed freqs
    freqs_map = write_frequency_distributions_csv(combined_datasets, output_dir)

    # Save large jumps summary
    if all_large_jumps:
        csv_path = output_dir / "large_jumps_over_100.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Trace", "Jump_Size"])
            writer.writerows(all_large_jumps)
        print(f"\n[+] Saved all large jumps (>100) to {csv_path}")

    # --- Previously plotted outputs ---
    if pairs_by_trace:
        write_pairs_csv(pairs_by_trace, output_dir)
    else:
        print("No (block_len, jump_size) pairs to write.")

    # Optional: write raw CSVs if requested
    if args.rawCSV and raw_dict:
        write_raw_csv(raw_dict, output_dir)
        write_percentile_table("raw_cdf", raw_dict, output_dir)

    # --- Single app family analytics ---
    if args.familyApp:
        if single_app_instrs is None or single_app_blocks is None:
            # Try to pick a file matching (case-insensitive contains) if exact stem not found
            fallback = None
            for p in sorted(input_dir.iterdir()):
                if p.is_file() and args.familyApp.lower() in p.stem.lower():
                    fallback = p
                    break
            if fallback:
                print(f"[familyApp] Exact stem '{args.familyApp}' not found; using '{fallback.stem}' as fallback.")
                single_app_instrs, _ = parse_trace(fallback)
                single_app_blocks = group_blocks(single_app_instrs)

        if single_app_instrs and single_app_blocks:
                per_block_counts, family_lists = family_counts_per_block(single_app_instrs, single_app_blocks)
                write_family_counts_csv_single_app(args.familyApp, per_block_counts, output_dir)
                # family lists were included earlier into the combined distributions CSV
                write_percentile_table("instruction_cdf", family_lists, output_dir)
        else:
            print(f"[familyApp] No matching trace for '{args.familyApp}'. Skipping family plots.")

if __name__ == "__main__":
    main()
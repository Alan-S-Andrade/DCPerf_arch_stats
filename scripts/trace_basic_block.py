#!/usr/bin/env python3
"""
trace_basic_block_fixed.py

Parse perf-style instruction trace lines and group into basic blocks using the
strict PC vs symbol-offset equality heuristic.

Heuristic:
  Two consecutive instructions prev -> cur are sequential (same basic block)
  iff ALL hold:
    - prev.object == cur.object
    - normalized(prev.symbol) == normalized(cur.symbol)
    - prev.offset is not None and cur.offset is not None
    - (cur.offset - prev.offset) == (cur.pc - prev.pc)

This script aims to be robust to symbol forms like:
  ICacheBuster::RunNextMethod()+0x95
  ICacheBuster::RunNextMethod()@plt+0x0
  foo(int)::bar+0x12
and to optional "insn:" bytes at end of line.

Usage:
    python3 trace_basic_block_fixed.py dummy.data
    python3 trace_basic_block_fixed.py dummy.data -o blocks.csv
"""

from __future__ import annotations
import re
import argparse
import csv
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

# Regex: tolerant extraction
# We search for:
#   ... <pc_hex> <symbol-and-possible-suffix>+0x<off> (<object>) [insn: ...]
# This uses non-greedy matching for symbol part so we capture the full symbol even
# when it contains spaces or other characters.
LINE_RE = re.compile(
    r"""
    ^\s*
    (?P<exe>\S+)\s+                 # executable/task
    (?P<pid>\d+)\s+                 # pid
    \[(?P<cpu>\d+)\]\s+             # [cpu]
    (?P<ts>[\d\.]+):\s+             # timestamp:
    (?P<pc>[0-9A-Fa-f]+)\s+         # virtual address (hex, no 0x)
    (?P<sympart>.+?)                # symbol part (non-greedy) - we will extract +0x from it or later
    \s*\(\s*(?P<object>[^)]+)\s*\)  # (object)
    (?:\s+insn:\s*(?P<insn>.*))?    # optional instruction bytes
    """,
    re.VERBOSE,
)

# within sympart we expect something like:
#   SYMBOL_NAME()+0x95
#   SYMBOL_NAME@plt+0x0
#   SYMBOL_NAME+0x12
SYM_OFF_RE = re.compile(r'^(?P<sym>.+?)\s*(?:@plt)?\s*\+\s*0x(?P<off>[0-9A-Fa-f]+)\s*$')

def normalize_symbol(sym: str) -> str:
    """Normalize symbol: strip @plt, trailing decorations and whitespace."""
    s = sym.strip()
    # remove @plt, @plt+..., trailing spaces
    s = re.sub(r'@plt(?:\+0x[0-9A-Fa-f]+)?', '', s)
    # remove trailing parentheses or templates if needed, keep core name
    # but don't be too aggressive — keep most of the symbol so same symbols match
    return s.strip()

@dataclass
class Instr:
    original_line: str
    pc: int
    pc_hex: str
    symbol_raw: str
    symbol: str        # normalized
    offset: Optional[int]
    object: str
    exe: str
    pid: int
    cpu: int
    ts: str
    insn: Optional[str]

@dataclass
class BasicBlock:
    start_pc_hex: str
    end_pc_hex: str
    symbol: str
    object: str
    instr_count: int
    lines: List[Instr]

def parse_line(line: str) -> Optional[Instr]:
    m = LINE_RE.match(line.rstrip("\n"))
    if not m:
        return None
    exe = m.group("exe")
    pid = int(m.group("pid"))
    cpu = int(m.group("cpu"))
    ts = m.group("ts")
    pc_text = m.group("pc")
    try:
        pc = int(pc_text, 16)
    except ValueError:
        return None
    pc_hex = "0x" + pc_text.lower()
    sympart = m.group("sympart").strip()
    obj = m.group("object").strip()
    insn = m.group("insn").strip() if m.group("insn") else None

    # Try to extract symbol and offset from sympart
    off = None
    sym_raw = sympart
    mo = SYM_OFF_RE.search(sympart)
    if mo:
        sym_raw = mo.group("sym").strip()
        off_text = mo.group("off")
        try:
            off = int(off_text, 16)
        except Exception:
            off = None
    else:
        # fallback: try to find last +0xNN pattern anywhere
        m2 = re.search(r'\+0x([0-9A-Fa-f]+)', sympart)
        if m2:
            off_text = m2.group(1)
            try:
                off = int(off_text, 16)
            except Exception:
                off = None
            # strip the +0x... suffix to get symbol raw
            sym_raw = re.sub(r'\+0x[0-9A-Fa-f]+', '', sympart).strip()

    symbol_norm = normalize_symbol(sym_raw)
    return Instr(
        original_line=line.rstrip("\n"),
        pc=pc,
        pc_hex=pc_hex,
        symbol_raw=sym_raw,
        symbol=symbol_norm,
        offset=off,
        object=obj,
        exe=exe,
        pid=pid,
        cpu=cpu,
        ts=ts,
        insn=insn,
    )

def read_trace(path: Path) -> List[Instr]:
    instrs: List[Instr] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for ln in f:
            parsed = parse_line(ln)
            if parsed:
                instrs.append(parsed)
    return instrs

def group_blocks_strict(instrs: List[Instr]) -> List[BasicBlock]:
    blocks: List[BasicBlock] = []
    if not instrs:
        return blocks

    cur = instrs[0]
    cur_block = BasicBlock(
        start_pc_hex=cur.pc_hex,
        end_pc_hex=cur.pc_hex,
        symbol=cur.symbol,
        object=cur.object,
        instr_count=1,
        lines=[cur],
    )

    for nxt in instrs[1:]:
        sequential = False

        # only sequential if same object and same symbol and both offsets present
        if (nxt.object == cur.object) and (nxt.symbol == cur.symbol):
            if (cur.offset is not None) and (nxt.offset is not None):
                pc_delta = nxt.pc - cur.pc
                off_delta = nxt.offset - cur.offset
                if pc_delta == off_delta and pc_delta >= 0:
                    sequential = True

        if sequential:
            # extend
            cur_block.end_pc_hex = nxt.pc_hex
            cur_block.instr_count += 1
            cur_block.lines.append(nxt)
        else:
            # close block and start new
            blocks.append(cur_block)
            cur_block = BasicBlock(
                start_pc_hex=nxt.pc_hex,
                end_pc_hex=nxt.pc_hex,
                symbol=nxt.symbol,
                object=nxt.object,
                instr_count=1,
                lines=[nxt],
            )

        cur = nxt

    blocks.append(cur_block)
    return blocks

def print_summary(blocks: List[BasicBlock]) -> None:
    total = sum(b.instr_count for b in blocks)
    print(f"Parsed {total} instructions into {len(blocks)} basic blocks.\n")
    print("Index | Instrs | Symbol (object) | start_pc - end_pc")
    for i, b in enumerate(blocks, 1):
        print(f"{i:5d} | {b.instr_count:6d} | {b.symbol} ({b.object}) | {b.start_pc_hex} - {b.end_pc_hex}")

def write_csv(blocks: List[BasicBlock], out: Path) -> None:
    with out.open("w", newline="", encoding="utf-8") as outf:
        w = csv.writer(outf)
        w.writerow(["block_index","instr_count","symbol","object","start_pc","end_pc","first_line","last_line"])
        for idx, b in enumerate(blocks):
            first = b.lines[0].original_line if b.lines else ""
            last = b.lines[-1].original_line if b.lines else ""
            w.writerow([idx, b.instr_count, b.symbol, b.object, b.start_pc_hex, b.end_pc_hex, first, last])

def main():
    ap = argparse.ArgumentParser(description="Group trace into basic blocks using strict PC vs symbol-offset heuristic.")
    ap.add_argument("trace", help="trace file path")
    ap.add_argument("-o","--out-csv", help="output CSV file path", type=Path, default=None)
    args = ap.parse_args()

    p = Path(args.trace)
    if not p.exists():
        print(f"Error: file {p} not found.")
        return

    instrs = read_trace(p)
    if not instrs:
        print("No valid instruction lines parsed.")
        return

    blocks = group_blocks_strict(instrs)
    print_summary(blocks)
    if args.out_csv:
        write_csv(blocks, args.out_csv)
        print(f"\nWrote {len(blocks)} blocks to {args.out_csv}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Disassemble Intel PT `.data` files into instruction-trace text using perf script.

For each `*.data` file found under --input-dir (non-recursive by default), this
script runs:

  perf script --insn-trace --xed -i <file>

and writes the output with lines containing the word "perf" removed to
`<basename>_dis.data` in the output directory.

Usage examples:

  python3 disassemble_pt_data.py --input-dir ./pt_records --output-dir ./pt_dis
  python3 disassemble_pt_data.py --input-dir ./pt_records --recursive

This avoids using a shell pipeline by filtering in Python for portability.
"""

import argparse
import subprocess
from pathlib import Path
import sys
import shlex


def process_file(input_path: Path, output_path: Path) -> bool:
    # Escape parentheses in the input path for shell
    escaped_input = str(input_path).replace('(', r'\(').replace(')', r'\)')
    escaped_output = str(output_path).replace('(', r'\(').replace(')', r'\)')
    
    # Use timeout command with shell redirection
    cmd = f"timeout 4s perf script --insn-trace --xed -i {escaped_input} > {escaped_output}"

    try:
        print(f"Running: {cmd}")
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=4)

        if proc.returncode != 0:
            # Save stderr for diagnosis
            err_path = output_path.with_suffix(output_path.suffix + ".error")
            with open(err_path, "w") as ef:
                ef.write(proc.stderr or "")
            print(f"perf script failed for {input_path} (code {proc.returncode}), stderr -> {err_path}")
            return False

        # Read the output file and filter out lines containing 'perf'
        try:
            with open(output_path, "r") as f:
                out_lines = [line for line in f if 'perf' not in line]
            
            with open(output_path, "w") as f:
                f.write("\n".join(out_lines))
            
            print(f"Wrote {output_path}")
            return True
        except FileNotFoundError:
            print(f"Output file not created: {output_path}")
            return False

    except subprocess.TimeoutExpired:
        print(f"Command execution timed out for {input_path}")
        return False
    except FileNotFoundError:
        print("perf not found in PATH. Install perf or adjust PATH.")
        return False
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def main():
    p = argparse.ArgumentParser(description="Disassemble Intel PT .data files to XED instruction traces")
    p.add_argument("--input-dir", default="./pt_records", help="Directory containing .data files")
    p.add_argument("--output-dir", default="./pt_dis", help="Directory to write disassembly outputs")
    p.add_argument("--recursive", action="store_true", help="Search input-dir recursively for .data files")
    p.add_argument("--pattern", default="*.data", help="Filename glob pattern to match data files")
    p.add_argument("--dry-run", action="store_true", help="Print files that would be processed, do not run perf")

    args = p.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    if args.recursive:
        files = list(input_dir.rglob(args.pattern))
    else:
        files = list(input_dir.glob(args.pattern))

    if not files:
        print("No .data files found to process.")
        return

    for f in sorted(files):
        # Skip files that already look like disassembled outputs
        if f.name.endswith("_dis.data") or f.name.endswith(".error"):
            continue

        out_name = f.stem + "_dis.data"
        out_path = out_dir / out_name

        if args.dry_run:
            print(f"DRY-RUN: {f} -> {out_path}")
            continue

        ok = process_file(f, out_path)
        # if not ok:
        #     print(f"Failed to process: {f}")


if __name__ == '__main__':
    main()

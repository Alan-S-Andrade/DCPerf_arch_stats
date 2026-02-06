#!/usr/bin/env python3
"""
Disassemble Intel PT `.data` files into instruction-trace text using perf script.

Takes a parent directory with subdirectories for each binary. Each binary 
subdirectory contains multiple microbenchmark `.data` files. The script:

  1. Iterates through each subdirectory (binary)
  2. Finds all `.data` files (skips `.txt` files)
  3. For each `.data` file, runs:
     
     perf script --insn-trace --xed -i <file>
  
  4. Filters lines containing "perf" from output and writes to `<basename>_dis.data`
  5. Deletes the source `.data` file after successful disassembly

Usage example:

  python3 dissasemble.py --parent-dir ./pt_records --output-dir ./pt_dis

This avoids using a shell pipeline by filtering in Python for portability.
"""

import argparse
import subprocess
from pathlib import Path
import sys
import shlex


def process_file(input_path: Path, output_path: Path) -> bool:
    # Properly quote paths for shell execution to handle spaces and special characters
    escaped_input = shlex.quote(str(input_path))
    escaped_output = shlex.quote(str(output_path))
    
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
            
            # Delete the source .data file after successful disassembly
            try:
                input_path.unlink()
                print(f"Deleted source file: {input_path}")
            except Exception as e:
                print(f"Warning: Failed to delete {input_path}: {e}")
            
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
    p = argparse.ArgumentParser(description="Disassemble Intel PT .data files from binary subdirectories")
    p.add_argument("--parent-dir", default="./pt_records", help="Parent directory containing subdirectories for each binary")
    p.add_argument("--output-dir", default="./pt_dis", help="Directory to write disassembly outputs")
    p.add_argument("--dry-run", action="store_true", help="Print files that would be processed, do not run perf")

    args = p.parse_args()

    parent_dir = Path(args.parent_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not parent_dir.exists():
        print(f"Parent directory does not exist: {parent_dir}")
        sys.exit(1)

    # Iterate through subdirectories (one per binary)
    binary_dirs = [d for d in parent_dir.iterdir() if d.is_dir()]
    
    if not binary_dirs:
        print(f"No subdirectories found in {parent_dir}")
        return

    total_files = 0
    for binary_dir in sorted(binary_dirs):
        print(f"\nProcessing binary directory: {binary_dir.name}")
        
        # Find all .data files in this binary's directory, skip .txt files
        data_files = [f for f in binary_dir.glob("*.data") if f.is_file()]
        
        if not data_files:
            print(f"  No .data files found in {binary_dir}")
            continue
        
        print(f"  Found {len(data_files)} .data file(s)")
        total_files += len(data_files)
        
        for f in sorted(data_files):
            # Skip files that already look like disassembled outputs
            if f.name.endswith("_dis.data") or f.name.endswith(".error"):
                continue

            out_name = f.stem + "_dis.data"
            out_path = out_dir / out_name
            
            # Check if the disassembled output already exists
            if out_path.exists():
                file_size = out_path.stat().st_size
                if file_size > 0:
                    print(f"  SKIP: {out_path} already exists with {file_size} bytes")
                    continue
                else:
                    # Delete empty file and proceed
                    try:
                        out_path.unlink()
                        print(f"  Deleted empty file: {out_path}")
                    except Exception as e:
                        print(f"  Warning: Failed to delete empty {out_path}: {e}")

            if args.dry_run:
                print(f"  DRY-RUN: {f} -> {out_path}")
                continue

            ok = process_file(f, out_path)
            # if not ok:
            #     print(f"  Failed to process: {f}")
    
    if total_files == 0:
        print("No .data files found to process in any subdirectory.")



if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import sys
from pathlib import Path

if (len(sys.argv) < 2):
    print("Usage: python trim_trace_data.py <file_to_remove_perf_instructions>")

INPUT_TRACE = Path(sys.argv[1])
TRIMMED_FILE = INPUT_TRACE.name + '_trimmed'

with open(INPUT_TRACE, 'r') as infile, open(TRIMMED_FILE, 'w') as outfile:
    for line in infile:
        if "perf" not in line.strip():
            outfile.write(line)
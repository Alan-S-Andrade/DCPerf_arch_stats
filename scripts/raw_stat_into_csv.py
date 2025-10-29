#!/usr/bin/env python3
import csv
import re
import sys

if len(sys.argv) != 3:
    print(f"usage: {sys.argv[0]} <input_file> <output_file.csv>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]
counters = []

# match perf output lines
line_re = re.compile(r"^\s*([\d,<not supported>]+)\s+([\w\.\-:]+)")

with open(input_file, "r") as f:
    for line in f:
        m = line_re.match(line)
        if m:
            value, name = m.groups()
            if "<not supported>" in value:
                value = ""
            else:
                value = value.replace(",", "")
            counters.append((name, value))

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Counter", "Value"])
    for name, value in counters:
        writer.writerow([name, value])

#!/usr/bin/env python3
"""
emit_curl.py

Reads a metrics JSON object (from stdin or file) and prints the full
multi-line curl command exactly in the requested wrapped format.

Usage:
  cat metrics.json | python3 emit_curl.py --run-id r1
  python3 emit_curl.py --run-id r1 --in metrics.json
"""

import argparse
import json
import sys
from typing import Any, Dict


def read_input_json(path: str | None) -> Dict[str, Any]:
    if path:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.load(sys.stdin)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="r1")
    ap.add_argument("--url", default="http://127.0.0.1:5001/search")
    ap.add_argument("--in", dest="in_path", default=None,
                    help="input JSON file; otherwise stdin")
    args = ap.parse_args()

    metrics = read_input_json(args.in_path)

    payload = {
        "run_id": args.run_id,
        "query_json": metrics,
    }

    # Pretty JSON with 4-space indentation to match your example style
    pretty_json = json.dumps(payload, indent=4, ensure_ascii=False)

    curl_cmd = (
        f"curl -X POST {args.url} \\\n"
        f"  -H 'Content-Type: application/json' \\\n"
        f"  -d '{pretty_json}'"
    )

    print(curl_cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
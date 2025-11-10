#!/usr/bin/env python3
"""
Fill homeless_count in master.csv from monthly_counts.csv.

- Matches on the shared column name: year_month
- Only updates rows that already exist in master.csv
  (does not add new dates/rows)
- Writes a new file by default: master_with_counts.csv

Usage:
  python add_counts_to_master.py \
    --master master.csv \
    --monthly monthly_counts.csv \
    --output master_with_counts.csv
"""

from __future__ import annotations

import argparse
import csv
from typing import Dict, List


def read_monthly_map(path: str) -> Dict[str, str]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"year_month", "count"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"{path} must contain columns: year_month,count"
            )
        mapping: Dict[str, str] = {}
        for row in reader:
            ym = row.get("year_month", "").strip()
            cnt = row.get("count", "").strip()
            if ym:
                mapping[ym] = cnt
        return mapping


def update_master_rows(master_rows: List[dict], monthly_map: Dict[str, str]) -> List[dict]:
    # Ensure homeless_count column exists; if not, add it with empty values
    for row in master_rows:
        ym = (row.get("year_month") or "").strip()
        if ym in monthly_map:
            row["homeless_count"] = monthly_map[ym]
        else:
            # Do not modify existing value if already set; otherwise keep blank
            row.setdefault("homeless_count", row.get("homeless_count", ""))
    return master_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", default="master.csv", help="Path to master CSV")
    parser.add_argument("--monthly", default="monthly_counts.csv", help="Path to monthly counts CSV")
    parser.add_argument(
        "--output", default="master_with_counts.csv", help="Path to output CSV"
    )
    args = parser.parse_args()

    monthly_map = read_monthly_map(args.monthly)

    with open(args.master, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "year_month" not in fieldnames:
            raise ValueError("master CSV must contain a 'year_month' column")
        if "homeless_count" not in fieldnames:
            fieldnames = list(fieldnames) + ["homeless_count"]
        rows = list(reader)

    updated_rows = update_master_rows(rows, monthly_map)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)


if __name__ == "__main__":
    main()


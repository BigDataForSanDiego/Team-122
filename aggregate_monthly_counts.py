#!/usr/bin/env python3
"""
Aggregate monthly counts from counts_all_data.csv into year_month,count.

Usage:
  python aggregate_monthly_counts.py \
    --input counts_all_data.csv \
    --output monthly_counts.csv

The input is expected to have at least the columns:
  COUNT, MONTH, YEAR

This script sums COUNT per (YEAR, MONTH) and writes a new CSV with:
  year_month,count
where year_month is like 2014-01.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from typing import Dict, Tuple


def parse_int(val: str) -> int:
    """Parse an integer that may be serialized as a float-like string (e.g., '1.00000000000')."""
    return int(float(val))


def parse_float(val: str) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def aggregate_monthly(input_path: str) -> Dict[Tuple[int, int], float]:
    totals: Dict[Tuple[int, int], float] = defaultdict(float)
    with open(input_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = {"COUNT", "MONTH", "YEAR"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Input CSV missing required columns: {', '.join(sorted(missing))}"
            )
        for row in reader:
            try:
                year = parse_int(row["YEAR"])  # e.g., '2014.00000000000'
                month = parse_int(row["MONTH"])  # e.g., '1.00000000000'
                count = parse_float(row["COUNT"])  # e.g., '5.00000000000'
            except Exception:
                # Skip rows that can't be parsed
                continue

            if month < 1 or month > 12:
                # Guard against bad data
                continue

            totals[(year, month)] += count

    return totals


def write_output(totals: Dict[Tuple[int, int], float], output_path: str) -> None:
    # Sort by (year, month)
    items = sorted(totals.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["year_month", "count"])
        for (year, month), total in items:
            # COUNT values are integers in the source, but stored as float strings.
            # Round to nearest int for clearer reporting.
            total_int = int(round(total))
            writer.writerow([f"{year}-{month:02d}", total_int])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="counts_all_data.csv",
        help="Path to input CSV (default: counts_all_data.csv)",
    )
    parser.add_argument(
        "--output",
        default="monthly_counts.csv",
        help="Path to output CSV (default: monthly_counts.csv)",
    )
    args = parser.parse_args()

    totals = aggregate_monthly(args.input)
    write_output(totals, args.output)


if __name__ == "__main__":
    main()


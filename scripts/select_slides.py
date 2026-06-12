#!/usr/bin/env python3
"""
Select 200 patients from a slide manifest CSV, picking 3 consecutive slides
per patient from either the DORS or VEN series (randomly).

Usage:
    python select_slides.py manifest.csv output.csv
    python select_slides.py manifest.csv output.csv --n_patients 100 --seed 42
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SERIES_RE = re.compile(r"^(DORS|VEN)\s+(\d+)$", re.IGNORECASE)


# ── parsing ────────────────────────────────────────────────────────────────────

def _parse_slide_short(short_name: str) -> tuple[str, int] | None:
    match = _SERIES_RE.match(str(short_name).strip())
    if not match:
        return None
    return match.group(1).upper(), int(match.group(2))


def _find_consecutive_windows(numbers: list[int], window: int) -> list[list[int]]:
    """Return all contiguous windows of `window` consecutive integers."""
    sorted_nums = sorted(set(numbers))
    if len(sorted_nums) < window:
        return []

    windows = []
    run: list[int] = [sorted_nums[0]]

    for n in sorted_nums[1:]:
        if n == run[-1] + 1:
            run.append(n)
        else:
            for i in range(len(run) - window + 1):
                windows.append(run[i : i + window])
            run = [n]

    for i in range(len(run) - window + 1):
        windows.append(run[i : i + window])

    return windows


# ── eligibility ────────────────────────────────────────────────────────────────

def _build_pool(df: pd.DataFrame, n_consecutive: int) -> dict[str, dict[str, list[list[int]]]]:
    """
    Returns:
        pool[case_id][series] = list of all valid slide-number windows (each a list of n_consecutive ints)
    Only cases that have at least one eligible series are included.
    """
    parsed = df["SlideNameShort"].apply(_parse_slide_short)
    df = df.copy()
    df["_series"] = parsed.apply(lambda x: x[0] if x else None)
    df["_num"] = parsed.apply(lambda x: x[1] if x else None)
    df = df.dropna(subset=["_series", "_num"])
    df["_num"] = df["_num"].astype(int)

    pool: dict[str, dict[str, list[list[int]]]] = {}

    for case_id, case_df in df.groupby("Case"):
        eligible: dict[str, list[list[int]]] = {}
        for series, series_df in case_df.groupby("_series"):
            windows = _find_consecutive_windows(series_df["_num"].tolist(), n_consecutive)
            if windows:
                eligible[series] = windows
        if eligible:
            pool[case_id] = eligible

    return pool, df


# ── selection ──────────────────────────────────────────────────────────────────

def select_slides(
    df: pd.DataFrame,
    n_patients: int,
    n_consecutive: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    pool, df_parsed = _build_pool(df, n_consecutive)

    eligible_cases = list(pool.keys())
    if len(eligible_cases) < n_patients:
        print(
            f"Warning: only {len(eligible_cases)} patients have ≥{n_consecutive} "
            f"consecutive slides in one series; selecting all of them.",
            file=sys.stderr,
        )
        n_patients = len(eligible_cases)

    sampled_cases: list[str] = rng.choice(eligible_cases, size=n_patients, replace=False).tolist()

    rows = []
    for case_id in sampled_cases:
        eligible_series = pool[case_id]

        # If both DORS and VEN are eligible, pick one at random
        series: str = rng.choice(list(eligible_series.keys()))

        # Pick one of the valid windows at random
        windows = eligible_series[series]
        chosen_window: list[int] = windows[rng.integers(len(windows))]

        case_df = df_parsed[df_parsed["Case"] == case_id]
        for num in chosen_window:
            match = case_df[(case_df["_series"] == series) & (case_df["_num"] == num)]
            if match.empty:
                continue
            row = match.iloc[0]
            rows.append({
                "case_id": case_id,
                "slide_id": row["SlideNameShort"],
                "slide_name": row["SlideName"],
                "original_path": row.get("OriginalPath", ""),
            })

    return pd.DataFrame(rows)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Randomly select patients and consecutive slides from a slide manifest."
    )
    parser.add_argument("manifest", type=Path, help="Input slide manifest CSV/TSV")
    parser.add_argument("output", type=Path, help="Output CSV path")
    parser.add_argument("--n_patients", type=int, default=200, help="Number of patients to select (default: 200)")
    parser.add_argument("--n_consecutive", type=int, default=3, help="Number of consecutive slides per patient (default: 3)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    df = pd.read_csv(args.manifest, sep=None, engine="python")

    rng = np.random.default_rng(args.seed)
    result = select_slides(df, args.n_patients, args.n_consecutive, rng)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Selected {len(result)} slide rows ({len(result) // args.n_consecutive} patients) → {args.output}")


if __name__ == "__main__":
    main()

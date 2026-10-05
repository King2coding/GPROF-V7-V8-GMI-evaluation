#!/usr/bin/env python3
"""Validate Stage-IV-era matchup files and emit timing/QC CSV evidence."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import xarray as xr

REQUIRED = {
    "scan_time", "latitude", "longitude", "surfacePrecipitation_V7",
    "surfacePrecipitation_V8", "MRMS_Pass2", "RAQI", "ERA5_precipitation",
    "MERRA2_T2M", "MERRA2_T2MWET", "AutoSnow", "StageIV", "land_mask",
    "MRMS_interval_start", "MRMS_interval_end", "StageIV_interval_start",
    "StageIV_interval_end", "StageIV_status", "stageiv_interval_valid_flag",
}


def validate(path, example_count=8):
    with xr.open_dataset(path, decode_timedelta=False) as ds:
        missing = sorted(REQUIRED - set(ds.variables))
        if missing:
            raise KeyError(f"{path} lacks required variables: {missing}")
        scan = ds["scan_time"].values.astype("datetime64[ns]")
        stage_start = ds["StageIV_interval_start"].values.astype("datetime64[ns]")
        stage_end = ds["StageIV_interval_end"].values.astype("datetime64[ns]")
        mrms_start = ds["MRMS_interval_start"].values.astype("datetime64[ns]")
        mrms_end = ds["MRMS_interval_end"].values.astype("datetime64[ns]")
        stage_time_ok = (stage_start < scan) & (scan <= stage_end)
        mrms_time_ok = (mrms_start < scan) & (scan <= mrms_end)
        stage = ds["StageIV"].values
        finite_by_scan = np.isfinite(stage).any(axis=1)
        candidates = np.flatnonzero(finite_by_scan)
        if candidates.size:
            take = candidates[np.linspace(0, candidates.size - 1, min(example_count, candidates.size)).astype(int)]
        else:
            take = np.linspace(0, len(scan) - 1, min(example_count, len(scan))).astype(int)
        examples = []
        for i in take:
            pixels = np.flatnonzero(np.isfinite(stage[i]))
            j = int(np.nanargmax(stage[i])) if pixels.size else 0
            examples.append({
                "file": Path(path).name, "scan_index": int(i), "pixel_index": j,
                "GMI_ScanTime": str(scan[i]),
                "StageIV_interval_start": str(stage_start[i]),
                "StageIV_interval_end": str(stage_end[i]),
                "StageIV_interval_contains_scan": bool(stage_time_ok[i]),
                "MRMS_interval_start": str(mrms_start[i]),
                "MRMS_interval_end": str(mrms_end[i]),
                "MRMS_interval_contains_scan": bool(mrms_time_ok[i]),
                "latitude": float(ds["latitude"].values[i, j]),
                "longitude": float(ds["longitude"].values[i, j]),
                "StageIV_mm": float(stage[i, j]) if np.isfinite(stage[i, j]) else np.nan,
                "MRMS_mm": float(ds["MRMS_Pass2"].values[i, j]) if np.isfinite(ds["MRMS_Pass2"].values[i, j]) else np.nan,
                "StageIV_status": float(ds["StageIV_status"].values[i]),
            })
        summary = {
            "file": Path(path).name,
            "scan_count": int(ds.sizes["scan"]),
            "footprint_count": int(ds.sizes["scan"] * ds.sizes["pixel"]),
            "stageiv_finite_footprints": int(np.isfinite(stage).sum()),
            "stageiv_finite_fraction": float(np.isfinite(stage).mean()),
            "mrms_finite_footprints": int(np.isfinite(ds["MRMS_Pass2"].values).sum()),
            "t2mwet_finite_footprints": int(np.isfinite(ds["MERRA2_T2MWET"].values).sum()),
            "land_footprints": int((ds["land_mask"].values == 1).sum()),
            "stageiv_interval_scans_valid": int(stage_time_ok.sum()),
            "stageiv_interval_scans_total": int(stage_time_ok.size),
            "mrms_interval_scans_valid": int(mrms_time_ok.sum()),
            "mrms_interval_scans_total": int(mrms_time_ok.size),
            "passed": bool(stage_time_ok.all() and mrms_time_ok.all()),
        }
    return summary, examples


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--example-count", type=int, default=8)
    args = parser.parse_args()
    summaries, examples = [], []
    for path in args.inputs:
        summary, file_examples = validate(path, args.example_count)
        summaries.append(summary); examples.extend(file_examples)
        print(summary)
    write_csv(args.output_dir / "validation_summary.csv", summaries)
    write_csv(args.output_dir / "temporal_match_examples.csv", examples)
    if not all(row["passed"] for row in summaries):
        raise SystemExit("validation failed: at least one interval-containment check failed")


if __name__ == "__main__":
    main()

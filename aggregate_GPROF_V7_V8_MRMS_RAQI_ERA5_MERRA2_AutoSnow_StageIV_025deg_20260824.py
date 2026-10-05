#!/usr/bin/env python3
"""Create derived 0.25-degree x Stage-IV-interval statistics with GMI counts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import my_functions_GPROF_V7_V8_MRMS_RAQI_ERA5_MERRA2_AutoSnow_StageIV_20260824 as mf

VALUES = [
    "surfacePrecipitation_V7", "surfacePrecipitation_V8", "MRMS_Pass2",
    "RAQI", "ERA5_precipitation", "MERRA2_T2M", "MERRA2_T2MWET", "StageIV",
]
MEAN_VALUES = [
    "surfacePrecipitation_V7", "surfacePrecipitation_V8", "MRMS_Pass2",
    "RAQI", "ERA5_precipitation", "StageIV",
]


def load_footprints(paths):
    frames = []
    for path in paths:
        with xr.open_dataset(path, decode_timedelta=False) as ds:
            required = {"latitude", "longitude", "StageIV_interval_end", "land_mask", "AutoSnow", *VALUES}
            missing = sorted(required - set(ds.variables))
            if missing:
                raise KeyError(f"{path} lacks {missing}")
            nscan, npixel = ds.sizes["scan"], ds.sizes["pixel"]
            frame = {
                "latitude": ds["latitude"].values.ravel(),
                "longitude": ds["longitude"].values.ravel(),
                "land_mask": ds["land_mask"].values.ravel(),
                "AutoSnow": ds["AutoSnow"].values.ravel(),
                "reference_interval_end": np.repeat(
                    ds["StageIV_interval_end"].values.astype("datetime64[s]"), npixel
                ),
            }
            frame.update({name: ds[name].values.ravel() for name in VALUES})
            frames.append(pd.DataFrame(frame))
    return pd.concat(frames, ignore_index=True)


def _mode(values):
    values = np.asarray(values)
    values = values[np.isin(values, [0, 1, 2, 3])]
    return np.nan if not values.size else int(np.bincount(values.astype(int), minlength=4).argmax())


def aggregate(data, resolution=0.25, extent=(-125.0, -66.0, 24.0, 50.0), land_only=True):
    work = data.copy()
    west, east, south, north = extent
    work = work[
        work["longitude"].between(west, east, inclusive="both")
        & work["latitude"].between(south, north, inclusive="both")
    ]
    if land_only:
        work = work[work["land_mask"] == 1]
    if work.empty:
        raise ValueError("no footprints remain after extent/land filtering")
    work = work.copy()
    work["grid_latitude"] = np.floor(work["latitude"] / resolution) * resolution + resolution / 2
    work["grid_longitude"] = np.floor(work["longitude"] / resolution) * resolution + resolution / 2
    keys = ["reference_interval_end", "grid_latitude", "grid_longitude"]
    grouped = work.groupby(keys, observed=True, dropna=False)
    means = grouped[MEAN_VALUES].mean()
    means[["MERRA2_T2M", "MERRA2_T2MWET"]] = grouped[["MERRA2_T2M", "MERRA2_T2MWET"]].first()
    means["AutoSnow"] = grouped["AutoSnow"].agg(_mode)
    means["land_mask"] = grouped["land_mask"].agg(_mode)
    means = means.reset_index()
    means["GMI_footprint_count"] = grouped.size().to_numpy(dtype=np.uint32)
    return means.sort_values(keys).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resolution", type=float, default=0.25)
    parser.add_argument("--extent", nargs=4, type=float, metavar=("WEST", "EAST", "SOUTH", "NORTH"), default=(-125.0, -66.0, 24.0, 50.0))
    parser.add_argument("--include-water", action="store_true", help="include water footprints; default is land_mask == 1")
    args = parser.parse_args()
    if args.resolution <= 0:
        parser.error("--resolution must be positive")
    west, east, south, north = args.extent
    if not (west < east and south < north):
        parser.error("--extent must satisfy WEST < EAST and SOUTH < NORTH")
    result = aggregate(load_footprints(args.inputs), args.resolution, tuple(args.extent), not args.include_water)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() != ".pkl":
        parser.error("large derived tables must use a .pkl output")
    result.to_pickle(args.output)
    print(f"wrote {len(result):,} grid-cell/interval rows to {args.output}")


if __name__ == "__main__":
    main()

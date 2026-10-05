#!/usr/bin/env python3
"""Count raw paired GPROF V7/V8 GMI footprint locations on a 0.25° grid.

This script deliberately operates before precipitation- and reference-validity
screening.  A sample is counted when a V7 and V8 source granule share the same
orbit/acquisition key, their scans can be aligned by time, both products have a
finite geolocation for the same scan/pixel index, and the V7 footprint center
falls over CONUS land.  Precipitation values are never read.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shapereader
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import shapely
import xarray as xr


V7_ROOT = Path("/scratch/kkumah/GPM_GMI/V7")
V8_ROOT = Path("/scratch/kkumah/GPM_GMI/V8")
PROJECT_ROOT = Path("/home/kkumah/Projects/Comparing_V7_V8_GPROF-GMI_data")
RESULTS_ROOT = PROJECT_ROOT / "Results_ERA5_interval_fullarchive_CONUS_land_20260917"
OUTPUT_TABLE = RESULTS_ROOT / "dfs/methods_sampling_coverage/raw_v7_v8_paired_footprint_coverage_025deg.pkl"
OUTPUT_FIGURE = RESULTS_ROOT / "plots/methods_sampling_coverage/raw_v7_v8_paired_footprint_coverage_025deg.png"

WEST, EAST, SOUTH, NORTH = -125.0, -66.0, 24.0, 50.0
RESOLUTION = 0.25


def parse_file(path: Path) -> dict[str, object] | None:
    parts = path.name.split(".")
    if len(parts) < 6:
        return None
    match = re.fullmatch(r"(\d{8})-S(\d{6})-E(\d{6})", parts[4])
    if match is None:
        return None
    ymd, start_hms, end_hms = match.groups()
    return {
        "date": dt.datetime.strptime(ymd, "%Y%m%d").date(),
        "key": f"{ymd}-S{start_hms}-E{end_hms}.{parts[5]}",
        "orbit_id": f"{parts[5]}_{ymd}",
        "path": path,
    }


def discover_pairs(start: dt.date, end: dt.date) -> list[tuple[dict, dict]]:
    def index(root: Path, pattern: str) -> dict[str, dict]:
        result: dict[str, dict] = {}
        for path in root.glob(pattern):
            info = parse_file(path)
            if info is not None and start <= info["date"] <= end:
                result[str(info["key"])] = info
        return result

    v7 = index(V7_ROOT, "*.HDF5")
    v8 = index(V8_ROOT, "*.nc")
    return [(v7[key], v8[key]) for key in sorted(set(v7) & set(v8))]


def read_scan_keys(group: h5py.Group) -> list[tuple[int, ...]]:
    names = ("Year", "Month", "DayOfMonth", "Hour", "Minute", "Second")
    values = [np.asarray(group[name][:], dtype=np.int64) for name in names]
    if "MilliSecond" in group:
        values.append(np.asarray(group["MilliSecond"][:], dtype=np.int64))
    else:
        values.append(np.zeros_like(values[0]))
    return list(zip(*values))


def read_geometry(path: Path) -> tuple[np.ndarray, np.ndarray, list[tuple[int, ...]]]:
    with h5py.File(path, "r") as stream:
        group = stream["S1"]
        latitude = np.asarray(group["Latitude"][:], dtype=np.float32)
        longitude = np.asarray(group["Longitude"][:], dtype=np.float32)
        scan_keys = read_scan_keys(group["ScanTime"])
    longitude = np.where(longitude > 180.0, longitude - 360.0, longitude)
    return latitude, longitude, scan_keys


def conus_geometry():
    boundary = shapereader.natural_earth(
        resolution="10m", category="cultural", name="admin_0_countries"
    )
    matches = [
        record.geometry
        for record in shapereader.Reader(boundary).records()
        if record.attributes.get("ADM0_A3") == "USA"
    ]
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one Natural Earth USA geometry.")
    geometry = matches[0]
    shapely.prepare(geometry)
    return geometry


def aligned_scan_indices(v7_keys, v8_keys) -> tuple[np.ndarray, np.ndarray]:
    v8_lookup = {key: index for index, key in enumerate(v8_keys)}
    v7_indices, v8_indices = [], []
    for index, key in enumerate(v7_keys):
        match = v8_lookup.get(key)
        if match is not None:
            v7_indices.append(index)
            v8_indices.append(match)
    return np.asarray(v7_indices, dtype=int), np.asarray(v8_indices, dtype=int)


def calculate(pairs, *, maximum_pairs: int | None = None) -> xr.Dataset:
    if maximum_pairs is not None:
        pairs = pairs[:maximum_pairs]
    lat_edges = np.arange(SOUTH, NORTH + RESOLUTION, RESOLUTION)
    lon_edges = np.arange(WEST, EAST + RESOLUTION, RESOLUTION)
    counts = np.zeros((len(lat_edges) - 1, len(lon_edges) - 1), dtype=np.int64)
    geometry = conus_geometry()
    used_pairs = used_scans = raw_locations = 0

    for number, (v7_info, v8_info) in enumerate(pairs, start=1):
        v7_lat, v7_lon, v7_time = read_geometry(Path(v7_info["path"]))
        v8_lat, v8_lon, v8_time = read_geometry(Path(v8_info["path"]))
        v7_scan, v8_scan = aligned_scan_indices(v7_time, v8_time)
        if not len(v7_scan):
            continue
        pixels = min(v7_lat.shape[1], v8_lat.shape[1])
        latitude = v7_lat[v7_scan, :pixels]
        longitude = v7_lon[v7_scan, :pixels]
        finite_pair = (
            np.isfinite(latitude)
            & np.isfinite(longitude)
            & np.isfinite(v8_lat[v8_scan, :pixels])
            & np.isfinite(v8_lon[v8_scan, :pixels])
        )
        candidate = (
            finite_pair
            & (latitude >= SOUTH)
            & (latitude < NORTH)
            & (longitude >= WEST)
            & (longitude < EAST)
        )
        if not np.any(candidate):
            continue
        lat = latitude[candidate]
        lon = longitude[candidate]
        inside = shapely.contains_xy(geometry, lon, lat)
        if np.any(inside):
            histogram = np.histogram2d(
                lat[inside], lon[inside], bins=(lat_edges, lon_edges)
            )[0]
            counts += histogram.astype(np.int64)
            raw_locations += int(inside.sum())
        used_pairs += 1
        used_scans += len(v7_scan)
        if number % 250 == 0 or number == len(pairs):
            print(f"Processed {number:,}/{len(pairs):,} common granules")

    dataset = xr.Dataset(
        data_vars={"paired_footprint_count": (("latitude", "longitude"), counts)},
        coords={
            "latitude": (lat_edges[:-1] + lat_edges[1:]) / 2.0,
            "longitude": (lon_edges[:-1] + lon_edges[1:]) / 2.0,
        },
        attrs={
            "population": "raw paired V7-V8 GMI scan-pixel locations over CONUS land",
            "precipitation_validity_filter": "none",
            "reference_validity_filter": "none",
            "coordinate_assignment": "V7 footprint center after scan-time alignment",
            "common_granules_processed": int(used_pairs),
            "aligned_scans": int(used_scans),
            "paired_locations_counted": int(raw_locations),
            "grid_resolution_degrees": RESOLUTION,
        },
    )
    return dataset


def plot(dataset: xr.Dataset, output: Path) -> None:
    field = dataset["paired_footprint_count"].values.astype(float)
    field[field <= 0] = np.nan
    positive = field[np.isfinite(field)]
    # Sparse coastal boundary cells have counts as low as one and otherwise
    # compress the color variation across the populated CONUS interior. Use a
    # robust positive lower bound while retaining the true observed maximum.
    norm = Normalize(
        vmin=float(np.nanpercentile(positive, 5.0)),
        vmax=float(np.nanmax(positive)),
    )
    figure = plt.figure(figsize=(10.5, 5.2), dpi=200)
    axis = figure.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    artist = axis.pcolormesh(
        dataset.longitude,
        dataset.latitude,
        field,
        cmap="cividis",
        norm=norm,
        shading="auto",
        transform=ccrs.PlateCarree(),
        zorder=1,
    )
    axis.set_extent((WEST, EAST, SOUTH, NORTH), crs=ccrs.PlateCarree())
    # Mask coastal grid-cell spillover and inland water visually. Counts remain
    # defined by land-located footprint centers and are not recalculated here.
    axis.add_feature(
        cfeature.OCEAN.with_scale("10m"), facecolor="white", edgecolor="none", zorder=2,
    )
    axis.add_feature(
        cfeature.LAKES.with_scale("10m"), facecolor="white", edgecolor="0.35",
        linewidth=0.35, zorder=2,
    )
    axis.coastlines(resolution="10m", linewidth=0.8, zorder=3)
    axis.add_feature(
        cfeature.STATES.with_scale("10m"), linewidth=0.35, edgecolor="0.35",
        facecolor="none", zorder=3,
    )
    gridlines = axis.gridlines(draw_labels=True, linewidth=0.35, alpha=0.45)
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.xlabel_style = {"size": 13}
    gridlines.ylabel_style = {"size": 13}
    colorbar = figure.colorbar(
        artist, ax=axis, orientation="horizontal", pad=0.10, shrink=0.82,
        extend="min",
    )
    colorbar.set_label("Paired footprint count per 0.25° grid cell", fontsize=12)
    colorbar.ax.tick_params(labelsize=11)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", type=dt.date.fromisoformat, default=dt.date(2021, 1, 1))
    parser.add_argument("--end-date", type=dt.date.fromisoformat, default=dt.date(2024, 12, 31))
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--output-table", type=Path, default=OUTPUT_TABLE)
    parser.add_argument("--output-figure", type=Path, default=OUTPUT_FIGURE)
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Rerender the existing output table without rereading the GPROF archive.",
    )
    args = parser.parse_args()
    if args.plot_only:
        table = pd.read_pickle(args.output_table)
        required = {"latitude", "longitude", "paired_footprint_count"}
        if not required.issubset(table.columns):
            raise KeyError(f"{args.output_table} lacks {sorted(required - set(table.columns))}")
        dataset = table.set_index(["latitude", "longitude"]).to_xarray()
        plot(dataset, args.output_figure)
        print(f"Rerendered {args.output_figure} from {args.output_table}")
        return
    pairs = discover_pairs(args.start_date, args.end_date)
    print(f"Discovered {len(pairs):,} common raw V7-V8 granules")
    dataset = calculate(pairs, maximum_pairs=args.max_pairs)
    args.output_table.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_dataframe().reset_index().to_pickle(args.output_table)
    plot(dataset, args.output_figure)
    print(dataset.attrs)
    print(f"Saved {args.output_table}")
    print(f"Saved {args.output_figure}")


if __name__ == "__main__":
    main()

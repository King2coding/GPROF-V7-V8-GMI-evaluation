#!/usr/bin/env python3
"""
Small helpers for GMI V7/V8, ERA5, MERRA2, and AutoSnow collocation.

This module intentionally avoids IMERG and GeoTIFF AutoSnow logic.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import h5py
import numpy as np


EPOCH = dt.datetime(1970, 1, 1)


def as_text(value):
    """Return a compact text representation for HDF5/netCDF attributes."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return as_text(value.reshape(-1)[0])
        return ", ".join(as_text(v) for v in value.reshape(-1))
    if isinstance(value, np.generic):
        return value.item()
    return value


def list_files(directory, suffix):
    files = sorted(Path(directory).glob(f"*{suffix}"))
    return [str(path) for path in files if path.is_file()]


def parse_gmi_filename(path):
    """
    Parse GMI filename fields used for V7/V8 pairing.

    Expected example:
        2A-CLIM.GPM.GMI.GPROFNNv1.20140304-S175932-E193159.000079.V08A.nc
    """
    name = Path(path).name
    parts = name.split(".")
    if len(parts) < 6:
        raise ValueError(f"Could not parse GMI filename: {name}")

    date_time = parts[4]
    orbit = parts[5]
    match = re.match(r"(\d{8})-S(\d{6})-E(\d{6})", date_time)
    if not match:
        raise ValueError(f"Could not parse GMI start/end time from filename: {name}")

    ymd, start_hms, end_hms = match.groups()
    start = dt.datetime.strptime(ymd + start_hms, "%Y%m%d%H%M%S")
    end = dt.datetime.strptime(ymd + end_hms, "%Y%m%d%H%M%S")
    if end < start:
        end += dt.timedelta(days=1)

    return {
        "filename": name,
        "orbit": orbit,
        "ymd": ymd,
        "year": int(ymd[:4]),
        "start": start,
        "end": end,
        "pair_key": f"{ymd}-S{start_hms}-E{end_hms}.{orbit}",
        "orbit_id": f"{orbit}_{ymd}",
    }


def seconds_since_epoch(datetimes):
    return np.array([(value - EPOCH).total_seconds() for value in datetimes], dtype=np.float64)


def datetime_from_ymdhms(year, month, day, hour, minute, second):
    out = []
    for vals in zip(year, month, day, hour, minute, second):
        out.append(dt.datetime(*(int(v) for v in vals)))
    return np.array(out, dtype=object)


def read_gmi_s1(path):
    """Read S1 precipitation, lat/lon, scan datetimes, and useful attributes."""
    with h5py.File(path, "r") as h5:
        s1 = h5["S1"]
        precip_ds = s1["surfacePrecipitation"]
        lat_ds = s1["Latitude"]
        lon_ds = s1["Longitude"]

        precip = precip_ds[:].astype(np.float32)
        lat = lat_ds[:].astype(np.float32)
        lon = lon_ds[:].astype(np.float32)

        fill_value = as_text(precip_ds.attrs.get("_FillValue", np.float32(-9999.9)))
        try:
            precip = np.where(precip == float(fill_value), np.nan, precip).astype(np.float32)
        except (TypeError, ValueError):
            pass

        scan_time = s1["ScanTime"]
        scan_datetimes = datetime_from_ymdhms(
            scan_time["Year"][:],
            scan_time["Month"][:],
            scan_time["DayOfMonth"][:],
            scan_time["Hour"][:],
            scan_time["Minute"][:],
            scan_time["Second"][:],
        )

        attrs = {
            "surfacePrecipitation_units": as_text(
                precip_ds.attrs.get("units", precip_ds.attrs.get("Units", "source units not found"))
            ),
            "surfacePrecipitation_fill_value": fill_value,
            "latitude_units": as_text(lat_ds.attrs.get("units", lat_ds.attrs.get("Units", "degrees"))),
            "longitude_units": as_text(lon_ds.attrs.get("units", lon_ds.attrs.get("Units", "degrees"))),
        }

    return {
        "surface_precipitation": precip,
        "lat": lat,
        "lon": lon,
        "scan_datetimes": scan_datetimes,
        "scan_time_seconds": seconds_since_epoch(scan_datetimes),
        "attrs": attrs,
    }


def normalize_lon(lon, convention):
    lon = np.asarray(lon, dtype=np.float64)
    if convention == "0_360":
        return np.mod(lon, 360.0)
    if convention == "minus180_180":
        return ((lon + 180.0) % 360.0) - 180.0
    raise ValueError(f"Unknown longitude convention: {convention}")


def nearest_indices(values, grid):
    """Nearest index in a monotonic or non-monotonic 1D grid."""
    values = np.asarray(values)
    grid = np.asarray(grid)
    order = np.argsort(grid)
    sorted_grid = grid[order]
    pos = np.searchsorted(sorted_grid, values)
    pos0 = np.clip(pos - 1, 0, sorted_grid.size - 1)
    pos1 = np.clip(pos, 0, sorted_grid.size - 1)
    choose1 = np.abs(sorted_grid[pos1] - values) < np.abs(sorted_grid[pos0] - values)
    nearest_sorted_pos = np.where(choose1, pos1, pos0)
    return order[nearest_sorted_pos]


def autosnow_file_for_date(root, date_value):
    year = date_value.year
    doy = date_value.timetuple().tm_yday
    name = f"gmasi_snowice_reproc_v003_{year}{doy:03d}_native_0.04deg.nc"
    return Path(root) / f"{year}" / name


def merra2_file_for_date(root, date_value):
    root = Path(root)
    pattern = f"MERRA2_*.tavg1_2d_slv_Nx.{date_value:%Y%m%d}.SUB.nc"
    matches = sorted(root.glob(pattern))
    if matches:
        return matches[0]
    return None


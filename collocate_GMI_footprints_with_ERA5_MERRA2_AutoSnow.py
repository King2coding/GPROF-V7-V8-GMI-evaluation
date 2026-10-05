#!/usr/bin/env python3
"""
Collocate matched GMI GPROF V7/V8 native footprints with ERA5, MERRA2, and AutoSnow.

One output NetCDF is written per matched GMI orbit. IMERG is intentionally absent.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import xarray as xr

import my_functions_gmi_era5_merra2_autosnow as mf

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


DEFAULT_GMI_V7_DIR = "/scratch/kkumah/GPM_GMI/V7"
DEFAULT_GMI_V8_DIR = "/scratch/kkumah/GPM_GMI/V8"
DEFAULT_ERA5_DIR = "/scratch/kkumah/ERA5_tp_hourly"
DEFAULT_MERRA2_DIR = (
    "/ra1/pubdat/AVHRR_CloudSat_proj/MERRA2/"
    "merra2_archive_19800101_20251231"
)
DEFAULT_AUTOSNOW_DIR = "/scratch/kkumah/Autosnow_2014_to_2024_nc"
DEFAULT_OUTPUT_DIR = "/scratch/kkumah/Collocated_GMI_ERA5_MERRA_Autosnow"
DEFAULT_QC_LOG_SUBDIR = "QC_logs"

SCRIPT_NAME = Path(__file__).name
AUTOSNOW_FILL = np.uint8(255)
GEOMETRY_PASS_NORMAL = "PASS_NORMAL"
GEOMETRY_PASS_MINOR_EDGE_WARNING = "PASS_MINOR_EDGE_WARNING"
GEOMETRY_FAIL_TRUE_MISMATCH = "FAIL_TRUE_GEOMETRY_MISMATCH"
GEOMETRY_FAIL_SHAPE_OR_TIME = "FAIL_SHAPE_OR_TIME_MISMATCH"
GEOMETRY_MINOR_EDGE_PIXEL_MARGIN = 30
GEOMETRY_MINOR_MAX_EXCEEDANCE_FRACTION = 0.001
GEOMETRY_MINOR_HARD_LIMIT_DEG = 0.01
GEOMETRY_TIME_TOLERANCE_SECONDS = 1.0
GEOMETRY_QC_FIELDNAMES = [
    "orbit_id",
    "date",
    "classification",
    "max_lat_diff_deg",
    "max_wrapped_lon_diff_deg",
    "max_raw_lon_diff_deg",
    "median_lat_diff_deg",
    "p99_lat_diff_deg",
    "p999_lat_diff_deg",
    "median_wrapped_lon_diff_deg",
    "p99_wrapped_lon_diff_deg",
    "p999_wrapped_lon_diff_deg",
    "pixels_exceeding_tolerance",
    "percent_pixels_exceeding_tolerance",
    "edge_only_exceedances",
    "max_lat_diff_index",
    "max_wrapped_lon_diff_index",
    "max_raw_lon_diff_index",
    "max_lat_diff_v7_lon",
    "max_lat_diff_v7_lat",
    "max_lat_diff_v8_lon",
    "max_lat_diff_v8_lat",
    "max_wrapped_lon_diff_v7_lon",
    "max_wrapped_lon_diff_v7_lat",
    "max_wrapped_lon_diff_v8_lon",
    "max_wrapped_lon_diff_v8_lat",
    "max_raw_lon_diff_v7_lon",
    "max_raw_lon_diff_v7_lat",
    "max_raw_lon_diff_v8_lon",
    "max_raw_lon_diff_v8_lat",
    "max_scan_time_diff_seconds",
    "final_action",
    "note",
]
FAILED_ORBIT_FIELDNAMES = [
    "orbit_id",
    "date",
    "classification",
    "reason",
    "v7_file",
    "v8_file",
    "error",
]


def utc_from_seconds(seconds):
    return dt.datetime.fromtimestamp(float(seconds), dt.timezone.utc).replace(tzinfo=None)


def progress(iterable, total=None):
    if tqdm is not None:
        return tqdm(iterable, total=total, desc="Matched GMI orbits", unit="orbit")
    return iterable


def has_module(name):
    return importlib.util.find_spec(name) is not None


def decode_attr(value):
    return mf.as_text(value)


def discover_gmi_pairs(gmi_v7_dir, gmi_v8_dir, year_start, year_end):
    v7_files = mf.list_files(gmi_v7_dir, ".HDF5")
    v8_files = mf.list_files(gmi_v8_dir, ".nc")

    v7_by_key = {}
    v8_by_key = {}
    for path in v7_files:
        try:
            info = mf.parse_gmi_filename(path)
        except ValueError:
            continue
        if year_start <= info["year"] <= year_end:
            v7_by_key[info["pair_key"]] = (path, info)

    for path in v8_files:
        try:
            info = mf.parse_gmi_filename(path)
        except ValueError:
            continue
        if year_start <= info["year"] <= year_end:
            v8_by_key[info["pair_key"]] = (path, info)

    keys = sorted(set(v7_by_key) & set(v8_by_key))
    pairs = []
    for key in keys:
        v7_path, v7_info = v7_by_key[key]
        v8_path, v8_info = v8_by_key[key]
        pairs.append(
            {
                "key": key,
                "orbit_id": v8_info["orbit_id"],
                "ymd": v8_info["ymd"],
                "year": v8_info["year"],
                "start": v8_info["start"],
                "end": v8_info["end"],
                "v7_file": v7_path,
                "v8_file": v8_path,
            }
        )

    unmatched_v7 = sorted(set(v7_by_key) - set(v8_by_key))
    unmatched_v8 = sorted(set(v8_by_key) - set(v7_by_key))
    return pairs, unmatched_v7, unmatched_v8


def build_era5_index(era5_dir):
    files = sorted(Path(era5_dir).glob("ERA5_tp_hourly_*.nc"))
    file_by_year = {}
    for path in files:
        try:
            year = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        file_by_year[year] = str(path)
    return file_by_year


def get_lon_convention(lon):
    lon = np.asarray(lon)
    if np.nanmin(lon) >= 0.0 and np.nanmax(lon) > 180.0:
        return "0_360"
    return "minus180_180"


def sample_era5(scan_seconds, lat, lon, era5_files):
    out = np.full(lat.shape, np.nan, dtype=np.float32)
    matched_time = np.full(scan_seconds.shape, np.nan, dtype=np.float64)
    source_files = set()
    notes = []

    scan_datetimes = np.array([utc_from_seconds(sec) for sec in scan_seconds], dtype=object)
    nearest_hours = np.array(
        [
            value.replace(minute=0, second=0, microsecond=0)
            + (dt.timedelta(hours=1) if value.minute >= 30 else dt.timedelta(0))
            for value in scan_datetimes
        ],
        dtype=object,
    )

    flat_lat = lat.reshape(-1)
    flat_lon = lon.reshape(-1)
    valid_space = np.isfinite(flat_lat) & np.isfinite(flat_lon)

    for year in sorted({value.year for value in nearest_hours}):
        era5_file = era5_files.get(year)
        if era5_file is None:
            notes.append(f"missing ERA5 year file {year}")
            continue
        with h5py.File(era5_file, "r") as h5:
            time_name = "valid_time" if "valid_time" in h5 else "time"
            times = h5[time_name][:]
            era5_lat = h5["latitude"][:]
            era5_lon = h5["longitude"][:]
            tp_ds = h5["tp"]
            units = decode_attr(tp_ds.attrs.get("units", ""))
            scale = 1000.0 if units == "m" else 1.0
            lon_convention = get_lon_convention(era5_lon)

            flat_y = mf.nearest_indices(flat_lat, era5_lat)
            flat_x = mf.nearest_indices(mf.normalize_lon(flat_lon, lon_convention), era5_lon)

            for hour in sorted(set(value for value in nearest_hours if value.year == year)):
                rows = np.where(nearest_hours == hour)[0]
                if rows.size == 0:
                    continue
                target_sec = (hour - mf.EPOCH).total_seconds()
                time_idx = int(mf.nearest_indices([target_sec], times)[0])
                matched_time[rows] = float(times[time_idx])
                field = tp_ds[time_idx, :, :].astype(np.float32) * scale
                field = np.where(np.isfinite(field), field, np.nan).astype(np.float32)
                field = np.where(field < 0, 0, field).astype(np.float32)
                for row in rows:
                    start = row * lat.shape[1]
                    stop = start + lat.shape[1]
                    row_valid = valid_space[start:stop]
                    values = np.full(lat.shape[1], np.nan, dtype=np.float32)
                    values[row_valid] = field[flat_y[start:stop][row_valid], flat_x[start:stop][row_valid]]
                    out[row, :] = values
            source_files.add(era5_file)

    return out, matched_time, sorted(source_files), "; ".join(notes)


def parse_merra2_times(time_values, units):
    units = decode_attr(units)
    prefix = "minutes since "
    if not units.startswith(prefix):
        raise ValueError(f"Unsupported MERRA2 time units: {units}")
    base_text = units[len(prefix) :]
    base = dt.datetime.strptime(base_text, "%Y-%m-%d %H:%M:%S")
    datetimes = [base + dt.timedelta(minutes=float(value)) for value in time_values]
    return mf.seconds_since_epoch(datetimes)


def sample_merra2(scan_seconds, lat, lon, merra2_dir):
    t2m = np.full(lat.shape, np.nan, dtype=np.float32)
    ts = np.full(lat.shape, np.nan, dtype=np.float32)
    matched_time = np.full(scan_seconds.shape, np.nan, dtype=np.float64)
    source_files = set()
    notes = []

    scan_dates = [utc_from_seconds(sec).date() for sec in scan_seconds]
    flat_lat = lat.reshape(-1)
    flat_lon = lon.reshape(-1)
    valid_space = np.isfinite(flat_lat) & np.isfinite(flat_lon)

    for date_value in sorted(set(scan_dates)):
        merra2_file = mf.merra2_file_for_date(merra2_dir, date_value)
        if merra2_file is None:
            notes.append(f"missing MERRA2 date {date_value.isoformat()}")
            continue

        rows = np.where(np.array(scan_dates, dtype=object) == date_value)[0]
        with h5py.File(merra2_file, "r") as h5:
            if "TS" not in h5:
                notes.append(f"missing TS in {merra2_file.name}")
                continue

            merra_lat = h5["lat"][:]
            merra_lon = h5["lon"][:]
            time_seconds = parse_merra2_times(h5["time"][:], h5["time"].attrs["units"])
            lon_convention = get_lon_convention(merra_lon)
            flat_y = mf.nearest_indices(flat_lat, merra_lat)
            flat_x = mf.nearest_indices(mf.normalize_lon(flat_lon, lon_convention), merra_lon)

            row_time_idx = {
                int(row): int(mf.nearest_indices([scan_seconds[row]], time_seconds)[0])
                for row in rows
            }
            for time_idx in sorted(set(row_time_idx.values())):
                time_rows = [row for row, idx in row_time_idx.items() if idx == time_idx]
                t2m_field = h5["T2M"][time_idx, :, :].astype(np.float32)
                ts_field = h5["TS"][time_idx, :, :].astype(np.float32)
                for var_name, target in [("T2M", t2m), ("TS", ts)]:
                    field = t2m_field if var_name == "T2M" else ts_field
                    fill = h5[var_name].attrs.get("_FillValue", np.array([1.0e15], dtype=np.float32))
                    fill = float(np.asarray(fill).reshape(-1)[0])
                    field = np.where(np.abs(field - fill) < 1.0, np.nan, field).astype(np.float32)
                    for row in time_rows:
                        matched_time[row] = time_seconds[time_idx]
                        start = row * lat.shape[1]
                        stop = start + lat.shape[1]
                        row_valid = valid_space[start:stop]
                        values = np.full(lat.shape[1], np.nan, dtype=np.float32)
                        values[row_valid] = field[flat_y[start:stop][row_valid], flat_x[start:stop][row_valid]]
                        target[row, :] = values
            source_files.add(str(merra2_file))

    return t2m, ts, matched_time, sorted(source_files), "; ".join(notes)


def sample_autosnow(scan_seconds, lat, lon, autosnow_dir):
    values = np.full(lat.shape, AUTOSNOW_FILL, dtype=np.uint8)
    yyyydoy = np.full(scan_seconds.shape, -1, dtype=np.int32)
    source_files = set()
    missing_dates = []
    notes = []

    scan_dates = [utc_from_seconds(sec).date() for sec in scan_seconds]
    flat_lat = lat.reshape(-1)
    flat_lon = lon.reshape(-1)
    valid_space = np.isfinite(flat_lat) & np.isfinite(flat_lon)

    for date_value in sorted(set(scan_dates)):
        rows = np.where(np.array(scan_dates, dtype=object) == date_value)[0]
        doy = date_value.timetuple().tm_yday
        yyyydoy[rows] = date_value.year * 1000 + doy
        autosnow_file = mf.autosnow_file_for_date(autosnow_dir, date_value)
        if not autosnow_file.exists():
            missing_dates.append(date_value.isoformat())
            continue

        with h5py.File(autosnow_file, "r") as h5:
            snow_lat = h5["lat"][:]
            snow_lon = h5["lon"][:]
            flat_y = mf.nearest_indices(flat_lat, snow_lat)
            flat_x = mf.nearest_indices(mf.normalize_lon(flat_lon, "minus180_180"), snow_lon)
            snow = h5["autosnow_class"][:]
            for row in rows:
                start = row * lat.shape[1]
                stop = start + lat.shape[1]
                row_valid = valid_space[start:stop]
                row_values = np.full(lat.shape[1], AUTOSNOW_FILL, dtype=np.uint8)
                row_values[row_valid] = snow[flat_y[start:stop][row_valid], flat_x[start:stop][row_valid]]
                values[row, :] = row_values
            source_files.add(str(autosnow_file))

    if missing_dates:
        notes.append(f"missing AutoSnow dates: {', '.join(missing_dates)}")
    return values, yyyydoy, sorted(source_files), missing_dates, "; ".join(notes)


def _nan_stat(values, func):
    values = np.asarray(values)
    if values.size == 0 or not np.any(np.isfinite(values)):
        return np.nan
    return float(func(values))


def _nan_percentile(values, percentile):
    return _nan_stat(values, lambda x: np.nanpercentile(x, percentile))


def _nan_max_index(values):
    values = np.asarray(values)
    if values.size == 0 or not np.any(np.isfinite(values)):
        return ""
    idx = np.unravel_index(int(np.nanargmax(values)), values.shape)
    return ",".join(str(value) for value in idx)


def _max_index_tuple(values):
    values = np.asarray(values)
    if values.size == 0 or not np.any(np.isfinite(values)):
        return None
    return np.unravel_index(int(np.nanargmax(values)), values.shape)


def _geometry_values_at_index(v7, v8, idx):
    if idx is None:
        return (np.nan, np.nan, np.nan, np.nan)
    return (
        float(v7["lon"][idx]),
        float(v7["lat"][idx]),
        float(v8["lon"][idx]),
        float(v8["lat"][idx]),
    )


def geometry_note(qc):
    return (
        f"{qc['classification']}; "
        f"max lat diff {qc['max_lat_diff_deg']:g} deg, "
        f"max wrapped lon diff {qc['max_wrapped_lon_diff_deg']:g} deg, "
        f"max raw lon diff {qc['max_raw_lon_diff_deg']:g} deg, "
        f"max scan-time diff {qc['max_scan_time_diff_seconds']:g} s, "
        f"pixels exceeding tolerance {qc['pixels_exceeding_tolerance']} "
        f"({qc['percent_pixels_exceeding_tolerance']:g}%), "
        f"edge-only {qc['edge_only_exceedances']}"
    )


def classify_geometry(v7, v8, pair=None, tolerance_deg=5.0e-3):
    orbit_id = "" if pair is None else pair["orbit_id"]
    ymd = "" if pair is None else pair["ymd"]
    qc = {
        "orbit_id": orbit_id,
        "date": ymd,
        "classification": GEOMETRY_FAIL_SHAPE_OR_TIME,
        "max_lat_diff_deg": np.nan,
        "max_wrapped_lon_diff_deg": np.nan,
        "max_raw_lon_diff_deg": np.nan,
        "median_lat_diff_deg": np.nan,
        "p99_lat_diff_deg": np.nan,
        "p999_lat_diff_deg": np.nan,
        "median_wrapped_lon_diff_deg": np.nan,
        "p99_wrapped_lon_diff_deg": np.nan,
        "p999_wrapped_lon_diff_deg": np.nan,
        "pixels_exceeding_tolerance": 0,
        "percent_pixels_exceeding_tolerance": np.nan,
        "edge_only_exceedances": False,
        "max_lat_diff_index": "",
        "max_wrapped_lon_diff_index": "",
        "max_raw_lon_diff_index": "",
        "max_lat_diff_v7_lon": np.nan,
        "max_lat_diff_v7_lat": np.nan,
        "max_lat_diff_v8_lon": np.nan,
        "max_lat_diff_v8_lat": np.nan,
        "max_wrapped_lon_diff_v7_lon": np.nan,
        "max_wrapped_lon_diff_v7_lat": np.nan,
        "max_wrapped_lon_diff_v8_lon": np.nan,
        "max_wrapped_lon_diff_v8_lat": np.nan,
        "max_raw_lon_diff_v7_lon": np.nan,
        "max_raw_lon_diff_v7_lat": np.nan,
        "max_raw_lon_diff_v8_lon": np.nan,
        "max_raw_lon_diff_v8_lat": np.nan,
        "max_scan_time_diff_seconds": np.nan,
        "final_action": "skipped",
        "note": "",
    }

    if v7["lat"].shape != v8["lat"].shape or v7["lon"].shape != v8["lon"].shape:
        qc["note"] = "shape differs"
        return qc
    if v7["scan_time_seconds"].shape != v8["scan_time_seconds"].shape:
        qc["note"] = "scan-time shape differs"
        return qc

    lat_diff = np.abs(v7["lat"] - v8["lat"])
    raw_lon_diff = np.abs(v7["lon"] - v8["lon"])
    lon_diff = np.abs(((v8["lon"] - v7["lon"] + 180.0) % 360.0) - 180.0)
    time_diff = np.abs(v7["scan_time_seconds"] - v8["scan_time_seconds"])
    valid_pixels = np.isfinite(lat_diff) & np.isfinite(lon_diff)
    valid_count = int(np.count_nonzero(valid_pixels))
    exceed = valid_pixels & ((lat_diff > tolerance_deg) | (lon_diff > tolerance_deg))
    exceed_count = int(np.count_nonzero(exceed))
    exceed_fraction = exceed_count / valid_count if valid_count else np.nan
    exceed_pixels = np.where(exceed)[1] if exceed_count else np.array([], dtype=int)
    npixel = v7["lat"].shape[1]
    edge_margin = min(GEOMETRY_MINOR_EDGE_PIXEL_MARGIN, max(npixel // 2, 1))
    edge_only = bool(
        exceed_count == 0
        or np.all((exceed_pixels < edge_margin) | (exceed_pixels >= npixel - edge_margin))
    )
    max_lat_idx = _max_index_tuple(lat_diff)
    max_wrapped_lon_idx = _max_index_tuple(lon_diff)
    max_raw_lon_idx = _max_index_tuple(raw_lon_diff)
    max_lat_values = _geometry_values_at_index(v7, v8, max_lat_idx)
    max_wrapped_lon_values = _geometry_values_at_index(v7, v8, max_wrapped_lon_idx)
    max_raw_lon_values = _geometry_values_at_index(v7, v8, max_raw_lon_idx)

    qc.update(
        {
            "max_lat_diff_deg": _nan_stat(lat_diff, np.nanmax),
            "max_wrapped_lon_diff_deg": _nan_stat(lon_diff, np.nanmax),
            "max_raw_lon_diff_deg": _nan_stat(raw_lon_diff, np.nanmax),
            "median_lat_diff_deg": _nan_percentile(lat_diff, 50.0),
            "p99_lat_diff_deg": _nan_percentile(lat_diff, 99.0),
            "p999_lat_diff_deg": _nan_percentile(lat_diff, 99.9),
            "median_wrapped_lon_diff_deg": _nan_percentile(lon_diff, 50.0),
            "p99_wrapped_lon_diff_deg": _nan_percentile(lon_diff, 99.0),
            "p999_wrapped_lon_diff_deg": _nan_percentile(lon_diff, 99.9),
            "pixels_exceeding_tolerance": exceed_count,
            "percent_pixels_exceeding_tolerance": (
                float(exceed_fraction * 100.0) if np.isfinite(exceed_fraction) else np.nan
            ),
            "edge_only_exceedances": edge_only,
            "max_lat_diff_index": _nan_max_index(lat_diff),
            "max_wrapped_lon_diff_index": _nan_max_index(lon_diff),
            "max_raw_lon_diff_index": _nan_max_index(raw_lon_diff),
            "max_lat_diff_v7_lon": max_lat_values[0],
            "max_lat_diff_v7_lat": max_lat_values[1],
            "max_lat_diff_v8_lon": max_lat_values[2],
            "max_lat_diff_v8_lat": max_lat_values[3],
            "max_wrapped_lon_diff_v7_lon": max_wrapped_lon_values[0],
            "max_wrapped_lon_diff_v7_lat": max_wrapped_lon_values[1],
            "max_wrapped_lon_diff_v8_lon": max_wrapped_lon_values[2],
            "max_wrapped_lon_diff_v8_lat": max_wrapped_lon_values[3],
            "max_raw_lon_diff_v7_lon": max_raw_lon_values[0],
            "max_raw_lon_diff_v7_lat": max_raw_lon_values[1],
            "max_raw_lon_diff_v8_lon": max_raw_lon_values[2],
            "max_raw_lon_diff_v8_lat": max_raw_lon_values[3],
            "max_scan_time_diff_seconds": _nan_stat(time_diff, np.nanmax),
        }
    )

    time_ok = qc["max_scan_time_diff_seconds"] <= GEOMETRY_TIME_TOLERANCE_SECONDS
    if not time_ok:
        qc["classification"] = GEOMETRY_FAIL_SHAPE_OR_TIME
    elif exceed_count == 0:
        qc["classification"] = GEOMETRY_PASS_NORMAL
        qc["final_action"] = "collocated"
    else:
        max_spatial_diff = max(qc["max_lat_diff_deg"], qc["max_wrapped_lon_diff_deg"])
        minor_edge_warning = (
            np.isfinite(exceed_fraction)
            and exceed_fraction <= GEOMETRY_MINOR_MAX_EXCEEDANCE_FRACTION
            and edge_only
            and max_spatial_diff <= GEOMETRY_MINOR_HARD_LIMIT_DEG
        )
        if minor_edge_warning:
            qc["classification"] = GEOMETRY_PASS_MINOR_EDGE_WARNING
            qc["final_action"] = "collocated_with_warning"
        else:
            qc["classification"] = GEOMETRY_FAIL_TRUE_MISMATCH

    qc["note"] = geometry_note(qc)
    return qc


def geometry_matches(v7, v8, tolerance_deg=5.0e-3):
    qc = classify_geometry(v7, v8, tolerance_deg=tolerance_deg)
    return qc["final_action"] != "skipped", qc["note"]


def apply_lat_filter(arrays, lat, lat_abs_min):
    if lat_abs_min is None:
        return arrays
    keep = np.abs(lat) >= float(lat_abs_min)
    filtered = {}
    for name, array in arrays.items():
        if array.ndim == 2 and array.shape == lat.shape:
            if array.dtype == np.uint8:
                filtered[name] = np.where(keep, array, AUTOSNOW_FILL).astype(array.dtype)
            else:
                filtered[name] = np.where(keep, array, np.nan).astype(array.dtype)
        else:
            filtered[name] = array
    return filtered


def make_dataset(pair, v7, v8, era5, merra2, autosnow, notes, lat_abs_min):
    lat = v8["lat"]
    lon = v8["lon"]
    nscan, npixel = lat.shape

    era5_tp, era5_time, era5_files, era5_notes = era5
    merra2_t2m, merra2_ts, merra2_time, merra2_files, merra2_notes = merra2
    autosnow_class, autosnow_yyyydoy, autosnow_files, missing_autosnow_dates, autosnow_notes = autosnow

    arrays = apply_lat_filter(
        {
            "lat": lat,
            "lon": lon,
            "gmi_v7_surface_precipitation": v7["surface_precipitation"],
            "gmi_v8_surface_precipitation": v8["surface_precipitation"],
            "era5_tp": era5_tp,
            "merra2_t2m": merra2_t2m,
            "merra2_ts": merra2_ts,
            "autosnow_class": autosnow_class,
        },
        lat,
        lat_abs_min,
    )

    scan_dates = [utc_from_seconds(sec).date() for sec in v8["scan_time_seconds"]]
    autosnow_date = np.array(
        [(dt.datetime.combine(value, dt.time()) - mf.EPOCH).total_seconds() for value in scan_dates],
        dtype=np.float64,
    )

    ds = xr.Dataset(
        data_vars={
            "lat": (("scan", "pixel"), arrays["lat"], {"units": "degrees_north"}),
            "lon": (("scan", "pixel"), arrays["lon"], {"units": "degrees_east"}),
            "gmi_scan_time": (
                ("scan",),
                v8["scan_time_seconds"],
                {
                    "standard_name": "time",
                    "long_name": "GMI scan time",
                    "units": "seconds since 1970-01-01 00:00:00",
                    "calendar": "standard",
                },
            ),
            "gmi_v7_surface_precipitation": (
                ("scan", "pixel"),
                arrays["gmi_v7_surface_precipitation"],
                {
                    "long_name": "GMI GPROF V7 S1 surface precipitation",
                    "units": v7["attrs"]["surfacePrecipitation_units"],
                    "source_variable": "S1/surfacePrecipitation",
                },
            ),
            "gmi_v8_surface_precipitation": (
                ("scan", "pixel"),
                arrays["gmi_v8_surface_precipitation"],
                {
                    "long_name": "GMI GPROF V8 S1 surface precipitation",
                    "units": v8["attrs"]["surfacePrecipitation_units"],
                    "source_variable": "S1/surfacePrecipitation",
                },
            ),
            "era5_tp": (
                ("scan", "pixel"),
                arrays["era5_tp"],
                {
                    "long_name": "ERA5 hourly total precipitation sampled at GMI footprint",
                    "units": "mm",
                    "source_variable": "tp",
                    "comment": "ERA5 source units were meters and values were multiplied by 1000 when source metadata reported units=m.",
                },
            ),
            "era5_time": (
                ("scan",),
                era5_time,
                {
                    "standard_name": "time",
                    "long_name": "Matched ERA5 valid time",
                    "units": "seconds since 1970-01-01 00:00:00",
                    "calendar": "standard",
                },
            ),
            "merra2_t2m": (
                ("scan", "pixel"),
                arrays["merra2_t2m"],
                {
                    "long_name": "MERRA2 2-meter air temperature sampled at GMI footprint",
                    "units": "K",
                    "source_variable": "T2M",
                },
            ),
            "merra2_ts": (
                ("scan", "pixel"),
                arrays["merra2_ts"],
                {
                    "long_name": "MERRA2 surface skin temperature sampled at GMI footprint",
                    "units": "K",
                    "source_variable": "TS",
                },
            ),
            "merra2_time": (
                ("scan",),
                merra2_time,
                {
                    "standard_name": "time",
                    "long_name": "Matched MERRA2 time",
                    "units": "seconds since 1970-01-01 00:00:00",
                    "calendar": "standard",
                },
            ),
            "autosnow_class": (
                ("scan", "pixel"),
                arrays["autosnow_class"],
                {
                    "long_name": "GMASI AutoSnow snow/ice class sampled at GMI footprint",
                    "source_variable": "autosnow_class",
                    "valid_source_values": "0, 1, 2, 3",
                    "missing_value": int(AUTOSNOW_FILL),
                    "comment": "Raw daily AutoSnow class values preserved; missing daily files or invalid footprint locations are stored as 255.",
                },
            ),
            "autosnow_date": (
                ("scan",),
                autosnow_date,
                {
                    "standard_name": "time",
                    "long_name": "AutoSnow daily map date used for sampling",
                    "units": "seconds since 1970-01-01 00:00:00",
                    "calendar": "standard",
                },
            ),
            "autosnow_yyyydoy": (
                ("scan",),
                autosnow_yyyydoy,
                {"long_name": "AutoSnow date as YYYYDDD"},
            ),
            "gmi_scan_year": (
                ("scan",),
                np.array([value.year for value in scan_dates], dtype=np.int16),
            ),
            "gmi_scan_doy": (
                ("scan",),
                np.array([value.timetuple().tm_yday for value in scan_dates], dtype=np.int16),
            ),
            "gmi_scan_hour": (
                ("scan",),
                np.array([utc_from_seconds(sec).hour for sec in v8["scan_time_seconds"]], dtype=np.int8),
            ),
        },
        coords={
            "scan": ("scan", np.arange(nscan, dtype=np.uint16)),
            "pixel": ("pixel", np.arange(npixel, dtype=np.uint16)),
        },
        attrs={
            "title": "GMI GPROF V7/V8 footprint collocation with ERA5, MERRA2, and AutoSnow",
            "source_gmi_v7_file": pair["v7_file"],
            "source_gmi_v8_file": pair["v8_file"],
            "source_era5_files": "; ".join(era5_files),
            "source_merra2_files": "; ".join(merra2_files),
            "source_autosnow_files": "; ".join(autosnow_files),
            "collocation_method": "Native GMI S1 footprints with nearest-neighbor spatial sampling of ancillary grids.",
            "time_matching_method": "Nearest hourly/available ancillary time to each GMI scan time; AutoSnow uses the GMI scan date.",
            "spatial_matching_method": "Nearest neighbor in latitude/longitude after normalizing longitude to each source convention.",
            "missing_autosnow_handling": "Missing AutoSnow dates are stored as autosnow_class=255 and logged.",
            "v7_v8_pairing_method": "Matched by GMI filename date/start/end time and orbit number.",
            "orbit_id": pair["orbit_id"],
            "orbit_start_time": pair["start"].isoformat(),
            "orbit_end_time": pair["end"].isoformat(),
            "creation_time": dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds") + "Z",
            "software_script_name": SCRIPT_NAME,
            "lat_abs_min": "none" if lat_abs_min is None else str(lat_abs_min),
            "autosnow_product_note": "AutoSnow is a daily product sampled at each GMI footprint location for the overpass date.",
            "era5_interpretation_note": "ERA5 tp metadata reports GRIB_stepType=accum; output preserves matched valid time and stores tp in mm.",
            "processing_notes": "; ".join(note for note in [notes, era5_notes, merra2_notes, autosnow_notes] if note),
            "missing_autosnow_dates": "; ".join(missing_autosnow_dates),
        },
    )

    return ds


def output_path_for_pair(output_dir, pair):
    name = f"GMI_GPROF_V7_V8_ERA5_MERRA2_AutoSnow_{pair['orbit_id']}_{pair['ymd']}.nc"
    return Path(output_dir) / name


def choose_netcdf_engine():
    if has_module("netCDF4"):
        return "netcdf4"
    if has_module("h5netcdf"):
        return "h5netcdf"
    return "scipy"


def write_dataset(ds, output_file):
    engine = choose_netcdf_engine()
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    ds.attrs["netcdf_write_engine"] = engine

    if engine in {"netcdf4", "h5netcdf"}:
        encoding = {}
        for name, da in ds.data_vars.items():
            if da.ndim == 2:
                encoding[name] = {"zlib": True, "complevel": 4, "shuffle": True}
                if name == "autosnow_class":
                    encoding[name]["dtype"] = "uint8"
                    encoding[name]["_FillValue"] = int(AUTOSNOW_FILL)
                elif np.issubdtype(da.dtype, np.floating):
                    encoding[name]["dtype"] = "float32"
                    encoding[name]["_FillValue"] = np.nan
            elif name == "autosnow_class":
                encoding[name] = {"dtype": "uint8", "_FillValue": int(AUTOSNOW_FILL)}
        ds.to_netcdf(output_file, engine=engine, format="NETCDF4", encoding=encoding)
        return engine

    ds_out = ds.copy()
    ds_out.attrs["netcdf_write_note"] = (
        "netCDF4/h5netcdf engines were unavailable; wrote xarray-readable NETCDF3 via scipy without compression."
    )
    if ds_out["autosnow_class"].dtype == np.uint8:
        ds_out["autosnow_class"] = ds_out["autosnow_class"].astype(np.int16)
        ds_out["autosnow_class"].attrs["missing_value"] = int(AUTOSNOW_FILL)
    ds_out.to_netcdf(
        output_file,
        engine="scipy",
        format="NETCDF3_64BIT",
        encoding={"autosnow_class": {"_FillValue": int(AUTOSNOW_FILL)}},
    )
    return engine


def collocate_pair(pair, args_dict):
    output_file = output_path_for_pair(args_dict["output_dir"], pair)
    v7 = mf.read_gmi_s1(pair["v7_file"])
    v8 = mf.read_gmi_s1(pair["v8_file"])
    geometry_qc = classify_geometry(v7, v8, pair, args_dict["geometry_tolerance_deg"])
    geom_note = geometry_qc["note"]
    if geometry_qc["final_action"] == "skipped":
        return {
            "status": "failed_geometry",
            "output_file": str(output_file),
            "orbit_id": pair["orbit_id"],
            "geometry_qc": geometry_qc,
            "error": f"V7/V8 geometry mismatch for {pair['orbit_id']}: {geom_note}",
        }
    if output_file.exists() and not args_dict["overwrite"]:
        return {
            "status": "skipped_existing",
            "output_file": str(output_file),
            "orbit_id": pair["orbit_id"],
            "geometry_qc": geometry_qc,
        }

    lat = v8["lat"]
    lon = v8["lon"]
    era5_files = build_era5_index(args_dict["era5_dir"])

    era5 = sample_era5(v8["scan_time_seconds"], lat, lon, era5_files)
    merra2 = sample_merra2(v8["scan_time_seconds"], lat, lon, args_dict["merra2_dir"])
    autosnow = sample_autosnow(v8["scan_time_seconds"], lat, lon, args_dict["autosnow_dir"])
    ds = make_dataset(pair, v7, v8, era5, merra2, autosnow, geom_note, args_dict["lat_abs_min"])

    engine = write_dataset(ds, output_file)
    return {
        "status": "ok",
        "output_file": str(output_file),
        "orbit_id": pair["orbit_id"],
        "engine": engine,
        "missing_autosnow_dates": ";".join(autosnow[3]),
        "geometry_qc": geometry_qc,
    }


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary(path, lines):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Collocate matched GMI GPROF V7/V8 footprints with ERA5, MERRA2, and AutoSnow."
    )
    parser.add_argument("--year-start", type=int, default=2014)
    parser.add_argument("--year-end", type=int, default=2025)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument(
        "--orbit-ids",
        nargs="+",
        default=None,
        help="Optional orbit IDs to process, e.g. 016729_20170206 016738_20170207. Comma-separated values are also accepted.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory for run logs. Defaults to <output-dir>/QC_logs.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional log filename suffix. Defaults to UTC YYYYMMDD_HHMMSS.",
    )
    parser.add_argument("--lat-abs-min", type=float, default=None)
    parser.add_argument("--gmi-v7-dir", default=DEFAULT_GMI_V7_DIR)
    parser.add_argument("--gmi-v8-dir", default=DEFAULT_GMI_V8_DIR)
    parser.add_argument("--era5-dir", default=DEFAULT_ERA5_DIR)
    parser.add_argument("--merra2-dir", default=DEFAULT_MERRA2_DIR)
    parser.add_argument("--autosnow-dir", default=DEFAULT_AUTOSNOW_DIR)
    parser.add_argument(
        "--geometry-tolerance-deg",
        type=float,
        default=5.0e-3,
        help="Maximum allowed absolute V7/V8 lat/lon difference before skipping an orbit.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_start = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    run_id = args.run_id or run_start.strftime("%Y%m%d_%H%M%S")
    year_tag = f"{args.year_start}_{args.year_end}"

    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir) if args.log_dir is not None else output_dir / DEFAULT_QC_LOG_SUBDIR
    log_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_paths = {
        "failed": log_dir / f"collocation_failed_orbits_{year_tag}_{run_id}.csv",
        "geometry_qc": log_dir / f"collocation_geometry_qc_{year_tag}_{run_id}.csv",
        "unmatched": log_dir / f"collocation_unmatched_v7_v8_{year_tag}_{run_id}.csv",
        "missing_autosnow": log_dir / f"collocation_missing_autosnow_dates_{year_tag}_{run_id}.csv",
        "summary": log_dir / f"collocation_production_summary_{year_tag}_{run_id}.txt",
    }

    pairs_all, unmatched_v7, unmatched_v8 = discover_gmi_pairs(
        args.gmi_v7_dir,
        args.gmi_v8_dir,
        args.year_start,
        args.year_end,
    )
    pairs = pairs_all
    orbit_ids = None
    missing_requested_orbits = []
    if args.orbit_ids is not None:
        orbit_ids = []
        for value in args.orbit_ids:
            orbit_ids.extend(part.strip() for part in value.split(",") if part.strip())
        pairs_by_id = {pair["orbit_id"]: pair for pair in pairs_all}
        pairs = []
        for orbit_id in orbit_ids:
            if orbit_id in pairs_by_id:
                pairs.append(pairs_by_id[orbit_id])
            else:
                missing_requested_orbits.append(orbit_id)
    if args.max_files is not None:
        pairs = pairs[: args.max_files]

    write_csv(
        log_paths["unmatched"],
        [{"version": "v7", "pair_key": key} for key in unmatched_v7]
        + [{"version": "v8", "pair_key": key} for key in unmatched_v8],
        ["version", "pair_key"],
    )

    print(f"Matched pairs selected: {len(pairs)}")
    print(f"Unmatched V7 files in range: {len(unmatched_v7)}")
    print(f"Unmatched V8 files in range: {len(unmatched_v8)}")
    print(f"Output directory: {output_dir}")
    print(f"Log directory: {log_dir}")
    print(f"Run ID: {run_id}")
    print(f"NetCDF writer engine available for this run: {choose_netcdf_engine()}")
    print(f"Unmatched-file log: {log_paths['unmatched']}")
    print(f"Failed-orbit log: {log_paths['failed']}")
    print(f"Geometry-QC log: {log_paths['geometry_qc']}")
    print(f"Missing-AutoSnow log: {log_paths['missing_autosnow']}")
    print(f"Summary log: {log_paths['summary']}")
    if orbit_ids is not None:
        print(f"Requested orbit IDs: {', '.join(orbit_ids)}")
    if missing_requested_orbits:
        print(f"Requested orbit IDs not found: {', '.join(missing_requested_orbits)}")

    args_dict = vars(args)
    failures = []
    geometry_qc_rows = []
    missing_autosnow = []
    status_counts = {"ok": 0, "skipped_existing": 0, "failed_geometry": 0, "failed": 0}

    if not pairs:
        print("No matched pairs selected.")
    elif args.workers == 1:
        for pair in progress(pairs, total=len(pairs)):
            print(f"Processing orbit {pair['orbit_id']} {pair['start'].isoformat()}")
            try:
                result = collocate_pair(pair, args_dict)
                status_counts[result["status"]] = status_counts.get(result["status"], 0) + 1
                print(f"{result['status']}: {result['orbit_id']} -> {result['output_file']}")
                if result.get("geometry_qc"):
                    geometry_qc_rows.append(result["geometry_qc"])
                if result["status"] == "failed_geometry":
                    failures.append(
                        {
                            "orbit_id": pair["orbit_id"],
                            "date": pair["ymd"],
                            "classification": result["geometry_qc"]["classification"],
                            "reason": result["geometry_qc"]["note"],
                            "v7_file": pair["v7_file"],
                            "v8_file": pair["v8_file"],
                            "error": result["error"],
                        }
                    )
                if result.get("missing_autosnow_dates"):
                    missing_autosnow.append(
                        {
                            "orbit_id": result["orbit_id"],
                            "missing_dates": result["missing_autosnow_dates"],
                        }
                    )
            except Exception as exc:
                print(f"failed: {pair['orbit_id']} {exc}")
                status_counts["failed"] += 1
                failures.append(
                    {
                        "orbit_id": pair["orbit_id"],
                        "date": pair["ymd"],
                        "classification": "PROCESSING_FAILURE",
                        "reason": repr(exc),
                        "v7_file": pair["v7_file"],
                        "v8_file": pair["v8_file"],
                        "error": repr(exc),
                    }
                )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(collocate_pair, pair, args_dict): pair for pair in pairs}
            for future in progress(as_completed(futures), total=len(futures)):
                pair = futures[future]
                try:
                    result = future.result()
                    status_counts[result["status"]] = status_counts.get(result["status"], 0) + 1
                    print(f"{result['status']}: {result['orbit_id']} -> {result['output_file']}")
                    if result.get("geometry_qc"):
                        geometry_qc_rows.append(result["geometry_qc"])
                    if result["status"] == "failed_geometry":
                        failures.append(
                            {
                                "orbit_id": pair["orbit_id"],
                                "date": pair["ymd"],
                                "classification": result["geometry_qc"]["classification"],
                                "reason": result["geometry_qc"]["note"],
                                "v7_file": pair["v7_file"],
                                "v8_file": pair["v8_file"],
                                "error": result["error"],
                            }
                        )
                    if result.get("missing_autosnow_dates"):
                        missing_autosnow.append(
                            {
                                "orbit_id": result["orbit_id"],
                                "missing_dates": result["missing_autosnow_dates"],
                            }
                        )
                except Exception as exc:
                    print(f"failed: {pair['orbit_id']} {exc}")
                    status_counts["failed"] += 1
                    failures.append(
                        {
                            "orbit_id": pair["orbit_id"],
                            "date": pair["ymd"],
                            "classification": "PROCESSING_FAILURE",
                            "reason": repr(exc),
                            "v7_file": pair["v7_file"],
                            "v8_file": pair["v8_file"],
                            "error": repr(exc),
                        }
                    )

    write_csv(
        log_paths["failed"],
        failures,
        FAILED_ORBIT_FIELDNAMES,
    )
    write_csv(log_paths["geometry_qc"], geometry_qc_rows, GEOMETRY_QC_FIELDNAMES)
    write_csv(
        log_paths["missing_autosnow"],
        missing_autosnow,
        ["orbit_id", "missing_dates"],
    )

    run_end = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    geometry_class_counts = {
        name: sum(1 for row in geometry_qc_rows if row.get("classification") == name)
        for name in [
            GEOMETRY_PASS_NORMAL,
            GEOMETRY_PASS_MINOR_EDGE_WARNING,
            GEOMETRY_FAIL_TRUE_MISMATCH,
            GEOMETRY_FAIL_SHAPE_OR_TIME,
        ]
    }
    output_file_count = len(list(output_dir.glob("GMI_GPROF_V7_V8_ERA5_MERRA2_AutoSnow_*.nc")))
    summary_lines = [
        "GMI V7/V8 ERA5 MERRA2 AutoSnow Collocation Summary",
        f"run_id: {run_id}",
        f"script: {SCRIPT_NAME}",
        f"start_utc: {run_start.isoformat(timespec='seconds')}Z",
        f"end_utc: {run_end.isoformat(timespec='seconds')}Z",
        f"year_start: {args.year_start}",
        f"year_end: {args.year_end}",
        f"max_files: {args.max_files}",
        f"orbit_ids: {'none' if orbit_ids is None else ','.join(orbit_ids)}",
        f"workers: {args.workers}",
        f"overwrite: {args.overwrite}",
        f"output_dir: {output_dir}",
        f"log_dir: {log_dir}",
        f"gmi_v7_dir: {args.gmi_v7_dir}",
        f"gmi_v8_dir: {args.gmi_v8_dir}",
        f"era5_dir: {args.era5_dir}",
        f"merra2_dir: {args.merra2_dir}",
        f"autosnow_dir: {args.autosnow_dir}",
        f"netcdf_write_engine: {choose_netcdf_engine()}",
        f"matched_pairs_in_range: {len(pairs_all)}",
        f"matched_pairs_selected: {len(pairs)}",
        f"successfully_collocated_this_run: {status_counts.get('ok', 0)}",
        f"pass_normal: {geometry_class_counts[GEOMETRY_PASS_NORMAL]}",
        f"pass_minor_edge_warning: {geometry_class_counts[GEOMETRY_PASS_MINOR_EDGE_WARNING]}",
        f"fail_true_geometry_mismatch: {geometry_class_counts[GEOMETRY_FAIL_TRUE_MISMATCH]}",
        f"fail_shape_or_time_mismatch: {geometry_class_counts[GEOMETRY_FAIL_SHAPE_OR_TIME]}",
        f"other_processing_failures: {status_counts.get('failed', 0)}",
        f"output_file_count: {output_file_count}",
        f"unmatched_v7_files: {len(unmatched_v7)}",
        f"unmatched_v8_files: {len(unmatched_v8)}",
        f"ok_outputs: {status_counts.get('ok', 0)}",
        f"skipped_existing_outputs: {status_counts.get('skipped_existing', 0)}",
        f"failed_geometry_orbits: {status_counts.get('failed_geometry', 0)}",
        f"failed_orbits: {status_counts.get('failed', 0)}",
        f"orbits_with_missing_autosnow_dates: {len(missing_autosnow)}",
        f"failed_orbit_log: {log_paths['failed']}",
        f"geometry_qc_log: {log_paths['geometry_qc']}",
        f"unmatched_file_log: {log_paths['unmatched']}",
        f"missing_autosnow_log: {log_paths['missing_autosnow']}",
    ]
    write_summary(log_paths["summary"], summary_lines)
    print("\n".join(summary_lines))
    print(f"Wrote summary: {log_paths['summary']}")


if __name__ == "__main__":
    main()

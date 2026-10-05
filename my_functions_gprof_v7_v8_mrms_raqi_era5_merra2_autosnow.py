#!/usr/bin/env python3
"""Shared science and I/O helpers for generalized GPROF V7/V8 collocation."""

from __future__ import annotations

import datetime as dt
import gzip
import json
import os
import re
import sys
from pathlib import Path

import eccodes
import h5py
import numpy as np
import xarray as xr

# Some login shells on this system export PROJ_LIB from another Conda env.
# Pin rasterio/pyproj to the database belonging to the running interpreter.
_PROJ_DATA = Path(sys.prefix) / "share" / "proj"
if (_PROJ_DATA / "proj.db").exists():
    os.environ["PROJ_DATA"] = str(_PROJ_DATA)
    os.environ["PROJ_LIB"] = str(_PROJ_DATA)
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject

EPOCH = dt.datetime(1970, 1, 1)
FILL_FLOAT = np.float32(np.nan)
AUTOSNOW_FILL = np.uint8(255)
TIME_UNITS = "seconds since 1970-01-01 00:00:00 UTC"

DEFAULT_OUTPUT_ROOT = Path(
    "/scratch/kkumah/GPROF_V7_V8_MRMS_RAQI_ERA5_MERRA2_AutoSnow_matchups"
)

SENSOR_CONFIGS = {
    "GMI": {
        "sensor": "GMI", "platform": "GPM", "group": "S1",
        "v7_root": "/scratch/kkumah/GPM_GMI/V7",
        "v8_root": "/scratch/kkumah/GPM_GMI/V8",
        "v7_pattern": "*.HDF5", "v8_pattern": "*.nc",
        "latitude_variable": "S1/Latitude", "longitude_variable": "S1/Longitude",
        "scan_time_variable": "S1/ScanTime", "precipitation_variable": "S1/surfacePrecipitation",
    },
    "MHS_METOPB": {
        "sensor": "MHS", "platform": "METOPB", "group": "S1",
        "v7_root": "/scratch/kkumah/GPM-Constellation-Satellites_MI_and_Sounders/L2_Hourly/V7/MHS/MetopB",
        "v8_root": "/scratch/kkumah/GPM-Constellation-Satellites_MI_and_Sounders/L2_Hourly/V8/MHS/MetopB",
        "v7_pattern": "*.HDF5", "v8_pattern": "*.nc",
        "latitude_variable": "S1/Latitude", "longitude_variable": "S1/Longitude",
        "scan_time_variable": "S1/ScanTime", "precipitation_variable": "S1/surfacePrecipitation",
    },
    "SSMIS_F18": {
        "sensor": "SSMIS", "platform": "F18", "group": "S1",
        "v7_root": "/scratch/kkumah/GPM-Constellation-Satellites_MI_and_Sounders/L2_Hourly/V7/DMSP-SSMIS/F18",
        "v8_root": "/scratch/kkumah/GPM-Constellation-Satellites_MI_and_Sounders/L2_Hourly/V8/DMSP-SSMIS/F18",
        "v7_pattern": "*.HDF5", "v8_pattern": "*.nc",
        "latitude_variable": "S1/Latitude", "longitude_variable": "S1/Longitude",
        "scan_time_variable": "S1/ScanTime", "precipitation_variable": "S1/surfacePrecipitation",
    },
}

ANCILLARY_ROOTS = {
    "mrms": "/scratch/omidzandi/MRMS/MultiSensor_QPE_01H_Pass2",
    "raqi": "/scratch/kkumah/MRMS/RAQI",
    "era5": "/scratch/kkumah/ERA5_tp_hourly",
    "merra2": "/ra1/pubdat/AVHRR_CloudSat_proj/MERRA2/merra2_archive_19800101_20251231",
    "autosnow": "/scratch/kkumah/Autosnow_2014_to_2024_nc",
}


def as_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return as_text(value.reshape(-1)[0]) if value.size == 1 else ", ".join(map(str, value))
    if isinstance(value, np.generic):
        return value.item()
    return value


def sensor_config(name):
    config = dict(SENSOR_CONFIGS[name])
    config["output_dir"] = str(DEFAULT_OUTPUT_ROOT / "matchups" / name)
    return config


def parse_granule_filename(path):
    """Legacy pairing key: filename date/start/end plus orbit number."""
    name = Path(path).name
    parts = name.split(".")
    if len(parts) < 6:
        raise ValueError(f"Unrecognized GPROF filename: {name}")
    match = re.fullmatch(r"(\d{8})-S(\d{6})-E(\d{6})", parts[4])
    if not match:
        raise ValueError(f"Unrecognized GPROF date/time field: {name}")
    ymd, start_hms, end_hms = match.groups()
    start = dt.datetime.strptime(ymd + start_hms, "%Y%m%d%H%M%S")
    end = dt.datetime.strptime(ymd + end_hms, "%Y%m%d%H%M%S")
    if end < start:
        end += dt.timedelta(days=1)
    return {
        "filename": name, "ymd": ymd, "date": start.date(), "year": start.year,
        "start": start, "end": end, "orbit": parts[5],
        "pair_key": f"{ymd}-S{start_hms}-E{end_hms}.{parts[5]}",
        "orbit_id": f"{parts[5]}_{ymd}",
    }


def discover_pairs(config, allowed_dates=None):
    def index(root, pattern):
        result = {}
        files = sorted(Path(root).glob(pattern))
        for path in files:
            try:
                info = parse_granule_filename(path)
            except ValueError:
                continue
            if allowed_dates is None or info["date"] in allowed_dates:
                result[info["pair_key"]] = (str(path), info)
        return files, result
    v7_files, v7 = index(config["v7_root"], config["v7_pattern"])
    v8_files, v8 = index(config["v8_root"], config["v8_pattern"])
    common = sorted(set(v7) & set(v8))
    pairs = []
    for key in common:
        v7_path, info = v7[key]
        v8_path, _ = v8[key]
        pairs.append({**info, "key": key, "v7_file": v7_path, "v8_file": v8_path})
    return {
        "v7_count": len(v7_files), "v8_count": len(v8_files), "matched_count": len(pairs),
        "unpaired_v7": sorted(set(v7) - set(v8)), "unpaired_v8": sorted(set(v8) - set(v7)),
        "pairs": pairs,
    }


def finalized_raqi_index(root=ANCILLARY_ROOTS["raqi"]):
    out = {}
    regex = re.compile(r"Reconstructed_RAQI_01H_(\d{8})\.nc$")
    for path in sorted(Path(root).glob("Reconstructed_RAQI_01H_*.nc")):
        if path.name.endswith(".partial"):
            continue
        match = regex.fullmatch(path.name)
        if match:
            out[dt.datetime.strptime(match.group(1), "%Y%m%d").date()] = str(path)
    return out


def decode_scan_time(group):
    fields = ["Year", "Month", "DayOfMonth", "Hour", "Minute", "Second"]
    values = [group[name][:] for name in fields]
    millis = group["MilliSecond"][:] if "MilliSecond" in group else np.zeros_like(values[0])
    dates = [dt.datetime(*(int(x) for x in row), int(ms) * 1000) for row, ms in zip(zip(*values), millis)]
    return np.array(dates, dtype=object)


def seconds_since_epoch(values):
    return np.array([(value - EPOCH).total_seconds() for value in values], dtype=np.float64)


def datetime64_seconds(value):
    """Convert a scalar NumPy/xarray datetime coordinate to Unix seconds."""
    value_ns = np.asarray(value).astype("datetime64[ns]").reshape(-1)[0]
    return float(value_ns.astype(np.int64)) / 1.0e9


def read_gprof(path, config):
    with h5py.File(path, "r") as h5:
        group = h5[config["group"]]
        lat = group["Latitude"][:].astype(np.float32)
        lon = group["Longitude"][:].astype(np.float32)
        pds = group["surfacePrecipitation"]
        precip = pds[:].astype(np.float32)
        fill = float(np.asarray(pds.attrs.get("_FillValue", -9999.9)).reshape(-1)[0])
        # GPROF archives are not fully uniform: some constellation files store
        # -9999.0 while advertising -9999.9. Physical precipitation is >= 0.
        precip[(precip == fill) | (precip < 0) | ~np.isfinite(precip)] = np.nan
        scan_dt = decode_scan_time(group["ScanTime"])
        units = as_text(pds.attrs.get("units", pds.attrs.get("Units", "mm h-1")))
    return {"lat": lat, "lon": lon, "precip": precip, "scan_datetimes": scan_dt,
            "scan_seconds": seconds_since_epoch(scan_dt), "precip_units": str(units), "fill": fill}


def wrapped_lon_difference(a, b):
    return np.abs(((np.asarray(a) - np.asarray(b) + 180.0) % 360.0) - 180.0)


def geometry_qc(v7, v8, tolerance_deg=5e-3):
    qc = {"classification": "FAIL_SHAPE_OR_TIME_MISMATCH", "passed": False}
    if v7["lat"].shape != v8["lat"].shape or v7["scan_seconds"].shape != v8["scan_seconds"].shape:
        return {**qc, "note": "V7/V8 footprint or scan-time shape differs"}
    lat_diff = np.abs(v7["lat"].astype(float) - v8["lat"].astype(float))
    lon_diff = wrapped_lon_difference(v7["lon"], v8["lon"])
    time_diff = np.abs(v7["scan_seconds"] - v8["scan_seconds"])
    max_lat, max_lon, max_time = map(float, (np.nanmax(lat_diff), np.nanmax(lon_diff), np.nanmax(time_diff)))
    exceed = (lat_diff > tolerance_deg) | (lon_diff > tolerance_deg)
    frac = float(np.count_nonzero(exceed) / exceed.size)
    edge = np.zeros(exceed.shape, bool)
    edge[:, :30] = True; edge[:, -30:] = True
    if max_time > 1.0:
        classification, passed = "FAIL_SHAPE_OR_TIME_MISMATCH", False
    elif not np.any(exceed):
        classification, passed = "PASS_NORMAL", True
    elif np.all(~exceed | edge) and frac <= 0.001 and max(max_lat, max_lon) <= 0.01:
        classification, passed = "PASS_MINOR_EDGE_WARNING", True
    else:
        classification, passed = "FAIL_TRUE_GEOMETRY_MISMATCH", False
    return {"classification": classification, "passed": passed, "max_lat_diff_deg": max_lat,
            "max_wrapped_lon_diff_deg": max_lon, "max_scan_time_diff_seconds": max_time,
            "percent_pixels_exceeding_tolerance": frac * 100.0,
            "note": f"{classification}; max lat/lon={max_lat:.6g}/{max_lon:.6g} deg; max time={max_time:.3g} s"}


def nearest_indices(values, grid):
    values, grid = np.asarray(values), np.asarray(grid)
    order = np.argsort(grid); sorted_grid = grid[order]
    pos = np.searchsorted(sorted_grid, values)
    p0 = np.clip(pos - 1, 0, len(grid) - 1); p1 = np.clip(pos, 0, len(grid) - 1)
    use1 = np.abs(sorted_grid[p1] - values) < np.abs(sorted_grid[p0] - values)
    return order[np.where(use1, p1, p0)]


def nearest_hour(value):
    """IMERG-style nearest-time selection; exact half-hour ties select earlier."""
    base = value.replace(minute=0, second=0, microsecond=0)
    delta = (value - base).total_seconds()
    return base + (dt.timedelta(hours=1) if delta > 1800 else dt.timedelta(0))


def required_valid_hours(scan_datetimes):
    return sorted({nearest_hour(value) for value in scan_datetimes})


def time_difference_minutes(selected_seconds, scan_seconds):
    """Signed ancillary-minus-scan offset in minutes."""
    return ((np.asarray(selected_seconds, dtype=np.float64) -
             np.asarray(scan_seconds, dtype=np.float64)) / 60.0).astype(np.float32)


def quality_flags(mrms, raqi, autosnow, latitude, longitude, mrms_bounds):
    """Return explicit uint8 domain/validity/reference flags."""
    lat = np.asarray(latitude); lon = np.asarray(longitude)
    south, north, west, east = mrms_bounds
    lon360 = np.mod(lon, 360.0)
    west360, east360 = west % 360.0, east % 360.0
    if west360 <= east360:
        in_lon = (lon360 >= west360) & (lon360 <= east360)
    else:
        in_lon = (lon360 >= west360) | (lon360 <= east360)
    domain = np.isfinite(lat) & np.isfinite(lon) & (lat >= south) & (lat <= north) & in_lon
    mrms_valid = np.isfinite(mrms)
    raqi_valid = np.isfinite(raqi)
    autosnow_valid = np.asarray(autosnow) != AUTOSNOW_FILL
    reference_valid = mrms_valid & raqi_valid
    return tuple(value.astype(np.uint8) for value in
                 (domain, mrms_valid, raqi_valid, autosnow_valid, reference_valid))


def era5_file(root, year):
    path = Path(root) / f"ERA5_tp_hourly_{year}.nc"
    return str(path) if path.exists() else None


def read_era5_hour(root, valid_hour):
    path = era5_file(root, valid_hour.year)
    if path is None:
        raise FileNotFoundError(f"missing ERA5 year {valid_hour.year}")
    with xr.open_dataset(path, decode_timedelta=False) as ds:
        time_name = "valid_time" if "valid_time" in ds.coords else "time"
        field = ds["tp"].sel({time_name: np.datetime64(valid_hour)}, method="nearest").load()
        actual_time_seconds = datetime64_seconds(field[time_name].values)
        units = field.attrs.get("units", "m")
        data = field.values.astype(np.float32) * (1000.0 if units == "m" else 1.0)
        data[data < 0] = 0
        return data, ds["latitude"].values, ds["longitude"].values, path, actual_time_seconds


MRMS_RE = re.compile(r"MultiSensor_QPE_01H_Pass2_00\.00_(\d{8})-(\d{6})\.grib2\.gz$")


def mrms_file(root, valid_hour):
    stamp = valid_hour.strftime("%Y%m%d-%H%M%S")
    path = Path(root) / valid_hour.strftime("%Y%m%d") / f"MultiSensor_QPE_01H_Pass2_00.00_{stamp}.grib2.gz"
    return str(path) if path.exists() else None


def read_mrms(path):
    """Decode the operational north-to-south, west-to-east MRMS GRIB2 grid."""
    with gzip.open(path, "rb") as stream:
        gid = eccodes.codes_new_from_message(stream.read())
    try:
        ni, nj = int(eccodes.codes_get(gid, "Ni")), int(eccodes.codes_get(gid, "Nj"))
        scan_keys = {key: int(eccodes.codes_get(gid, key)) for key in
                     ("iScansNegatively", "jScansPositively", "jPointsAreConsecutive",
                      "alternativeRowScanning")}
        expected_scanning = {
            "iScansNegatively": 0, "jScansPositively": 0,
            "jPointsAreConsecutive": 0, "alternativeRowScanning": 0,
        }
        if scan_keys != expected_scanning:
            raise ValueError(
                f"Unsupported MRMS GRIB scanning orientation {scan_keys}; "
                f"expected {expected_scanning}"
            )
        data = eccodes.codes_get_values(gid).reshape(nj, ni).astype(np.float32)
        missing = float(eccodes.codes_get(gid, "missingValue"))
        data[(data == missing) | (data < 0) | ~np.isfinite(data)] = np.nan
        lat0 = float(eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees"))
        lat1 = float(eccodes.codes_get(gid, "latitudeOfLastGridPointInDegrees"))
        lon0 = float(eccodes.codes_get(gid, "longitudeOfFirstGridPointInDegrees"))
        lon1 = float(eccodes.codes_get(gid, "longitudeOfLastGridPointInDegrees"))
        if not (lat0 > lat1 and lon0 < lon1):
            raise ValueError(
                f"MRMS corner ordering is inconsistent with scanning keys: "
                f"first=({lat0}, {lon0}), last=({lat1}, {lon1})"
            )
        lats = np.linspace(lat0, lat1, nj); lons = np.linspace(lon0, lon1, ni)
        meta = {key: as_text(eccodes.codes_get(gid, key)) for key in
                ("dataDate", "dataTime", "validityDate", "validityTime", "stepType")}
        meta.update(scan_keys)
        valid_time = dt.datetime.strptime(
            f"{int(meta['validityDate']):08d}{int(meta['validityTime']):04d}",
            "%Y%m%d%H%M",
        )
        meta["valid_time_seconds"] = (valid_time - EPOCH).total_seconds()
        meta.update({"Ni": ni, "Nj": nj, "first_latitude": lat0,
                     "last_latitude": lat1, "first_longitude": lon0,
                     "last_longitude": lon1})
    finally:
        eccodes.codes_release(gid)
    return data, lats, lons, meta


def inspect_mrms_grib_orientation(path):
    """Independently decode ecCodes coordinate arrays and verify known MRMS corners."""
    with gzip.open(path, "rb") as stream:
        gid = eccodes.codes_new_from_message(stream.read())
    try:
        ni, nj = int(eccodes.codes_get(gid, "Ni")), int(eccodes.codes_get(gid, "Nj"))
        keys = {key: int(eccodes.codes_get(gid, key)) for key in
                ("iScansNegatively", "jScansPositively", "jPointsAreConsecutive",
                 "alternativeRowScanning")}
        latitudes = eccodes.codes_get_array(gid, "latitudes").reshape(nj, ni)
        lat_corners = [float(latitudes[0, 0]), float(latitudes[0, -1]),
                       float(latitudes[-1, 0]), float(latitudes[-1, -1])]
        lat_row_step = float(latitudes[1, 0] - latitudes[0, 0])
        del latitudes
        longitudes = eccodes.codes_get_array(gid, "longitudes").reshape(nj, ni)
        lon_corners = [float(longitudes[0, 0]), float(longitudes[0, -1]),
                       float(longitudes[-1, 0]), float(longitudes[-1, -1])]
        lon_column_step = float(longitudes[0, 1] - longitudes[0, 0])
        del longitudes
    finally:
        eccodes.codes_release(gid)
    known_lat = [54.995, 54.995, 20.005, 20.005]
    known_lon = [230.005, 299.995, 230.005, 299.995]
    expected_keys = {"iScansNegatively": 0, "jScansPositively": 0,
                     "jPointsAreConsecutive": 0, "alternativeRowScanning": 0}
    passed = (keys == expected_keys and np.allclose(lat_corners, known_lat, atol=2e-5) and
              np.allclose(lon_corners, known_lon, atol=2e-5) and
              np.isclose(lat_row_step, -0.01, atol=2e-5) and
              np.isclose(lon_column_step, 0.01, atol=2e-5))
    return {"passed": bool(passed), "Ni": ni, "Nj": nj, "scanning_keys": keys,
            "decoded_latitude_corners": lat_corners,
            "decoded_longitude_corners": lon_corners,
            "decoded_row_latitude_step": lat_row_step,
            "decoded_column_longitude_step": lon_column_step,
            "method": "ecCodes latitude/longitude arrays reshaped independently to (Nj, Ni) and compared with known MRMS corners"}


def raqi_file(root, valid_hour):
    path = Path(root) / f"Reconstructed_RAQI_01H_{valid_hour:%Y%m%d}.nc"
    return str(path) if path.exists() and not path.name.endswith(".partial") else None


def read_raqi(root, valid_hour):
    path = raqi_file(root, valid_hour)
    if path is None:
        raise FileNotFoundError(f"missing finalized RAQI day {valid_hour:%Y-%m-%d}")
    with xr.open_dataset(path, decode_timedelta=False) as ds:
        available = ds["time"].values.astype("datetime64[s]")
        target = np.datetime64(valid_hour, "s")
        matches = np.where(available == target)[0]
        if not matches.size:
            raise KeyError(f"RAQI hour {valid_hour.isoformat()} absent from {path}")
        field = ds["RAQI"].isel(time=int(matches[0])).load().values.astype(np.float32)
        field[(field < 0) | (field > 1)] = np.nan
        actual_time_seconds = datetime64_seconds(available[int(matches[0])])
        return field, ds["latitude"].values, ds["longitude"].values, path, actual_time_seconds


def _regular_transform(lat, lon):
    lat, lon = np.asarray(lat), np.asarray(lon)
    dx = float(np.median(np.diff(lon))); dy = abs(float(np.median(np.diff(lat))))
    return from_origin(float(lon[0]) - dx / 2, float(lat[0]) + dy / 2, dx, dy)


def _prepare_regular_grid(data, lat, lon):
    data, lat, lon = np.asarray(data), np.asarray(lat), np.asarray(lon, dtype=float)
    if lat[0] < lat[-1]:
        lat, data = lat[::-1], data[::-1, :]
    lon = np.mod(lon, 360.0)
    order = np.argsort(lon)
    return data[:, order], lat, lon[order]


def resample_to_era5(data, src_lat, src_lon, era_lat, era_lon, categorical=False):
    """In-memory geospatial average/mode resampling to actual ERA5 coordinates."""
    data, src_lat, src_lon = _prepare_regular_grid(data, src_lat, src_lon)
    dst = np.full((len(era_lat), len(era_lon)), 255 if categorical else np.nan,
                  dtype=np.uint8 if categorical else np.float32)
    src = data.astype(np.uint8 if categorical else np.float32, copy=False)
    reproject(source=src, destination=dst, src_transform=_regular_transform(src_lat, src_lon),
              src_crs="EPSG:4326", src_nodata=255 if categorical else np.nan,
              dst_transform=_regular_transform(era_lat, era_lon), dst_crs="EPSG:4326",
              dst_nodata=255 if categorical else np.nan,
              resampling=Resampling.mode if categorical else Resampling.average,
              num_threads=1)
    return dst


def sample_era_grid(field, lat, lon, footprint_lat, footprint_lon):
    y = nearest_indices(np.asarray(footprint_lat).ravel(), lat)
    x = nearest_indices(np.mod(np.asarray(footprint_lon).ravel(), 360.0), lon)
    return field[y, x].reshape(np.asarray(footprint_lat).shape)


def read_merra2_t2m(root, scan_hour):
    date = scan_hour.date()
    paths = sorted(Path(root).glob(f"MERRA2_*.tavg1_2d_slv_Nx.{date:%Y%m%d}.SUB.nc"))
    if not paths:
        raise FileNotFoundError(f"missing MERRA-2 day {date}")
    with xr.open_dataset(paths[0]) as ds:
        selected = ds["T2M"].sel(time=np.datetime64(scan_hour), method="nearest").load()
        actual_time_seconds = datetime64_seconds(selected["time"].values)
        field = selected.values.astype(np.float32)
        field[(field < 150) | (field > 350)] = np.nan
        return (field, ds["lat"].values, ds["lon"].values, str(paths[0]),
                ds["T2M"].attrs.get("units", "K"), actual_time_seconds)


def autosnow_file(root, date):
    doy = date.timetuple().tm_yday
    path = Path(root) / str(date.year) / f"gmasi_snowice_reproc_v003_{date.year}{doy:03d}_native_0.04deg.nc"
    return str(path) if path.exists() else None


def read_autosnow(root, date):
    path = autosnow_file(root, date)
    if path is None:
        raise FileNotFoundError(f"missing AutoSnow day {date}")
    with xr.open_dataset(path, mask_and_scale=False) as ds:
        field = ds["autosnow_class"].load().values.astype(np.uint8)
        field[~np.isin(field, [0, 1, 2, 3])] = AUTOSNOW_FILL
        return field, ds["lat"].values, ds["lon"].values, path


def atomic_write_netcdf(ds, final_path):
    final_path = Path(final_path); partial = Path(str(final_path) + ".partial")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if partial.exists():
        partial.unlink()
    encoding = {}
    for name, var in ds.data_vars.items():
        if var.ndim:
            encoding[name] = {"zlib": True, "complevel": 4, "shuffle": True}
            if np.issubdtype(var.dtype, np.floating): encoding[name]["_FillValue"] = np.nan
            elif name == "AutoSnow": encoding[name]["_FillValue"] = int(AUTOSNOW_FILL)
    ds.to_netcdf(partial, engine="netcdf4", encoding=encoding)
    os.replace(partial, final_path)


def json_ready(value):
    if isinstance(value, (dt.date, dt.datetime)): return value.isoformat()
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=json_ready) + "\n")

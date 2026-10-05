#!/usr/bin/env python3
"""Footprint-aligned helpers for native-footprint GPROF V7/V8 collocation.

This module deliberately leaves the legacy helper module unchanged.  Stable
GPROF, MRMS, RAQI, ERA5, AutoSnow, geometry, and NetCDF utilities are reused;
new temporal semantics and Stage IV/MERRA-2 wet-bulb support live here.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import xarray as xr

import my_functions_gprof_v7_v8_mrms_raqi_era5_merra2_autosnow as legacy

# Re-export stable legacy interfaces used by the new driver.
EPOCH = legacy.EPOCH
TIME_UNITS = legacy.TIME_UNITS
AUTOSNOW_FILL = legacy.AUTOSNOW_FILL
SENSOR_CONFIGS = legacy.SENSOR_CONFIGS
sensor_config = legacy.sensor_config
discover_pairs = legacy.discover_pairs
read_gprof = legacy.read_gprof
geometry_qc = legacy.geometry_qc
read_era5_hour = legacy.read_era5_hour
read_mrms = legacy.read_mrms
mrms_file = legacy.mrms_file
read_raqi = legacy.read_raqi
read_autosnow = legacy.read_autosnow
resample_to_era5 = legacy.resample_to_era5
sample_era_grid = legacy.sample_era_grid
quality_flags = legacy.quality_flags
time_difference_minutes = legacy.time_difference_minutes
seconds_since_epoch = legacy.seconds_since_epoch
datetime64_seconds = legacy.datetime64_seconds
nearest_indices = legacy.nearest_indices
nearest_hour = legacy.nearest_hour
atomic_write_netcdf = legacy.atomic_write_netcdf
json_ready = legacy.json_ready
write_json = legacy.write_json

EXPERIMENT_ID = "20260824"
DATASET_TAG = "GPROF_V7_V8_MRMS_RAQI_ERA5_MERRA2_AutoSnow_StageIV"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / f"analysis_outputs_StageIV_{EXPERIMENT_ID}"
DEFAULT_LAND_MASK = Path(
    "/ra1/pubdat/AVHRR_CloudSat_proj/IMERG/ancillary_imerg_data/"
    "GPM_IMERG_LandSeaMask.2.nc4"
)

ANCILLARY_ROOTS = {
    **legacy.ANCILLARY_ROOTS,
    "stageiv": "/scratch/kkumah/Stage_iv/hourly",
}


def _matched_scan_indices(v7_seconds, v8_seconds, tolerance_seconds=1.0,
                          v7_lat=None, v7_lon=None, v8_lat=None, v8_lon=None):
    """Return a monotonic, one-to-one nearest-time scan matching.

    The scan order is preserved and a V8 scan can be used at most once.  This
    handles unequal scan counts without silently shifting the rest of an orbit.
    """
    a = np.asarray(v7_seconds, dtype=float)
    b = np.asarray(v8_seconds, dtype=float)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("scan times must be one-dimensional")
    if tolerance_seconds < 0:
        raise ValueError("scan-time tolerance must be non-negative")
    if np.any(np.diff(a) < 0) or np.any(np.diff(b) < 0):
        raise ValueError("scan times must be monotonic non-decreasing")
    use_geometry = all(x is not None for x in (v7_lat, v7_lon, v8_lat, v8_lon))
    if use_geometry:
        n_pixel = min(np.asarray(v7_lat).shape[1], np.asarray(v8_lat).shape[1])
        sample_pixels = np.unique(np.linspace(0, n_pixel - 1, min(7, n_pixel)).astype(int))
    proposals = []
    for i, value in enumerate(a):
        lower = np.searchsorted(b, value - tolerance_seconds, side="left")
        upper = np.searchsorted(b, value + tolerance_seconds, side="right")
        candidates = np.arange(lower, upper, dtype=np.int64)
        if candidates.size == 0:
            continue
        scored = []
        for j in candidates:
            time_score = abs(b[j] - value)
            if use_geometry:
                distances = haversine_distance_km(
                    np.asarray(v7_lat)[i, sample_pixels], np.asarray(v7_lon)[i, sample_pixels],
                    np.asarray(v8_lat)[j, sample_pixels], np.asarray(v8_lon)[j, sample_pixels],
                )
                geometry_score = float(np.nanmedian(distances)) if np.isfinite(distances).any() else np.inf
            else:
                geometry_score = 0.0
            scored.append((geometry_score, time_score, int(j)))
        geometry_score, time_score, j = min(scored)
        proposals.append((i, j, geometry_score, time_score))

    # Resolve competition for the same V8 scan by geolocation first, then time.
    best_for_v8 = {}
    for proposal in proposals:
        key = proposal[1]
        rank = (proposal[2], proposal[3], proposal[0])
        if key not in best_for_v8 or rank < best_for_v8[key][0]:
            best_for_v8[key] = (rank, proposal)
    selected = sorted((item[1] for item in best_for_v8.values()), key=lambda x: x[0])
    monotonic = []
    last_j = -1
    for proposal in selected:
        if proposal[1] > last_j:
            monotonic.append(proposal)
            last_j = proposal[1]
    return (
        np.asarray([x[0] for x in monotonic], dtype=np.int64),
        np.asarray([x[1] for x in monotonic], dtype=np.int64),
    )


def haversine_distance_km(lat1, lon1, lat2, lon2):
    """Great-circle separation between paired footprint centers."""
    lat1 = np.asarray(lat1, dtype=float)
    lon1 = np.asarray(lon1, dtype=float)
    lat2 = np.asarray(lat2, dtype=float)
    lon2 = np.asarray(lon2, dtype=float)
    finite = np.isfinite(lat1) & np.isfinite(lon1) & np.isfinite(lat2) & np.isfinite(lon2)
    out = np.full(np.broadcast_shapes(lat1.shape, lon1.shape, lat2.shape, lon2.shape), np.nan)
    p1, p2 = np.deg2rad(lat1[finite]), np.deg2rad(lat2[finite])
    dp = p2 - p1
    dl = np.deg2rad(((lon2[finite] - lon1[finite] + 180.0) % 360.0) - 180.0)
    h = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    out[finite] = 6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))
    return out.astype(np.float32)


def align_gprof_footprints(v7, v8, *, scan_time_tolerance_seconds=1.0,
                           maximum_separation_km=1.0):
    """Align common scans/pixels and construct a footprint-level pair mask."""
    i7, i8 = _matched_scan_indices(
        v7["scan_seconds"], v8["scan_seconds"], scan_time_tolerance_seconds,
        v7["lat"], v7["lon"], v8["lat"], v8["lon"],
    )
    if i7.size == 0:
        raise ValueError("V7 and V8 have no scans within the scan-time tolerance")
    pixels = min(v7["lat"].shape[1], v8["lat"].shape[1])
    if pixels < 1:
        raise ValueError("V7 and V8 have no common pixel positions")
    lat7, lon7 = v7["lat"][i7, :pixels], v7["lon"][i7, :pixels]
    lat8, lon8 = v8["lat"][i8, :pixels], v8["lon"][i8, :pixels]
    separation = haversine_distance_km(lat7, lon7, lat8, lon8)
    pair_valid = np.isfinite(separation) & (separation <= maximum_separation_km)
    finite_sep = separation[np.isfinite(separation)]
    quantiles = (
        np.nanpercentile(finite_sep, [50, 95, 99, 100]).tolist()
        if finite_sep.size else [np.nan] * 4
    )
    qc = {
        "source_v7_scans": int(len(v7["scan_seconds"])),
        "source_v8_scans": int(len(v8["scan_seconds"])),
        "aligned_scans": int(i7.size),
        "unmatched_v7_scans": int(len(v7["scan_seconds"]) - i7.size),
        "unmatched_v8_scans": int(len(v8["scan_seconds"]) - i8.size),
        "source_v7_pixels": int(v7["lat"].shape[1]),
        "source_v8_pixels": int(v8["lat"].shape[1]),
        "aligned_pixels": int(pixels),
        "aligned_footprints": int(pair_valid.size),
        "valid_footprint_pairs": int(pair_valid.sum()),
        "invalid_footprint_pairs": int((~pair_valid).sum()),
        "valid_pair_fraction": float(pair_valid.mean()),
        "scan_time_tolerance_seconds": float(scan_time_tolerance_seconds),
        "maximum_separation_km": float(maximum_separation_km),
        "separation_km_p50": float(quantiles[0]),
        "separation_km_p95": float(quantiles[1]),
        "separation_km_p99": float(quantiles[2]),
        "separation_km_max": float(quantiles[3]),
        "pairs_within_1km": int(np.count_nonzero(separation <= 1.0)),
        "pairs_within_2p5km": int(np.count_nonzero(separation <= 2.5)),
        "pairs_within_5km": int(np.count_nonzero(separation <= 5.0)),
    }
    return {
        "v7_scan_indices": i7,
        "v8_scan_indices": i8,
        "latitude_v7": lat7,
        "longitude_v7": lon7,
        "latitude_v8": lat8,
        "longitude_v8": lon8,
        "precipitation_v7": v7["precip"][i7, :pixels],
        "precipitation_v8": v8["precip"][i8, :pixels],
        "scan_datetimes": v7["scan_datetimes"][i7],
        "scan_seconds": v7["scan_seconds"][i7],
        "v8_scan_seconds": v8["scan_seconds"][i8],
        "scan_time_difference_seconds": (v8["scan_seconds"][i8] - v7["scan_seconds"][i7]).astype(np.float32),
        "separation_km": separation,
        "pair_valid_flag": pair_valid.astype(np.uint8),
        "qc": qc,
    }


def accumulation_end_hour(value: dt.datetime) -> dt.datetime:
    """Return the hourly interval end satisfying ``start < value <= end``."""
    floor = value.replace(minute=0, second=0, microsecond=0)
    return floor if value == floor else floor + dt.timedelta(hours=1)


def accumulation_bounds(end: dt.datetime) -> tuple[dt.datetime, dt.datetime]:
    """Return the one-hour ``(start, end]`` accumulation interval."""
    return end - dt.timedelta(hours=1), end


def interval_contains(
    value: dt.datetime, lower: dt.datetime, upper: dt.datetime
) -> bool:
    """True when ``value`` is uniquely contained by ``(lower, upper]``."""
    return lower < value <= upper


def required_accumulation_end_hours(scan_datetimes) -> list[dt.datetime]:
    return sorted({accumulation_end_hour(value) for value in scan_datetimes})


def nearest_merra2_time(value: dt.datetime) -> dt.datetime:
    """Nearest MERRA-2 tavg1 field center (:30); exact ties choose earlier."""
    base = dt.datetime.combine(value.date(), dt.time(0, 30))
    elapsed = (value - base).total_seconds()
    lower = base + dt.timedelta(hours=np.floor(elapsed / 3600.0))
    upper = lower + dt.timedelta(hours=1)
    return upper if (upper - value) < (value - lower) else lower


def mrms_file_for_scan(root, scan_time: dt.datetime):
    """Locate MRMS 01H using interval containment instead of nearest hour."""
    return legacy.mrms_file(root, accumulation_end_hour(scan_time))


def read_merra2_temperatures(root, selected_time: dt.datetime):
    """Read T2M and T2MWET from one MERRA-2 time-averaged hourly field."""
    date = selected_time.date()
    paths = sorted(Path(root).glob(f"MERRA2_*.tavg1_2d_slv_Nx.{date:%Y%m%d}.SUB.nc"))
    if not paths:
        raise FileNotFoundError(f"missing MERRA-2 day {date}")
    with xr.open_dataset(paths[0]) as ds:
        missing = [name for name in ("T2M", "T2MWET") if name not in ds]
        if missing:
            raise KeyError(f"{paths[0]} lacks required MERRA-2 variable(s): {missing}")
        actual = ds["time"].sel(time=np.datetime64(selected_time), method="nearest")
        index = int(np.argmin(np.abs(ds["time"].values - actual.values)))
        fields = []
        units = []
        for name in ("T2M", "T2MWET"):
            item = ds[name].isel(time=index).load()
            values = item.values.astype(np.float32)
            values[(values < 150.0) | (values > 350.0) | ~np.isfinite(values)] = np.nan
            fields.append(values)
            units.append(str(item.attrs.get("units", "K")))
        return (
            fields[0], fields[1], ds["lat"].values, ds["lon"].values,
            str(paths[0]), units[0], units[1], datetime64_seconds(actual.values),
        )


def sample_regular_nearest(field, lat, lon, footprint_lat, footprint_lon):
    """Nearest-neighbor sample a regular lat/lon field at footprint centers."""
    y = nearest_indices(np.asarray(footprint_lat).ravel(), np.asarray(lat))
    grid_lon = np.mod(np.asarray(lon, dtype=float), 360.0)
    x = nearest_indices(np.mod(np.asarray(footprint_lon).ravel(), 360.0), grid_lon)
    return np.asarray(field)[y, x].reshape(np.asarray(footprint_lat).shape)


def sample_imerg_land_mask(
    path, footprint_lat, footprint_lon, variable="landseamask", threshold=25.0
):
    """Return uint8 land mask at footprint centers (1=land, 0=water)."""
    with xr.open_dataset(path) as ds:
        if variable not in ds:
            raise KeyError(f"{variable!r} is absent from {path}")
        source = ds[variable].squeeze(drop=True).transpose("lat", "lon").load()
        values = sample_regular_nearest(
            source.values, source["lat"].values, source["lon"].values,
            footprint_lat, footprint_lon,
        )
    return (values < threshold).astype(np.uint8)


def stageiv_file(root, year: int):
    path = Path(root) / f"{year}_stage4_hourly.nc"
    return str(path) if path.exists() else None


def readable_stageiv_years(root) -> list[int]:
    """List annual files whose NetCDF metadata and required variables are readable."""
    years = []
    for path in sorted(Path(root).glob("????_stage4_hourly.nc")):
        try:
            year = int(path.name[:4])
            with xr.open_dataset(path, decode_timedelta=False) as ds:
                required = {"time", "time_bnds", "lat", "lon", "p01m", "p01m_status"}
                if required.issubset(ds.variables) and ds.sizes.get("time", 0) > 0:
                    years.append(year)
        except (OSError, ValueError):
            continue
    return years


class StageIVAnnualReader:
    """Lazy annual Stage IV reader with cached coordinates and 0.25-degree mapping."""

    def __init__(self, path, target_lat, target_lon):
        self.path = str(path)
        self.ds = xr.open_dataset(self.path, decode_timedelta=False, cache=False)
        required = {"time", "time_bnds", "lat", "lon", "p01m", "p01m_status"}
        missing = sorted(required - set(self.ds.variables))
        if missing:
            self.close()
            raise KeyError(f"{self.path} lacks required Stage IV variables: {missing}")
        self.times = self.ds["time"].values.astype("datetime64[ns]")
        self.bounds = self.ds["time_bnds"].values.astype("datetime64[ns]")
        self.target_lat = np.asarray(target_lat)
        self.target_lon = np.asarray(target_lon)
        src_lat = self.ds["lat"].values.ravel()
        src_lon = np.mod(self.ds["lon"].values.ravel(), 360.0)
        finite = np.isfinite(src_lat) & np.isfinite(src_lon)
        y = nearest_indices(src_lat[finite], self.target_lat)
        x = nearest_indices(src_lon[finite], np.mod(self.target_lon, 360.0))
        self.source_flat_indices = np.flatnonzero(finite)
        self.destination_flat_indices = y * len(self.target_lon) + x
        self.destination_size = len(self.target_lat) * len(self.target_lon)

    def close(self):
        if getattr(self, "ds", None) is not None:
            self.ds.close()
            self.ds = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _time_index(self, end_time: dt.datetime) -> int:
        target = np.datetime64(end_time, "ns")
        matches = np.flatnonzero(self.times == target)
        if matches.size != 1:
            raise KeyError(f"Stage IV interval end {end_time.isoformat()} absent from {self.path}")
        return int(matches[0])

    def read_hour(self, end_time: dt.datetime):
        """Read one valid accumulation and average it onto the target grid."""
        index = self._time_index(end_time)
        lower64, upper64 = self.bounds[index]
        expected_lower, expected_upper = accumulation_bounds(end_time)
        if lower64 != np.datetime64(expected_lower, "ns") or upper64 != np.datetime64(expected_upper, "ns"):
            raise ValueError(
                f"Stage IV bounds {lower64}..{upper64} disagree with expected "
                f"{expected_lower.isoformat()}..{expected_upper.isoformat()}"
            )
        status = float(self.ds["p01m_status"].isel(time=index).values)
        # The IEM 2021 file uses 3 (copied + QC) although metadata lists 1 and 2.
        # Non-positive/non-finite values alone are treated as unavailable.
        if not np.isfinite(status) or status <= 0:
            raise ValueError(f"Stage IV unavailable status {status:g} at {end_time.isoformat()}")
        source = self.ds["p01m"].isel(time=index).load().values.astype(np.float32).ravel()
        source = source[self.source_flat_indices]
        valid = np.isfinite(source) & (source >= 0.0)
        sums = np.bincount(
            self.destination_flat_indices[valid], weights=source[valid],
            minlength=self.destination_size,
        )
        counts = np.bincount(
            self.destination_flat_indices[valid], minlength=self.destination_size,
        )
        output = np.full(self.destination_size, np.nan, dtype=np.float32)
        populated = counts > 0
        output[populated] = (sums[populated] / counts[populated]).astype(np.float32)
        return (
            output.reshape(len(self.target_lat), len(self.target_lon)),
            datetime64_seconds(upper64), datetime64_seconds(lower64), status,
            counts.reshape(len(self.target_lat), len(self.target_lon)).astype(np.uint32),
        )


def phase_regime(t2m, t2mwet):
    """Transparent provisional phase regime: 0 ambiguous, 1 snow, 2 rain."""
    t2m = np.asarray(t2m)
    wet = np.asarray(t2mwet)
    out = np.zeros(t2m.shape, dtype=np.uint8)
    out[(t2m < 275.15) & (wet <= 273.15)] = 1
    out[wet >= 275.15] = 2
    return out

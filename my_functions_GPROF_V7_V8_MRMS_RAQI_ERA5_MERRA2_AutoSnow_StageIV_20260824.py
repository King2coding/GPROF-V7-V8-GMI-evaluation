#!/usr/bin/env python3
"""Stage-IV-era helpers for native-footprint GPROF V7/V8 collocation.

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

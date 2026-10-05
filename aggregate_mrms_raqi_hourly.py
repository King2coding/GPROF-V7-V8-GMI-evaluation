#!/usr/bin/env python3
"""
Reconstruct hourly MRMS Radar Accumulation Quality Index (RAQI) from
instantaneous RadarQualityIndex GRIB2.GZ files and save one compressed
NetCDF file per day.

Key design choices
------------------
- Reads .grib2.gz directly into memory.
- Never writes uncompressed GRIB2 files to disk.
- Uses instantaneous RQI filenames to construct backward-looking one-hour accumulation windows.
- Masks RQI values outside [0, 1].
- Stores RAQI as uint8 with scale_factor=0.01.
- Stores valid instantaneous-scan count as uint8.
- Writes one restartable daily NetCDF:
    Reconstructed_RAQI_01H_YYYYMMDD.nc

Current hourly window:
    hour_end - 1 hour < RQI_time <= hour_end
Adjust select_rqi_files() if documentation confirms a different convention.
"""

from __future__ import annotations

import argparse
import gzip
import logging
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from netCDF4 import Dataset, date2num

RQI_RE = re.compile(
    r"RadarQualityIndex_00\.00_(\d{8})-(\d{6})\.grib2(?:\.gz)?$"
)

def parse_timestamp(path: Path, pattern: re.Pattern[str]) -> datetime:
    match = pattern.search(path.name)
    if not match:
        raise ValueError(f"Unrecognized filename: {path.name}")
    return datetime.strptime(
        "".join(match.groups()), "%Y%m%d%H%M%S"
    ).replace(tzinfo=timezone.utc)


def read_grib_from_memory(path: Path) -> tuple[np.ndarray, dict]:
    """Read compressed or uncompressed GRIB2 without writing temp files."""
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as f:
            payload = f.read()
        with MemoryFile(payload) as memfile:
            with memfile.open() as src:
                array = src.read(1)
                metadata = {
                    "transform": src.transform,
                    "crs": src.crs,
                    "width": src.width,
                    "height": src.height,
                }
    else:
        with rasterio.open(path) as src:
            array = src.read(1)
            metadata = {
                "transform": src.transform,
                "crs": src.crs,
                "width": src.width,
                "height": src.height,
            }

    return np.asarray(array, dtype=np.float32), metadata


def discover_rqi_files_for_date(
    root: Path,
    target_date,
) -> list[Path]:
    """
    Discover only the RQI files needed to process one QPE-valid date.

    Files from both the target date and previous date are included because
    the QPE field valid at 00:00 UTC uses RQI observations from the preceding
    hour.
    """
    previous_date = target_date - timedelta(days=1)

    date_strings = {
        target_date.strftime("%Y%m%d"),
        previous_date.strftime("%Y%m%d"),
    }

    files: list[Path] = []

    for date_string in sorted(date_strings):
        pattern = f"RadarQualityIndex_00.00_{date_string}-*.grib2*"
        files.extend(
            path
            for path in root.rglob(pattern)
            if path.is_file()
        )

    return sorted(files)

def build_rqi_index(files: Iterable[Path]) -> dict[datetime, Path]:
    index: dict[datetime, Path] = {}
    for path in files:
        try:
            index[parse_timestamp(path, RQI_RE)] = path
        except ValueError:
            continue
    return index


def hourly_end_time(timestamp: datetime) -> datetime:
    """
    Assign an instantaneous RQI timestamp to the ending time of its
    backward-looking one-hour accumulation window.

    Examples
    --------
    2021-01-01 00:00:00 -> 2021-01-01 00:00:00
    2021-01-01 00:02:00 -> 2021-01-01 01:00:00
    2021-01-01 00:58:00 -> 2021-01-01 01:00:00
    2021-01-01 01:00:00 -> 2021-01-01 01:00:00

    This implements the interval convention:

        end_time - 1 hour < RQI_time <= end_time
    """
    floored = timestamp.replace(minute=0, second=0, microsecond=0)

    if timestamp == floored:
        return floored

    return floored + timedelta(hours=1)


def group_rqi_by_hour(
    rqi_index: dict[datetime, Path],
    target_date,
) -> dict[datetime, list[tuple[datetime, Path]]]:
    """
    Group instantaneous RQI files by the hourly accumulation ending time.

    Only hourly ending times belonging to target_date are retained.
    """
    grouped: dict[datetime, list[tuple[datetime, Path]]] = {}

    for timestamp, path in rqi_index.items():
        end_time = hourly_end_time(timestamp)

        if end_time.date() != target_date:
            continue

        grouped.setdefault(end_time, []).append((timestamp, path))

    for end_time in grouped:
        grouped[end_time].sort(key=lambda item: item[0])

    return dict(sorted(grouped.items()))

def calculate_scan_diagnostics(
    selected: list[tuple[datetime, Path]],
) -> dict[str, float | int | datetime]:
    """
    Calculate temporal-sampling diagnostics for one hourly RQI group.

    The expected scan count is estimated from the median interval between
    consecutive instantaneous RQI timestamps.

    For a nominal 2-minute cadence:

        expected_scan_count = 3600 / 120 = 30
    """
    timestamps = [timestamp for timestamp, _ in selected]

    if not timestamps:
        raise ValueError("Cannot calculate diagnostics for an empty group.")

    scan_start = timestamps[0]
    scan_end = timestamps[-1]
    scan_count = len(timestamps)

    if scan_count >= 2:
        intervals = np.diff(
            np.array(
                [
                    timestamp.timestamp()
                    for timestamp in timestamps
                ],
                dtype=np.float64,
            )
        )

        positive_intervals = intervals[intervals > 0]

        if positive_intervals.size > 0:
            nominal_interval_seconds = float(
                np.median(positive_intervals)
            )

            expected_scan_count = int(
                round(3600.0 / nominal_interval_seconds)
            )
        else:
            nominal_interval_seconds = np.nan
            expected_scan_count = scan_count
    else:
        nominal_interval_seconds = np.nan
        expected_scan_count = scan_count

    if expected_scan_count > 0:
        scan_fraction = min(
            scan_count / expected_scan_count,
            1.0,
        )
    else:
        scan_fraction = np.nan

    return {
        "scan_start": scan_start,
        "scan_end": scan_end,
        "scan_count": scan_count,
        "nominal_interval_seconds": nominal_interval_seconds,
        "expected_scan_count": expected_scan_count,
        "scan_fraction": scan_fraction,
    }

def reconstruct_hour(
    selected: list[tuple[datetime, Path]],
) -> tuple[np.ndarray, np.ndarray, dict]:
    if not selected:
        raise ValueError("No RQI files selected.")

    sum_array = None
    count_array = None
    metadata = None

    for _, path in selected:
        array, current_meta = read_grib_from_memory(path)
        valid = np.isfinite(array) & (array >= 0.0) & (array <= 1.0)

        if sum_array is None:
            sum_array = np.zeros(array.shape, dtype=np.float32)
            count_array = np.zeros(array.shape, dtype=np.uint16)
            metadata = current_meta
        elif array.shape != sum_array.shape:
            raise ValueError(
                f"Grid mismatch: {path.name} has {array.shape}; "
                f"expected {sum_array.shape}"
            )

        sum_array[valid] += array[valid]
        count_array[valid] += 1

    mean_array = np.full(sum_array.shape, np.nan, dtype=np.float32)
    np.divide(
        sum_array,
        count_array,
        out=mean_array,
        where=count_array > 0,
    )

    return mean_array, count_array, metadata


def build_lat_lon(metadata: dict) -> tuple[np.ndarray, np.ndarray]:
    transform = metadata["transform"]
    width = metadata["width"]
    height = metadata["height"]

    lon = transform.c + (np.arange(width) + 0.5) * transform.a
    lat = transform.f + (np.arange(height) + 0.5) * transform.e
    return lat.astype(np.float32), lon.astype(np.float32)


def initialize_daily_netcdf(
    out_path: Path,
    metadata: dict,
) -> Dataset:
    """
    Create an empty daily NetCDF file that can be populated one hour at a time.
    """
    lat, lon = build_lat_lon(metadata)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    nc = Dataset(out_path, mode="w", format="NETCDF4")

    nc.createDimension("time", None)
    nc.createDimension("latitude", len(lat))
    nc.createDimension("longitude", len(lon))

    time_var = nc.createVariable(
        "time",
        "f8",
        ("time",),
    )
    time_var.standard_name = "time"
    time_var.long_name = "hourly RAQI interval ending time"
    time_var.units = "seconds since 1970-01-01 00:00:00 UTC"
    time_var.calendar = "standard"

    lat_var = nc.createVariable(
        "latitude",
        "f4",
        ("latitude",),
        zlib=True,
        complevel=4,
    )
    lat_var.standard_name = "latitude"
    lat_var.long_name = "latitude"
    lat_var.units = "degrees_north"
    lat_var[:] = lat

    lon_var = nc.createVariable(
        "longitude",
        "f4",
        ("longitude",),
        zlib=True,
        complevel=4,
    )
    lon_var.standard_name = "longitude"
    lon_var.long_name = "longitude"
    lon_var.units = "degrees_east"
    lon_var[:] = lon

    raqi_var = nc.createVariable(
        "RAQI",
        "u1",
        ("time", "latitude", "longitude"),
        zlib=True,
        complevel=4,
        shuffle=True,
        chunksizes=(1, 500, 500),
        fill_value=np.uint8(255),
    )
    raqi_var.long_name = (
        "Reconstructed hourly radar accumulation quality index"
    )
    raqi_var.units = "1"
    raqi_var.scale_factor = np.float32(0.01)
    raqi_var.add_offset = np.float32(0.0)
    raqi_var.valid_min = np.uint8(0)
    raqi_var.valid_max = np.uint8(100)
    raqi_var.aggregation = (
        "Arithmetic mean of valid instantaneous RQI fields"
    )

    count_var = nc.createVariable(
        "RQI_valid_count",
        "u1",
        ("time", "latitude", "longitude"),
        zlib=True,
        complevel=4,
        shuffle=True,
        chunksizes=(1, 500, 500),
        fill_value=np.uint8(255),
    )
    count_var.long_name = (
        "Number of valid instantaneous RQI values contributing to RAQI"
    )
    count_var.units = "count"

    scan_count_var = nc.createVariable(
        "RQI_scan_count",
        "u1",
        ("time",),
        zlib=True,
        complevel=4,
        shuffle=True,
        fill_value=np.uint8(255),
    )
    scan_count_var.long_name = (
        "Number of instantaneous RQI files selected for the hour"
    )
    scan_count_var.units = "count"

    expected_count_var = nc.createVariable(
        "RQI_expected_scan_count",
        "u1",
        ("time",),
        zlib=True,
        complevel=4,
        shuffle=True,
        fill_value=np.uint8(255),
    )
    expected_count_var.long_name = (
        "Expected number of instantaneous RQI scans in the one-hour interval"
    )
    expected_count_var.units = "count"
    expected_count_var.comment = (
        "Estimated as 3600 seconds divided by the median interval "
        "between available RQI timestamps."
    )

    scan_fraction_var = nc.createVariable(
        "RQI_scan_fraction",
        "f4",
        ("time",),
        zlib=True,
        complevel=4,
        shuffle=True,
        fill_value=np.float32(np.nan),
    )
    scan_fraction_var.long_name = (
        "Fraction of expected instantaneous RQI scans available"
    )
    scan_fraction_var.units = "1"
    scan_fraction_var.valid_min = np.float32(0.0)
    scan_fraction_var.valid_max = np.float32(1.0)

    interval_var = nc.createVariable(
        "RQI_nominal_interval_seconds",
        "f4",
        ("time",),
        zlib=True,
        complevel=4,
        shuffle=True,
        fill_value=np.float32(np.nan),
    )
    interval_var.long_name = (
        "Median time interval between instantaneous RQI scans"
    )
    interval_var.units = "seconds"

    scan_start_var = nc.createVariable(
        "RQI_scan_start_time",
        "f8",
        ("time",),
    )
    scan_start_var.long_name = (
        "Timestamp of first instantaneous RQI scan used"
    )
    scan_start_var.units = "seconds since 1970-01-01 00:00:00 UTC"
    scan_start_var.calendar = "standard"

    scan_end_var = nc.createVariable(
        "RQI_scan_end_time",
        "f8",
        ("time",),
    )
    scan_end_var.long_name = (
        "Timestamp of last instantaneous RQI scan used"
    )
    scan_end_var.units = "seconds since 1970-01-01 00:00:00 UTC"
    scan_end_var.calendar = "standard"

    nc.title = "Reconstructed MRMS Radar Accumulation Quality Index, 1-hour"
    nc.source = "MRMS instantaneous RadarQualityIndex"
    nc.aggregation_method = (
        "Pixelwise arithmetic mean of valid instantaneous RQI fields"
    )
    nc.time_window = "hour_end - 1 hour < RQI_time <= hour_end"
    nc.hour_timestamp_definition = (
        "Each time coordinate is the ending time of the one-hour "
        "RQI averaging interval."
    )
    nc.crs = str(metadata["crs"])
    nc.history = (
        f"Created {datetime.now(timezone.utc).isoformat()}"
    )

    return nc

def encode_raqi(raqi: np.ndarray) -> np.ndarray:
    """
    Pack RAQI values from 0–1 into uint8 values from 0–100.

    255 is reserved as the missing-value code.
    """
    packed = np.full(raqi.shape, 255, dtype=np.uint8)

    valid = np.isfinite(raqi) & (raqi >= 0.0) & (raqi <= 1.0)

    packed[valid] = np.rint(
        raqi[valid] * 100.0
    ).astype(np.uint8)

    return packed

def append_hour_to_netcdf(
    nc: Dataset,
    time_index: int,
    hour_end: datetime,
    raqi: np.ndarray,
    valid_count: np.ndarray,
    diagnostics: dict[str, float | int | datetime],
) -> None:
    """
    Append one reconstructed hourly field to an open daily NetCDF.
    """
    nc.variables["time"][time_index] = date2num(
        hour_end,
        units=nc.variables["time"].units,
        calendar=nc.variables["time"].calendar,
    )

    nc.variables["RAQI"].set_auto_maskandscale(False)
    nc.variables["RAQI"][time_index, :, :] = encode_raqi(raqi)

    encoded_count = np.clip(
        valid_count,
        0,
        254,
    ).astype(np.uint8)

    nc.variables["RQI_valid_count"][
        time_index, :, :
    ] = encoded_count

    nc.variables["RQI_scan_count"][time_index] = np.uint8(
    min(int(diagnostics["scan_count"]), 254)
    )

    nc.variables["RQI_expected_scan_count"][time_index] = np.uint8(
        min(int(diagnostics["expected_scan_count"]), 254)
    )

    nc.variables["RQI_scan_fraction"][time_index] = np.float32(
        diagnostics["scan_fraction"]
    )

    nc.variables["RQI_nominal_interval_seconds"][time_index] = np.float32(
        diagnostics["nominal_interval_seconds"]
    )

    scan_start_var = nc.variables["RQI_scan_start_time"]
    scan_end_var = nc.variables["RQI_scan_end_time"]

    scan_start_var[time_index] = date2num(
        diagnostics["scan_start"],
        units=scan_start_var.units,
        calendar=scan_start_var.calendar,
    )

    scan_end_var[time_index] = date2num(
        diagnostics["scan_end"],
        units=scan_end_var.units,
        calendar=scan_end_var.calendar,
    )

    nc.sync()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rqi-dir",
        type=Path,
        default=Path("/scratch/kkumah/MRMS/RadarQualityIndex"),
    )
   
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/scratch/kkumah/MRMS/RAQI"),
    )
    parser.add_argument(
        "--date",
        required=True,
        help="QPE valid date, YYYY-MM-DD.",
    )
    parser.add_argument(
        "--minimum-scans",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    final_out_path = (
        args.out_dir
        / f"Reconstructed_RAQI_01H_{target_date:%Y%m%d}.nc"
    )

    out_path = final_out_path.with_suffix(".nc.partial")

    if final_out_path.exists() and not args.overwrite:
        logging.info(
            "Output already exists: %s",
            final_out_path,
        )
        return 0

    if out_path.exists():
        logging.warning(
            "Removing incomplete output: %s",
            out_path,
        )
        out_path.unlink()

    rqi_files = discover_rqi_files_for_date(
    args.rqi_dir,
    target_date,
    )

    rqi_index = build_rqi_index(rqi_files)

    logging.info(
        "Indexed %d RQI files for %s and the preceding date",
        len(rqi_index),
        target_date,
    )

    if not rqi_index:
        logging.error(
            "No RQI files found for %s or the preceding date in %s",
            target_date,
            args.rqi_dir,
        )
        return 2    
    
    hourly_groups = group_rqi_by_hour(
    rqi_index,
    target_date,
    )

    logging.info(
        "Found %d hourly RQI groups for %s",
        len(hourly_groups),
        target_date,
    )

    if not hourly_groups:
        logging.error(
            "No hourly RQI groups could be constructed for %s",
            target_date,
        )
        return 2

    nc = None
    hours_written = 0

    try:
        for hour_end, selected in hourly_groups.items():
            diagnostics = calculate_scan_diagnostics(selected)
            logging.info(
                "%s | scans=%d/%d | fraction=%.3f | cadence=%.1f s "
                "| first=%s | last=%s",
                hour_end.isoformat(),
                diagnostics["scan_count"],
                diagnostics["expected_scan_count"],
                diagnostics["scan_fraction"],
                diagnostics["nominal_interval_seconds"],
                diagnostics["scan_start"].isoformat(),
                diagnostics["scan_end"].isoformat(),
            )

            if diagnostics["scan_count"] < args.minimum_scans:
                logging.warning(
                    "Skipping %s: only %d scans; minimum required is %d",
                    hour_end,
                    diagnostics["scan_count"],
                    args.minimum_scans,
                )
                continue

            raqi, counts, metadata = reconstruct_hour(selected)

            if nc is None:
                nc = initialize_daily_netcdf(
                    out_path,
                    metadata,
                )

            append_hour_to_netcdf(
                nc=nc,
                time_index=hours_written,
                hour_end=hour_end,
                raqi=raqi,
                valid_count=counts,
                diagnostics=diagnostics,
            )

            hours_written += 1

            del raqi
            del counts

    finally:
        if nc is not None:
            nc.close()

    if hours_written == 0:
        logging.error(
            "No valid hourly outputs created for %s",
            target_date,
        )

        if out_path.exists():
            out_path.unlink()

        return 2

   # Atomically promote the completed temporary file to the final output.
    out_path.replace(final_out_path)

    logging.info(
        "Completed %s with %d hourly fields",
        final_out_path,
        hours_written,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

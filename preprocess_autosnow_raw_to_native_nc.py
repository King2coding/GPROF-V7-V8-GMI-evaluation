#!/usr/bin/env python3
"""
Preprocess raw GMASI/AutoSnow snow-ice files to compressed native-resolution NetCDF.

Supports both:
    1. Uncompressed raw binary files:
       gmasi_snowice_reproc_v003_2022221

    2. Unix .Z compressed raw files:
       gmasi_snowice_reproc_v003_2023001.Z

Output:
    One compressed NetCDF file per day at native 0.04-degree resolution.

Native grid:
    rows = 4500
    cols = 9000
    resolution = 0.04 degrees
    shape = 4500 x 9000
    dtype = uint8
    valid class values observed: 0, 1, 2, 3

Recommended script name:
    preprocess_autosnow_raw_to_native_nc.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import xarray as xr


NROWS = 4500
NCOLS = 9000
NPIXELS = NROWS * NCOLS
EXPECTED_NBYTES = NPIXELS
RES_DEG = 0.04

DEFAULT_INPUT_DIR = Path(
    "/ra1/pubdat/AVHRR_CloudSat_proj/"
    "Autosnow_archive_1987_june2023/autosnow_raw_files"
)

DEFAULT_OUTPUT_DIR = Path("/scratch/kkumah/Autosnow_2014_to_2024_nc")


def parse_autosnow_date(path: Path) -> dt.date:
    """
    Parse date from AutoSnow filename.

    Expected patterns:
        gmasi_snowice_reproc_v003_YYYYDDD
        gmasi_snowice_reproc_v003_YYYYDDD.Z

    Example:
        gmasi_snowice_reproc_v003_2022221
        -> 2022-08-09
    """
    match = re.search(r"gmasi_snowice_reproc_v003_(\d{4})(\d{3})", path.name)

    if not match:
        raise ValueError(f"Could not parse year/day-of-year from filename: {path.name}")

    year = int(match.group(1))
    doy = int(match.group(2))

    return dt.date(year, 1, 1) + dt.timedelta(days=doy - 1)


def read_compressed_Z_file(path: Path) -> bytes:
    """
    Read a Unix .Z compressed AutoSnow file.
    """
    if shutil.which("uncompress"):
        cmd = ["uncompress", "-c", str(path)]
    elif shutil.which("zcat"):
        cmd = ["zcat", str(path)]
    else:
        raise RuntimeError(
            "Neither 'uncompress' nor 'zcat' was found. "
            "Install ncompress or use a system where .Z files can be decompressed."
        )

    try:
        return subprocess.check_output(cmd)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to decompress {path}") from exc


def read_autosnow_array(path: Path) -> np.ndarray:
    """
    Read one AutoSnow raw file and return a native-resolution 2D uint8 array.

    Handles:
        - uncompressed raw binary files with no extension
        - compressed .Z files
    """
    if path.name.endswith(".Z"):
        raw = read_compressed_Z_file(path)
    else:
        raw = path.read_bytes()

    if len(raw) != EXPECTED_NBYTES:
        raise ValueError(
            f"Unexpected raw size for {path.name}: {len(raw)} bytes. "
            f"Expected {EXPECTED_NBYTES} bytes for shape ({NROWS}, {NCOLS})."
        )

    arr = np.frombuffer(raw, dtype=np.uint8).reshape(NROWS, NCOLS)

    return arr


def build_native_coords() -> tuple[np.ndarray, np.ndarray]:
    """
    Build 1D native AutoSnow latitude and longitude coordinates.

    The raw AutoSnow grid is treated as:
        0.04-degree global grid
        4500 rows x 9000 columns
        north-to-south, west-to-east

    Coordinates represent pixel centers.
    """
    lat = 90.0 - (RES_DEG / 2.0) - np.arange(NROWS) * RES_DEG
    lon = -180.0 + (RES_DEG / 2.0) + np.arange(NCOLS) * RES_DEG

    return lat.astype(np.float32), lon.astype(np.float32)


def write_daily_nc(
    arr: np.ndarray,
    input_file: Path,
    output_file: Path,
    file_date: dt.date,
    compression_level: int = 5,
) -> None:
    """
    Write one daily AutoSnow array to compressed NetCDF.
    """
    lat, lon = build_native_coords()

    ds = xr.Dataset(
        data_vars={
            "autosnow_class": (
                ("lat", "lon"),
                arr,
                {
                    "long_name": "GMASI AutoSnow snow/ice class",
                    "description": (
                        "Raw GMASI/AutoSnow class values preserved from the source file. "
                        "Observed values in the test raw file are 0, 1, 2, and 3. "
                        "Class interpretation should be confirmed from GMASI/AutoSnow documentation."
                    ),
                    "grid_mapping": "crs",
                },
            ),
            "crs": (
                (),
                np.int32(0),
                {
                    "grid_mapping_name": "latitude_longitude",
                    "epsg_code": "EPSG:4326",
                    "semi_major_axis": 6378137.0,
                    "inverse_flattening": 298.257223563,
                },
            ),
        },
        coords={
            "lat": (
                "lat",
                lat,
                {
                    "standard_name": "latitude",
                    "long_name": "latitude",
                    "units": "degrees_north",
                    "axis": "Y",
                    "comment": "Pixel-center latitude. Grid is ordered north-to-south.",
                },
            ),
            "lon": (
                "lon",
                lon,
                {
                    "standard_name": "longitude",
                    "long_name": "longitude",
                    "units": "degrees_east",
                    "axis": "X",
                    "comment": "Pixel-center longitude. Grid is ordered west-to-east.",
                },
            ),
            "time": (
                (),
                np.datetime64(file_date),
                {
                    "standard_name": "time",
                    "long_name": "GMASI AutoSnow daily map date",
                },
            ),
        },
        attrs={
            "title": "Native-resolution GMASI AutoSnow daily snow/ice class map",
            "source_file": str(input_file),
            "source_filename": input_file.name,
            "processing_note": (
                "Converted directly from raw GMASI/AutoSnow binary file to compressed NetCDF. "
                "No GeoTIFF intermediate was created. No spatial resampling was applied."
            ),
            "native_resolution_degrees": RES_DEG,
            "native_shape": f"{NROWS} rows x {NCOLS} columns",
            "grid_orientation": "north-to-south, west-to-east",
            "coordinate_convention": "lat/lon coordinates represent pixel centers",
            "created_by": "preprocess_autosnow_raw_to_native_nc.py",
        },
    )

    encoding = {
        "autosnow_class": {
            "dtype": "uint8",
            "zlib": True,
            "complevel": compression_level,
            "shuffle": True,
            "chunksizes": (500, 1000),
        },
        "lat": {
            "dtype": "float32",
            "zlib": True,
            "complevel": compression_level,
        },
        "lon": {
            "dtype": "float32",
            "zlib": True,
            "complevel": compression_level,
        },
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_file, engine="netcdf4", format="NETCDF4", encoding=encoding)
    ds.close()


def find_input_files(input_dir: Path, year_start: int, year_end: int) -> list[Path]:
    """
    Find available AutoSnow raw files within the requested year range.

    Accepts both:
        gmasi_snowice_reproc_v003_YYYYDDD
        gmasi_snowice_reproc_v003_YYYYDDD.Z
    """
    candidates = sorted(input_dir.glob("gmasi_snowice_reproc_v003_*"))

    files = []

    for path in candidates:
        if not path.is_file():
            continue

        try:
            file_date = parse_autosnow_date(path)
        except ValueError:
            continue

        if year_start <= file_date.year <= year_end:
            files.append(path)

    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert raw GMASI/AutoSnow files to compressed native-resolution NetCDF. "
            "Supports both uncompressed raw files and .Z compressed files."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing raw AutoSnow files. Default: {DEFAULT_INPUT_DIR}",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for output NetCDF files. Default: {DEFAULT_OUTPUT_DIR}",
    )

    parser.add_argument(
        "--year-start",
        type=int,
        default=2014,
        help="First year to process. Default: 2014",
    )

    parser.add_argument(
        "--year-end",
        type=int,
        default=2025,
        help=(
            "Last year to attempt. Default: 2025. "
            "Only existing files are processed, so this is safe if AutoSnow ends earlier."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing NetCDF files.",
    )

    parser.add_argument(
        "--compression-level",
        type=int,
        default=5,
        choices=range(1, 10),
        help="NetCDF zlib compression level from 1 to 9. Default: 5",
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional maximum number of files to process for testing.",
    )

    args = parser.parse_args()

    input_files = find_input_files(args.input_dir, args.year_start, args.year_end)

    if args.max_files is not None:
        input_files = input_files[: args.max_files]

    print("=" * 80)
    print("GMASI/AutoSnow raw-to-NetCDF preprocessing")
    print(f"Input directory : {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Year range      : {args.year_start}–{args.year_end}")
    print(f"Files found     : {len(input_files)}")
    print("=" * 80)

    if not input_files:
        print("No matching input files found. Nothing to process.")
        return

    processed = 0
    skipped = 0
    failed = 0

    for i, input_file in enumerate(input_files, start=1):
        try:
            file_date = parse_autosnow_date(input_file)
            yyyyddd = re.search(r"(\d{7})", input_file.name).group(1)

            output_name = f"gmasi_snowice_reproc_v003_{yyyyddd}_native_0.04deg.nc"
            output_file = args.output_dir / str(file_date.year) / output_name

            if output_file.exists() and output_file.stat().st_size > 0 and not args.overwrite:
                skipped += 1
                if i % 50 == 0 or i == 1:
                    print(f"[{i}/{len(input_files)}] Skipping existing: {output_file.name}")
                continue

            print(f"[{i}/{len(input_files)}] Processing {input_file.name}")

            arr = read_autosnow_array(input_file)
            unique_values = np.unique(arr)

            print(f"    Date: {file_date}; unique values: {unique_values}")

            write_daily_nc(
                arr=arr,
                input_file=input_file,
                output_file=output_file,
                file_date=file_date,
                compression_level=args.compression_level,
            )

            processed += 1

            if processed % 25 == 0:
                print(f"    Progress: {processed} processed, {skipped} skipped, {failed} failed")

        except Exception as exc:
            failed += 1
            print(f"ERROR processing {input_file.name}: {exc}")

    print("=" * 80)
    print("Finished AutoSnow preprocessing.")
    print(f"Processed: {processed}")
    print(f"Skipped  : {skipped}")
    print(f"Failed   : {failed}")
    print(f"Output   : {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
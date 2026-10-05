#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compute_imerg_era5_reference_diagnostics.py

Compute climatological gridded-reference diagnostics from IMERG and ERA5
for comparison with GPROF GMI V7/V8 orbital diagnostics.

Important
---------
This script does NOT timestamp-collocate IMERG/ERA5 with GPROF GMI.
IMERG and ERA5 are treated as independent gridded climatological/product
reference estimates.

Main outputs
------------
1. reference_zonal_stats.csv
2. reference_pdf_stats.csv
3. reference_summary_stats.csv

These outputs are designed to be loaded later by the GPROF V7/V8 plotting code.

Key design
----------
- IMERG is read one half-hourly file at a time.
- IMERG is subset to approximate GMI coverage and remapped to 0.25°.
- Each worker returns compact diagnostic accumulators, not full arrays.
- ERA5 is read in time chunks.
- Surface classes are all / land / ocean.
- PDF and summary diagnostics are separated by ALL_HEMI / NH / SH.
"""

# =============================================================================
# Imports
# =============================================================================

import os
import glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm


# =============================================================================
# User settings
# =============================================================================

# Input directories
IMERG_BASE_DIR = "/ra1/pubdat/AVHRR_CloudSat_proj/IMERG/IMERGV7"
ERA5_DIR = "/ra1/pubdat/AVHRR_CloudSat_proj/ERA5_0.25deg"

# Land/ocean mask from previous GMI code
USE_LAND_OCEAN_MASK = True
LAND_SEA_MASK_PATH = "/ra1/pubdat/AVHRR_CloudSat_proj/IMERG/ancillary_imerg_data/GPM_IMERG_LandSeaMask.2.nc4"
LAND_SEA_MASK_VAR = "landseamask"

# Output directory
# OUT_DIR = "/home/kkumah/Projects/Comparing_V7_V8_GPROF-GMI_data/Outs/gridded_reference_diagnostics"
OUT_DIR = (
    "/home/kkumah/Projects/Comparing_V7_V8_GPROF-GMI_data/"
    "Outs/gridded_reference_diagnostics_wet002"
)
# Years to process
YEARS = [2010, 2016, 2018]  # change to [2010, 2016, 2018] for full run

# Actual checked GMI swath reaches about -69.37 to +69.40
# Use ±70 so IMERG/ERA5 cover the full practical GMI belt.
GMI_LAT_MIN = -70.0
GMI_LAT_MAX = 70.0

# Wet threshold in mm/hr or mm per hourly timestep
WET_THRESHOLD = 0.02

# Latitude bins.
# Keep same broad structure as GMI code. Since we subset data to ±70,
# bins outside available data will simply have no contribution.
LAT_BIN_WIDTH = 1.0
LAT_BINS = np.arange(-90.0, 90.0 + LAT_BIN_WIDTH, LAT_BIN_WIDTH)

# PDF bins in mm/hr / mm per timestep
PDF_BINS = np.array(
    [0.1, 0.2, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256],
    dtype=float
)

SURFACE_CLASSES = ["all", "land", "ocean"]
HEMISPHERES = ["ALL_HEMI", "NH", "SH"]

# Target reference grid. IMERG is 0.1°, ERA5 is 0.25°.
TARGET_RES = 0.25

# Parallel settings
N_WORKERS_IMERG = 20

# ERA5 time chunk size. 24 = daily chunks for hourly ERA5.
ERA5_TIME_CHUNK = 24

# Use these for quick testing before full run
MAX_IMERG_FILES_PER_YEAR = None
MAX_ERA5_TIMES_PER_YEAR = None


# =============================================================================
# Global worker variables
# =============================================================================

_WORKER_TARGET_LAT = None
_WORKER_TARGET_LON = None
_WORKER_SURFACE_MASKS = None


# =============================================================================
# General helpers
# =============================================================================

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def standardize_lon_180(ds, lon_name="lon"):
    """
    Convert longitude from 0..360 to -180..180 and sort.
    """
    lon = ds[lon_name]

    if float(lon.max()) > 180:
        ds = ds.assign_coords({lon_name: (((lon + 180) % 360) - 180)})
        ds = ds.sortby(lon_name)

    return ds


def build_target_grid(lat_min=GMI_LAT_MIN, lat_max=GMI_LAT_MAX, res=TARGET_RES):
    """
    Build ERA5-like 0.25° target grid over GMI latitude range.
    """
    target_lat = np.arange(lat_min, lat_max + res, res)
    target_lon = np.arange(-180.0, 180.0, res)

    return target_lat, target_lon


def safe_nan_mask(da):
    """
    Mask common missing/fill values.
    """
    da = da.where(np.isfinite(da))
    da = da.where(da > -9990)

    return da


def subset_gmi_lat(da, lat_name="lat"):
    """
    Subset DataArray to approximate GMI latitude coverage.
    Works for ascending or descending latitude.
    """
    lat = da[lat_name]

    if float(lat[0]) > float(lat[-1]):
        return da.sel({lat_name: slice(GMI_LAT_MAX, GMI_LAT_MIN)})
    else:
        return da.sel({lat_name: slice(GMI_LAT_MIN, GMI_LAT_MAX)})


def empty_stats():
    """
    Create an empty accumulator dictionary.
    """
    n_lat_bins = len(LAT_BINS) - 1
    n_pdf_bins = len(PDF_BINS)  # PDF_BINS are upper thresholds, not edges

    return {
        "zonal_sum": np.zeros(n_lat_bins, dtype=np.float64),
        "zonal_valid_count": np.zeros(n_lat_bins, dtype=np.int64),
        "zonal_wet_count": np.zeros(n_lat_bins, dtype=np.int64),
        "zonal_wet_sum": np.zeros(n_lat_bins, dtype=np.float64),

        "pdf_counts": np.zeros(n_pdf_bins, dtype=np.int64),
        "pdf_volume": np.zeros(n_pdf_bins, dtype=np.float64),

        "summary_sum": 0.0,
        "summary_valid_count": 0,
        "summary_wet_count": 0,
        "summary_wet_sum": 0.0,
    }

def add_stats_inplace(base, inc):
    """
    Add one accumulator into another in place.
    """
    base["zonal_sum"] += inc["zonal_sum"]
    base["zonal_valid_count"] += inc["zonal_valid_count"]
    base["zonal_wet_count"] += inc["zonal_wet_count"]
    base["zonal_wet_sum"] += inc["zonal_wet_sum"]

    base["pdf_counts"] += inc["pdf_counts"]
    base["pdf_volume"] += inc["pdf_volume"]

    base["summary_sum"] += inc["summary_sum"]
    base["summary_valid_count"] += inc["summary_valid_count"]
    base["summary_wet_count"] += inc["summary_wet_count"]
    base["summary_wet_sum"] += inc["summary_wet_sum"]

    return base


def init_nested_stats():
    """
    Stats container for all product/surface/hemisphere combinations.

    For zonal stats:
        surface_class matters
        hemisphere is not used because latitude bins carry that information

    For PDF and summary:
        surface_class and hemisphere both matter
    """
    stats = {}

    for surface_class in SURFACE_CLASSES:
        stats[("zonal", surface_class, "ALL_HEMI")] = empty_stats()

    for surface_class in SURFACE_CLASSES:
        for hemi in HEMISPHERES:
            stats[("pdf_summary", surface_class, hemi)] = empty_stats()

    return stats


def combine_nested_stats(stats_list):
    """
    Combine list of nested stats dictionaries.
    """
    combined = init_nested_stats()

    for stats in stats_list:
        for key, inc in stats.items():
            add_stats_inplace(combined[key], inc)

    return combined


# =============================================================================
# File discovery
# =============================================================================

def find_imerg_files_for_year(year):
    """
    Strictly find IMERG V7 half-hourly files for selected years.

    This avoids broad recursive fallback searches that accidentally pull
    duplicate/regridded/derived IMERG folders.
    """

    if year == 2010:
        search_dir = os.path.join(
            IMERG_BASE_DIR,
            "Data_V7_halfhourly_2010"
        )

    elif year in [2016, 2018]:
        search_dir = os.path.join(
            IMERG_BASE_DIR,
            "Data_V7_halfhourly_20162018"
        )

    else:
        raise ValueError(
            f"Year {year} is not configured. "
            "Only 2010, 2016, and 2018 are currently expected."
        )

    pattern = os.path.join(search_dir, f"*{year}*.HDF5")
    files = sorted(glob.glob(pattern))

    clean_files = []
    for f in files:
        fname = os.path.basename(f)

        # Example:
        # 3B-HHR.MS.MRG.3IMERG.20100101-S000000-E002959.0000.V07B.HDF5
        if f"3IMERG.{year}" in fname:
            clean_files.append(f)

    files = sorted(clean_files)

    if MAX_IMERG_FILES_PER_YEAR is not None:
        files = files[:MAX_IMERG_FILES_PER_YEAR]

    expected_non_leap = 365 * 48
    expected_leap = 366 * 48

    print(f"[IMERG] Search directory: {search_dir}")
    print(f"[IMERG] Year {year}: found {len(files)} strict files")

    if MAX_IMERG_FILES_PER_YEAR is None:
        if len(files) not in [expected_non_leap, expected_leap]:
            print(
                f"[IMERG WARNING] File count for {year} is unusual. "
                f"Expected about {expected_non_leap} or {expected_leap}, "
                f"but found {len(files)}."
            )
            if len(files) > 0:
                print("[IMERG] First file:", files[0])
                print("[IMERG] Last file: ", files[-1])

    return files


def find_era5_file_for_year(year):
    """
    Find ERA5 yearly total precipitation file.
    """
    candidates = [
        os.path.join(ERA5_DIR, f"Total_precip_{year}.nc"),
        os.path.join(ERA5_DIR, f"*{year}*.nc"),
    ]

    for pat in candidates:
        files = sorted(glob.glob(pat))
        if len(files) > 0:
            return files[0]

    return None


# =============================================================================
# Land/ocean mask
# =============================================================================

def load_land_ocean_mask(mask_path, varname="landseamask"):
    """
    Load IMERG/GPM land-sea mask and convert to binary land/ocean.

    This follows the exact convention used in the GMI comparison code:
        lsm_binary = xr.where(landseamask < 25, 1, 0)

    Meaning:
        land = 1
        ocean = 0
    """
    ds = xr.open_dataset(mask_path)
    lsm_raw = ds[varname]

    rename_dict = {}
    for d in lsm_raw.dims:
        dl = d.lower()

        if dl in ["latitude", "lat", "y"]:
            rename_dict[d] = "lat"

        if dl in ["longitude", "lon", "x"]:
            rename_dict[d] = "lon"

    if rename_dict:
        lsm_raw = lsm_raw.rename(rename_dict)

    if set(["lat", "lon"]).issubset(set(lsm_raw.dims)):
        lsm_raw = lsm_raw.transpose("lat", "lon")
    else:
        ds.close()
        raise ValueError("Land-sea mask must have lat/lon dimensions.")

    lon = lsm_raw["lon"].values
    if np.nanmax(lon) > 180:
        new_lon = ((lon + 180) % 360) - 180
        lsm_raw = lsm_raw.assign_coords(lon=new_lon).sortby("lon")

    lat = lsm_raw["lat"].values
    if lat[0] > lat[-1]:
        lsm_raw = lsm_raw.sortby("lat")

    lsm_binary = xr.where(lsm_raw < 25, 1, 0)
    lsm_binary.name = "land_ocean_mask"

    ds.close()

    return lsm_binary


def build_surface_masks_on_target_grid(target_lat, target_lon):
    """
    Build surface masks on the 0.25° target grid.

    Returns
    -------
    surface_masks : dict
        surface_masks["all"], ["land"], ["ocean"] are boolean arrays
        with shape (lat, lon).
    """
    all_mask = np.ones((len(target_lat), len(target_lon)), dtype=bool)

    if not USE_LAND_OCEAN_MASK:
        return {
            "all": all_mask,
            "land": np.zeros_like(all_mask, dtype=bool),
            "ocean": np.zeros_like(all_mask, dtype=bool),
        }

    print("[MASK] Loading land/ocean mask:")
    print(f"       {LAND_SEA_MASK_PATH}")

    lsm = load_land_ocean_mask(LAND_SEA_MASK_PATH, LAND_SEA_MASK_VAR)

    # Nearest-neighbor sample to target 0.25° grid.
    lsm_target = lsm.sel(
        lat=xr.DataArray(target_lat, dims="lat"),
        lon=xr.DataArray(target_lon, dims="lon"),
        method="nearest"
    )

    lsm_values = lsm_target.values.astype(float)

    land_mask = lsm_values == 1
    ocean_mask = lsm_values == 0

    print("[MASK] Target grid mask counts:")
    print(f"       all:   {int(all_mask.sum())}")
    print(f"       land:  {int(land_mask.sum())}")
    print(f"       ocean: {int(ocean_mask.sum())}")

    return {
        "all": all_mask,
        "land": land_mask,
        "ocean": ocean_mask,
    }


# =============================================================================
# Product readers and preprocessing
# =============================================================================

def read_one_imerg_file_to_025(file_path, target_lat, target_lon):
    """
    Read one IMERG half-hourly file and remap to 0.25° target grid.

    Returns
    -------
    pr_025 : xr.DataArray
        DataArray with dims (time, lat, lon), units mm/hr.
    """
    ds = xr.open_dataset(file_path, group="Grid")

    if "precipitation" not in ds:
        ds.close()
        raise KeyError(f"'precipitation' not found in {file_path}")

    pr = ds["precipitation"]

    # IMERG is often (time, lon, lat). Convert to conventional order.
    if set(["time", "lat", "lon"]).issubset(set(pr.dims)):
        pr = pr.transpose("time", "lat", "lon")
    else:
        ds.close()
        raise ValueError(f"Unexpected IMERG precipitation dims: {pr.dims}")

    pr = safe_nan_mask(pr)
    pr.attrs["units"] = "mm/hr"

    # Subset before interpolation to reduce memory.
    pr = subset_gmi_lat(pr, lat_name="lat")

    # Ensure latitude increasing for interpolation.
    if float(pr["lat"][0]) > float(pr["lat"][-1]):
        pr = pr.sortby("lat")

    pr_ds = pr.to_dataset(name="precipitation")
    pr_ds = standardize_lon_180(pr_ds, lon_name="lon")
    pr = pr_ds["precipitation"]

    pr_025 = pr.interp(
        lat=target_lat,
        lon=target_lon,
        method="nearest"
    )

    # Force loading inside worker, then close source dataset.
    pr_025 = pr_025.load()

    ds.close()

    return pr_025


def read_era5_year_to_mm(file_path):
    """
    Read ERA5 yearly total precipitation and convert from meters to mm.

    ERA5 file structure:
        tp(time, latitude, longitude), units=m

    Returns
    -------
    pr : xr.DataArray
        DataArray with dims (time, lat, lon), units mm per hourly timestep.
    """
    ds = xr.open_dataset(file_path)

    if "tp" not in ds:
        ds.close()
        raise KeyError(f"'tp' not found in {file_path}")

    pr = ds["tp"] * 1000.0
    pr.attrs["units"] = "mm per timestep"

    rename_dict = {}
    if "latitude" in pr.dims:
        rename_dict["latitude"] = "lat"
    if "longitude" in pr.dims:
        rename_dict["longitude"] = "lon"

    pr = pr.rename(rename_dict)

    pr_ds = pr.to_dataset(name="precipitation")
    pr_ds = standardize_lon_180(pr_ds, lon_name="lon")
    pr = pr_ds["precipitation"]

    pr = subset_gmi_lat(pr, lat_name="lat")

    if float(pr["lat"][0]) > float(pr["lat"][-1]):
        pr = pr.sortby("lat")

    if MAX_ERA5_TIMES_PER_YEAR is not None:
        pr = pr.isel(time=slice(0, MAX_ERA5_TIMES_PER_YEAR))

    return pr


# =============================================================================
# Diagnostic accumulation
# =============================================================================

def hemisphere_mask_from_lat(lat2d, hemi):
    """
    Build hemisphere mask from 2D latitude grid.
    """
    if hemi == "ALL_HEMI":
        return np.ones_like(lat2d, dtype=bool)

    if hemi == "NH":
        return lat2d >= 0

    if hemi == "SH":
        return lat2d < 0

    raise ValueError(f"Unknown hemisphere: {hemi}")


def compute_stats_for_values(values):
    """
    Compute compact stats from 1D precipitation values.

    Important:
    ----------
    Wet fraction uses WET_THRESHOLD.

    PDF bins follow the GPROF comparison code:
        first bin: values <= PDF_BINS[0]
        later bins: PDF_BINS[i-1] < values <= PDF_BINS[i]

    PDFc is based on all valid nonnegative values, including zeros/dry values.
    PDFv is based on precipitation volume in each bin; zeros contribute no volume.
    """
    stats = empty_stats()

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    # Match GPROF PDF convention: keep all valid nonnegative precipitation values.
    values = values[values >= 0]

    if values.size == 0:
        return stats

    # Summary and wet-fraction statistics
    wet = values >= WET_THRESHOLD
    wet_values = values[wet]

    stats["summary_sum"] = float(np.nansum(values))
    stats["summary_valid_count"] = int(values.size)
    stats["summary_wet_count"] = int(np.sum(wet))
    stats["summary_wet_sum"] = float(np.nansum(wet_values)) if wet_values.size > 0 else 0.0

    # PDFc/PDFv using same bin convention as GPROF code
    for i, high in enumerate(PDF_BINS):
        if i == 0:
            low = 0.0
            in_bin = values <= high
        else:
            low = PDF_BINS[i - 1]
            in_bin = (values > low) & (values <= high)

        bin_values = values[in_bin]

        stats["pdf_counts"][i] += int(bin_values.size)
        stats["pdf_volume"][i] += float(np.nansum(bin_values)) if bin_values.size > 0 else 0.0

    return stats


def compute_partial_stats_from_dataarray(pr, surface_masks):
    """
    Compute compact diagnostics from one DataArray.

    Input
    -----
    pr : xr.DataArray
        dims (time, lat, lon)

    surface_masks : dict
        boolean 2D masks on target lat/lon grid.
    """
    result = init_nested_stats()

    lat = pr["lat"].values
    lon = pr["lon"].values

    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")

    arr = pr.values

    # Should be (time, lat, lon)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array (time, lat, lon), got shape {arr.shape}")

    # -------------------------------------------------------------------------
    # Zonal stats by surface class
    # -------------------------------------------------------------------------
    for surface_class in SURFACE_CLASSES:
        smask = surface_masks[surface_class]

        stats = result[("zonal", surface_class, "ALL_HEMI")]

        for i in range(len(LAT_BINS) - 1):
            lat0 = LAT_BINS[i]
            lat1 = LAT_BINS[i + 1]

            lat_band = (lat2d >= lat0) & (lat2d < lat1)
            mask2d = smask & lat_band

            if not np.any(mask2d):
                continue

            vals = arr[:, mask2d]
            vals = vals[np.isfinite(vals)]
            vals = vals[vals >= 0]

            if vals.size == 0:
                continue

            wet = vals >= WET_THRESHOLD
            wet_vals = vals[wet]

            stats["zonal_sum"][i] += np.nansum(vals)
            stats["zonal_valid_count"][i] += vals.size
            stats["zonal_wet_count"][i] += np.sum(wet)
            stats["zonal_wet_sum"][i] += np.nansum(wet_vals) if wet_vals.size > 0 else 0.0

    # -------------------------------------------------------------------------
    # PDF + summary by surface class and hemisphere
    # -------------------------------------------------------------------------
    for surface_class in SURFACE_CLASSES:
        smask = surface_masks[surface_class]

        for hemi in HEMISPHERES:
            hmask = hemisphere_mask_from_lat(lat2d, hemi)
            mask2d = smask & hmask

            if not np.any(mask2d):
                continue

            vals = arr[:, mask2d]
            partial = compute_stats_for_values(vals.ravel())

            add_stats_inplace(
                result[("pdf_summary", surface_class, hemi)],
                partial
            )

    return result


def nested_stats_to_zonal_records(stats, product, year):
    """
    Convert combined zonal accumulators to records.
    """
    records = []

    for surface_class in SURFACE_CLASSES:
        s = stats[("zonal", surface_class, "ALL_HEMI")]

        for i in range(len(LAT_BINS) - 1):
            lat_min = LAT_BINS[i]
            lat_max = LAT_BINS[i + 1]
            lat_mid = 0.5 * (lat_min + lat_max)

            n_valid = int(s["zonal_valid_count"][i])
            n_wet = int(s["zonal_wet_count"][i])
            precip_sum = float(s["zonal_sum"][i])
            wet_sum = float(s["zonal_wet_sum"][i])

            mean_precip = precip_sum / n_valid if n_valid > 0 else np.nan
            wet_fraction_percent = 100.0 * n_wet / n_valid if n_valid > 0 else np.nan
            wet_mean_precip = wet_sum / n_wet if n_wet > 0 else np.nan

            records.append({
                "aggregation_method": "pooled",
                "product": product,
                "surface_class": surface_class,
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lat_mid": lat_mid,
                "n_granules_contributing": np.nan,
                "n_valid": n_valid,
                "n_wet": n_wet,
                "mean_precip_mmhr": mean_precip,
                "wet_fraction_percent": wet_fraction_percent,
                "wet_mean_precip_mmhr": wet_mean_precip,
                "year": year,
                "source": "gridded_reference",
                "wet_threshold_mmhr": WET_THRESHOLD,
            })

    return records


def nested_stats_to_pdf_records(stats, product, year):
    """
    Convert combined PDF accumulators to records.

    Uses the same bin convention as GPROF:
        <= 0.1
        0.1 - 0.2
        0.2 - 0.5
        ...
        128 - 256
    """
    records = []

    for surface_class in SURFACE_CLASSES:
        for hemi in HEMISPHERES:
            s = stats[("pdf_summary", surface_class, hemi)]

            total_count = int(np.sum(s["pdf_counts"]))
            total_volume = float(np.nansum(s["pdf_volume"]))

            for i, high in enumerate(PDF_BINS):
                if i == 0:
                    low = 0.0
                    bin_label = f"<= {high:g}"
                    bin_mid = high
                else:
                    low = PDF_BINS[i - 1]
                    bin_label = f"{low:g}-{high:g}"
                    bin_mid = np.sqrt(low * high)

                count = int(s["pdf_counts"][i])
                volume = float(s["pdf_volume"][i])

                pdfc = 100.0 * count / total_count if total_count > 0 else np.nan
                pdfv = 100.0 * volume / total_volume if total_volume > 0 else np.nan

                records.append({
                    "product": product,
                    "hemisphere": hemi,
                    "surface_class": surface_class,
                    "bin_low": low,
                    "bin_high": high,
                    "bin_label": bin_label,
                    "bin_mid": bin_mid,
                    "count": count,
                    "volume": volume,
                    "pdfc": pdfc,
                    "pdfv": pdfv,
                    "aggregation_method": "pooled",
                    "year": year,
                    "source": "gridded_reference",
                    "wet_threshold_mmhr": WET_THRESHOLD,
                })

    return records


def nested_stats_to_summary_records(stats, product, year):
    """
    Convert combined summary accumulators to records.
    """
    records = []

    for surface_class in SURFACE_CLASSES:
        for hemi in HEMISPHERES:
            s = stats[("pdf_summary", surface_class, hemi)]

            n_valid = int(s["summary_valid_count"])
            n_wet = int(s["summary_wet_count"])

            precip_sum = float(s["summary_sum"])
            wet_sum = float(s["summary_wet_sum"])

            mean_precip = precip_sum / n_valid if n_valid > 0 else np.nan
            wet_fraction_percent = 100.0 * n_wet / n_valid if n_valid > 0 else np.nan
            wet_mean_precip = wet_sum / n_wet if n_wet > 0 else np.nan

            records.append({
                "product": product,
                "year": year,
                "hemisphere": hemi,
                "surface_class": surface_class,
                "mean_precip_mmhr": mean_precip,
                "wet_fraction_percent": wet_fraction_percent,
                "wet_mean_precip_mmhr": wet_mean_precip,
                "n_valid": n_valid,
                "n_wet": n_wet,
                "aggregation_method": "pooled",
                "source": "gridded_reference",
                "wet_threshold_mmhr": WET_THRESHOLD,
            })

    return records


# =============================================================================
# IMERG parallel worker
# =============================================================================

def _init_imerg_worker(target_lat, target_lon, surface_masks):
    """
    Initializer for IMERG process workers.
    """
    global _WORKER_TARGET_LAT
    global _WORKER_TARGET_LON
    global _WORKER_SURFACE_MASKS

    _WORKER_TARGET_LAT = target_lat
    _WORKER_TARGET_LON = target_lon
    _WORKER_SURFACE_MASKS = surface_masks


def _process_one_imerg_file_worker(file_path):
    """
    Worker function for parallel IMERG streaming diagnostics.

    Returns compact nested stats, not a full precipitation array.
    """
    try:
        pr_025 = read_one_imerg_file_to_025(
            file_path,
            _WORKER_TARGET_LAT,
            _WORKER_TARGET_LON
        )

        partial = compute_partial_stats_from_dataarray(
            pr_025,
            _WORKER_SURFACE_MASKS
        )

        del pr_025

        return partial

    except Exception as e:
        return {
            "file": file_path,
            "error": str(e),
        }


# =============================================================================
# Processing functions
# =============================================================================

def process_imerg_year(year, target_lat, target_lon, surface_masks, n_workers=8):
    """
    Process one year of IMERG files in parallel using streaming diagnostics.

    This avoids concatenating all half-hourly fields.
    """
    files = find_imerg_files_for_year(year)

    if len(files) == 0:
        print(f"[IMERG] No files found for {year}")
        return [], [], []

    print(f"[IMERG] {year}: found {len(files)} files")
    print(f"[IMERG] Processing with {n_workers} workers")
    print("[IMERG] Streaming mode: workers return compact stats, not full arrays")

    partial_stats_list = []
    errors = []

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_imerg_worker,
        initargs=(target_lat, target_lon, surface_masks),
    ) as executor:

        futures = [executor.submit(_process_one_imerg_file_worker, f) for f in files]

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"IMERG {year}"):
            result = future.result()

            if result is None:
                continue

            if isinstance(result, dict) and "error" in result:
                errors.append(result)
                continue

            partial_stats_list.append(result)

    if len(errors) > 0:
        print(f"[IMERG] {year}: {len(errors)} files failed")
        for err in errors[:10]:
            print(f"    Failed: {err['file']}")
            print(f"    Error:  {err['error']}")

    if len(partial_stats_list) == 0:
        print(f"[IMERG] {year}: no valid files after processing")
        return [], [], []

    print(f"[IMERG] {year}: combining {len(partial_stats_list)} partial-stat chunks")

    combined = combine_nested_stats(partial_stats_list)

    zonal_records = nested_stats_to_zonal_records(combined, "IMERG", year)
    pdf_records = nested_stats_to_pdf_records(combined, "IMERG", year)
    summary_records = nested_stats_to_summary_records(combined, "IMERG", year)

    return zonal_records, pdf_records, summary_records


def process_era5_year(year, target_lat, target_lon, surface_masks):
    """
    Process one year of ERA5 in time chunks using the same streaming diagnostics.
    """
    file_path = find_era5_file_for_year(year)

    if file_path is None:
        print(f"[ERA5] No file found for {year}")
        return [], [], []

    print(f"[ERA5] {year}: {file_path}")
    print(f"[ERA5] Processing in time chunks of {ERA5_TIME_CHUNK}")

    pr = read_era5_year_to_mm(file_path)

    ntime = pr.sizes["time"]

    combined = init_nested_stats()

    for start in tqdm(range(0, ntime, ERA5_TIME_CHUNK), desc=f"ERA5 {year}"):
        stop = min(start + ERA5_TIME_CHUNK, ntime)

        pr_chunk = pr.isel(time=slice(start, stop)).load()

        partial = compute_partial_stats_from_dataarray(
            pr_chunk,
            surface_masks
        )

        combined = combine_nested_stats([combined, partial])

        del pr_chunk

    zonal_records = nested_stats_to_zonal_records(combined, "ERA5", year)
    pdf_records = nested_stats_to_pdf_records(combined, "ERA5", year)
    summary_records = nested_stats_to_summary_records(combined, "ERA5", year)

    return zonal_records, pdf_records, summary_records


# =============================================================================
# Main
# =============================================================================

def main():
    ensure_dir(OUT_DIR)

    target_lat, target_lon = build_target_grid()
    surface_masks = build_surface_masks_on_target_grid(target_lat, target_lon)

    print("\n[GRID]")
    print(f"Target lat: {target_lat[0]} to {target_lat[-1]}  n={len(target_lat)}")
    print(f"Target lon: {target_lon[0]} to {target_lon[-1]}  n={len(target_lon)}")
    print(f"Target resolution: {TARGET_RES} degree")

    all_zonal = []
    all_pdf = []
    all_summary = []

    for year in YEARS:
        print("\n" + "=" * 80)
        print(f"Processing year {year}")
        print("=" * 80)

        z, p, s = process_imerg_year(
            year,
            target_lat=target_lat,
            target_lon=target_lon,
            surface_masks=surface_masks,
            n_workers=N_WORKERS_IMERG,
        )

        all_zonal.extend(z)
        all_pdf.extend(p)
        all_summary.extend(s)

        z, p, s = process_era5_year(
            year,
            target_lat=target_lat,
            target_lon=target_lon,
            surface_masks=surface_masks,
        )

        all_zonal.extend(z)
        all_pdf.extend(p)
        all_summary.extend(s)

    zonal_df = pd.DataFrame(all_zonal)
    pdf_df = pd.DataFrame(all_pdf)
    summary_df = pd.DataFrame(all_summary)

    zonal_csv = os.path.join(OUT_DIR, "reference_zonal_stats.csv")
    pdf_csv = os.path.join(OUT_DIR, "reference_pdf_stats.csv")
    summary_csv = os.path.join(OUT_DIR, "reference_summary_stats.csv")

    zonal_pkl = os.path.join(OUT_DIR, "reference_zonal_stats.pkl")
    pdf_pkl = os.path.join(OUT_DIR, "reference_pdf_stats.pkl")
    summary_pkl = os.path.join(OUT_DIR, "reference_summary_stats.pkl")

    zonal_df.to_csv(zonal_csv, index=False)
    pdf_df.to_csv(pdf_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    zonal_df.to_pickle(zonal_pkl)
    pdf_df.to_pickle(pdf_pkl)
    summary_df.to_pickle(summary_pkl)

    print("\nSaved:")
    print(zonal_csv)
    print(pdf_csv)
    print(summary_csv)
    print(zonal_pkl)
    print(pdf_pkl)
    print(summary_pkl)

    print("\nSummary preview:")
    print(summary_df.head(20))


if __name__ == "__main__":
    main()
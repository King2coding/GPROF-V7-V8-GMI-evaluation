#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_gprof_v7_v8_orbit.py

Minimal orbital/granule-level comparison of matched GPROF V7 and V8 GPM-GMI files.

Purpose
-------
This script compares matched GPROF V7 and V8 orbital granules directly in swath space.
It is designed for quick diagnostics before moving to gridded monthly/seasonal analysis.

Main diagnostics
----------------
1. Per-orbit summary statistics
2. Orbit-sampled zonal mean precipitation
3. Orbit-sampled zonal wet-pixel fraction
4. PDF by occurrence, PDFc
5. PDF by volume, PDFv
6. Optional land/ocean separation using a gridded IMERG land-sea mask sampled to swath pixels

Notes
-----
- V7 sample format: HDF5
- V8 sample format: NetCDF
- Expected group: S1
- Expected dimensions: nscan x npixel
- Main precipitation variable: surfacePrecipitation
"""

# =============================================================================
# Imports
# =============================================================================

import os
from datetime import date
import re
import glob
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import xarray as xr

from joblib import Parallel, delayed


# =============================================================================
# User settings
# =============================================================================

# Main directories
V7_DIR = "/scratch/kkumah/GPM_GMI/V7"
V8_DIR = "/scratch/kkumah/GPM_GMI/V8"

# Output directory
OUT_DIR = "/home/kkumah/Projects/Comparing_V7_V8_GPROF-GMI_data/Outs/orbit_v7_v8_comparison"

# Optional land-sea mask
USE_LAND_OCEAN_MASK = True
LAND_SEA_MASK_PATH = "/ra1/pubdat/AVHRR_CloudSat_proj/IMERG/ancillary_imerg_data/GPM_IMERG_LandSeaMask.2.nc4"
LAND_SEA_MASK_VAR = "landseamask"

# Parallel computing
N_JOBS = 20

# Use only first N matched granules for testing.
# Set to None to process all matched files.
MAX_PAIRS = None

# Wet threshold in mm/hr
WET_THRESHOLD = 0.02

# Percent difference denominator threshold.
# Percent difference is only computed where V7 >= this value.
PERCENT_DENOM_THRESHOLD = 0.1

# Latitude bins for orbit-sampled zonal statistics
LAT_BIN_WIDTH = 1.0
LAT_BINS = np.arange(-90, 90 + LAT_BIN_WIDTH, LAT_BIN_WIDTH)

# PDF bins in mm/hr
PDF_BINS = np.array([0.1, 0.2, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256], dtype=float)

# Surface classes to summarize
# all is always used.
# land/ocean are used only if USE_LAND_OCEAN_MASK=True and mask is available.
SURFACE_CLASSES = ["all", "land", "ocean"]

# Aggregation method for multi-granule outputs
# "pooled"       = pixel/volume weighted aggregation across all matched pixels
# "granule_mean" = average of per-granule metrics; each granule gets equal weight
# "both"         = save both pooled and granule-mean outputs
AGGREGATION_METHOD = "both"

cde_run_date = date.today().strftime('%Y%m%d')

sve_fle_part = f"{MAX_PAIRS}_{cde_run_date}"

# =============================================================================
# Helper functions: file matching
# =============================================================================

def extract_granule_key(filepath):
    """
    Extract a matching key from GPROF filename.

    Example filename:
    2A-CLIM.GPM.GMI.GPROF2021v1.20140304-S175932-E193159.000079.V07A.HDF5
    2A-CLIM.GPM.GMI.GPROFNNv1.20140304-S175932-E193159.000079.V08A.nc

    Returned key:
    20140304-S175932-E193159.000079
    """
    fname = os.path.basename(filepath)

    pattern = r"(\d{8}-S\d{6}-E\d{6}\.\d{6})"
    match = re.search(pattern, fname)

    if match is None:
        return None

    return match.group(1)


def find_matched_pairs(v7_dir, v8_dir):
    """
    Find matched V7 and V8 files based on timestamp + granule number key.
    """
    v7_files = sorted(
        glob.glob(os.path.join(v7_dir, "*.HDF5")) +
        glob.glob(os.path.join(v7_dir, "*.h5")) +
        glob.glob(os.path.join(v7_dir, "*.hdf5"))
    )

    v8_files = sorted(
        glob.glob(os.path.join(v8_dir, "*.nc")) +
        glob.glob(os.path.join(v8_dir, "*.NC")) +
        glob.glob(os.path.join(v8_dir, "*.nc4"))
    )

    v7_dict = {}
    for f in v7_files:
        key = extract_granule_key(f)
        if key is not None:
            v7_dict[key] = f

    v8_dict = {}
    for f in v8_files:
        key = extract_granule_key(f)
        if key is not None:
            v8_dict[key] = f

    common_keys = sorted(set(v7_dict.keys()) & set(v8_dict.keys()))

    pairs = []
    for key in common_keys:
        pairs.append({
            "key": key,
            "v7_file": v7_dict[key],
            "v8_file": v8_dict[key],
        })

    return pairs


# =============================================================================
# Helper functions: reading GPROF files
# =============================================================================

def _read_hdf5_dataset(h5_group, name):
    """
    Read HDF5 dataset safely.
    """
    if name not in h5_group:
        raise KeyError(f"Dataset '{name}' not found in HDF5 group.")

    arr = h5_group[name][()]
    return np.asarray(arr)


def read_gprof_v7_hdf5(filepath, group_name="S1"):
    """
    Read GPROF V7 HDF5 swath data.
    """
    with h5py.File(filepath, "r") as f:
        if group_name not in f:
            raise KeyError(f"Group '{group_name}' not found in {filepath}")

        g = f[group_name]

        lat = _read_hdf5_dataset(g, "Latitude")
        lon = _read_hdf5_dataset(g, "Longitude")
        precip = _read_hdf5_dataset(g, "surfacePrecipitation")

        probability = _read_hdf5_dataset(g, "probabilityOfPrecip") if "probabilityOfPrecip" in g else None
        quality = _read_hdf5_dataset(g, "qualityFlag") if "qualityFlag" in g else None
        pixel_status = _read_hdf5_dataset(g, "pixelStatus") if "pixelStatus" in g else None

        attrs = {}
        for k, v in f.attrs.items():
            try:
                attrs[k] = v.decode() if isinstance(v, bytes) else v
            except Exception:
                attrs[k] = v

    precip = mask_invalid_precip(precip)

    return {
        "lat": lat,
        "lon": lon,
        "precip": precip,
        "probability": probability,
        "quality": quality,
        "pixel_status": pixel_status,
        "attrs": attrs,
    }


def read_gprof_v8_nc(filepath, group_name="S1"):
    """
    Read GPROF V8 NetCDF swath data.
    """
    ds = xr.open_dataset(filepath, group=group_name)

    lat = ds["Latitude"].values
    lon = ds["Longitude"].values
    precip = ds["surfacePrecipitation"].values

    probability = ds["probabilityOfPrecip"].values if "probabilityOfPrecip" in ds else None
    quality = ds["qualityFlag"].values if "qualityFlag" in ds else None
    pixel_status = ds["pixelStatus"].values if "pixelStatus" in ds else None

    attrs = dict(ds.attrs)
    ds.close()

    precip = mask_invalid_precip(precip)

    return {
        "lat": lat,
        "lon": lon,
        "precip": precip,
        "probability": probability,
        "quality": quality,
        "pixel_status": pixel_status,
        "attrs": attrs,
    }


def mask_invalid_precip(arr):
    """
    Convert invalid/fill precipitation values to NaN.

    GPROF files commonly use values near -9999.9 for fill.
    We also mask negative values because precipitation should not be negative.
    """
    arr = np.asarray(arr, dtype=float)

    arr = np.where(arr <= -9990, np.nan, arr)
    arr = np.where(arr < 0, np.nan, arr)

    return arr


# =============================================================================
# Helper functions: land/ocean mask
# =============================================================================

def load_land_ocean_mask(mask_path, varname="landseamask"):
    """
    Load IMERG/GPM land-sea mask and convert to binary land/ocean.

    Based on user's previous convention:
        lsm = xr.where(landseamask < 25, 1, 0)

    Here:
        land = 1
        ocean = 0

    The function makes sure latitude increases from south to north
    and longitude is in -180 to 180.
    """
    ds = xr.open_dataset(mask_path)
    lsm_raw = ds[varname]

    # Ensure dimensions are named lat/lon
    rename_dict = {}
    for d in lsm_raw.dims:
        if d.lower() in ["latitude", "y"]:
            rename_dict[d] = "lat"
        if d.lower() in ["longitude", "x"]:
            rename_dict[d] = "lon"

    if rename_dict:
        lsm_raw = lsm_raw.rename(rename_dict)

    # Ensure order is lat, lon
    if set(["lat", "lon"]).issubset(set(lsm_raw.dims)):
        lsm_raw = lsm_raw.transpose("lat", "lon")
    else:
        raise ValueError("Land-sea mask must have lat/lon dimensions.")

    # Convert longitude to -180 to 180 if needed
    lon = lsm_raw["lon"].values
    if np.nanmax(lon) > 180:
        new_lon = ((lon + 180) % 360) - 180
        lsm_raw = lsm_raw.assign_coords(lon=new_lon).sortby("lon")

    # Make latitude increasing south to north
    lat = lsm_raw["lat"].values
    if lat[0] > lat[-1]:
        lsm_raw = lsm_raw.sortby("lat")

    # User's previous convention
    lsm_binary = xr.where(lsm_raw < 25, 1, 0)
    lsm_binary.name = "land_ocean_mask"

    ds.close()

    return lsm_binary


def sample_lsm_to_swath(lat2d, lon2d, lsm):
    """
    Sample gridded land/ocean mask to GPROF swath pixels using nearest neighbor.

    Returns
    -------
    mask2d : np.ndarray
        land = 1, ocean = 0, invalid = NaN
    """
    lat_flat = lat2d.ravel()
    lon_flat = lon2d.ravel()

    # Convert lon to mask convention
    lon_flat = ((lon_flat + 180) % 360) - 180

    points = xr.Dataset(
        coords={
            "points": np.arange(lat_flat.size),
            "lat": ("points", lat_flat),
            "lon": ("points", lon_flat),
        }
    )

    sampled = lsm.sel(
        lat=points["lat"],
        lon=points["lon"],
        method="nearest"
    )

    mask_flat = sampled.values.astype(float)
    mask2d = mask_flat.reshape(lat2d.shape)

    return mask2d


# =============================================================================
# Helper functions: statistics
# =============================================================================

def safe_nanmean(arr):
    """
    Nanmean with warning suppression.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(arr)


def compute_basic_summary(values, prefix):
    """
    Compute basic summary stats for a precipitation-like array.
    """
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)

    if valid.sum() == 0:
        return {
            f"{prefix}_valid_count": 0,
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_p95": np.nan,
            f"{prefix}_p99": np.nan,
        }

    vv = values[valid]

    return {
        f"{prefix}_valid_count": int(vv.size),
        f"{prefix}_mean": float(np.nanmean(vv)),
        f"{prefix}_median": float(np.nanmedian(vv)),
        f"{prefix}_std": float(np.nanstd(vv)),
        f"{prefix}_min": float(np.nanmin(vv)),
        f"{prefix}_max": float(np.nanmax(vv)),
        f"{prefix}_p95": float(np.nanpercentile(vv, 95)),
        f"{prefix}_p99": float(np.nanpercentile(vv, 99)),
    }


def compute_pdf_elements_from_array(values, bins, label):
    """
    Compute PDF by occurrence and PDF by volume for 1-D precipitation values.

    Parameters
    ----------
    values : array
        Precipitation values.
    bins : array
        Upper bin edges.
    label : str
        Label for product/version, e.g., V7, V8.

    Returns
    -------
    DataFrame with:
        product, bin_low, bin_high, bin_label, pdfc, pdfv, count, volume
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    # For PDFs, keep only nonnegative valid precipitation values.
    values = values[values >= 0]

    total_count = len(values)

    rows = []

    if total_count == 0:
        for i, high in enumerate(bins):
            low = 0 if i == 0 else bins[i - 1]
            rows.append({
                "product": label,
                "bin_low": low,
                "bin_high": high,
                "bin_label": f"<= {high}" if i == 0 else f"{low} - {high}",
                "count": 0,
                "volume": 0.0,
                "pdfc": np.nan,
                "pdfv": np.nan,
                "wet_threshold_mmhr": WET_THRESHOLD,
            })
        return pd.DataFrame(rows)

    volumes = []

    for i, high in enumerate(bins):
        if i == 0:
            low = 0.0
            bin_values = values[values <= high]
            bin_label = f"<= {high}"
        else:
            low = bins[i - 1]
            bin_values = values[(values > low) & (values <= high)]
            bin_label = f"{low} - {high}"

        count = len(bin_values)
        volume = float(np.nansum(bin_values)) if count > 0 else 0.0
        volumes.append(volume)

        rows.append({
            "product": label,
            "bin_low": low,
            "bin_high": high,
            "bin_label": bin_label,
            "count": int(count),
            "volume": volume,
            "pdfc": 100.0 * count / total_count,
            "pdfv": np.nan,
            "wet_threshold_mmhr": WET_THRESHOLD,
        })

    total_volume = np.nansum(volumes)

    for r in rows:
        if total_volume > 0:
            r["pdfv"] = 100.0 * r["volume"] / total_volume
        else:
            r["pdfv"] = np.nan

    return pd.DataFrame(rows)


def compute_zonal_stats(lat, precip, wet_threshold, lat_bins, class_name, product_name):
    """
    Compute orbit-sampled zonal mean precipitation and wet-pixel fraction.

    This does not require gridding.
    It bins swath pixels by latitude.

    Returns
    -------
    DataFrame
    """
    lat = np.asarray(lat, dtype=float).ravel()
    precip = np.asarray(precip, dtype=float).ravel()

    valid = np.isfinite(lat) & np.isfinite(precip)

    lat = lat[valid]
    precip = precip[valid]

    rows = []

    for i in range(len(lat_bins) - 1):
        lat_min = lat_bins[i]
        lat_max = lat_bins[i + 1]
        lat_mid = 0.5 * (lat_min + lat_max)

        in_bin = (lat >= lat_min) & (lat < lat_max)

        n_valid = int(np.sum(in_bin))

        if n_valid == 0:
            mean_precip = np.nan
            wet_fraction = np.nan
            wet_mean_precip = np.nan
            n_wet = 0
        else:
            p = precip[in_bin]
            n_wet = int(np.sum(p >= wet_threshold))
            mean_precip = float(np.nanmean(p))
            wet_fraction = 100.0 * n_wet / n_valid

            if n_wet > 0:
                wet_mean_precip = float(np.nanmean(p[p >= wet_threshold]))
            else:
                wet_mean_precip = np.nan

        rows.append({
            "product": product_name,
            "surface_class": class_name,
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lat_mid": lat_mid,
            "n_valid": n_valid,
            "n_wet": n_wet,
            "mean_precip_mmhr": mean_precip,
            "wet_fraction_percent": wet_fraction,
            "wet_mean_precip_mmhr": wet_mean_precip,
            "wet_threshold_mmhr": wet_threshold,
        })

    return pd.DataFrame(rows)


# =============================================================================
# Main per-pair processing
# =============================================================================

def process_one_pair(pair, lsm=None):
    """
    Process one matched V7/V8 granule pair.

    Returns
    -------
    summary_df, zonal_df, pdf_df
    """
    key = pair["key"]
    v7_file = pair["v7_file"]
    v8_file = pair["v8_file"]

    try:
        v7 = read_gprof_v7_hdf5(v7_file)
        v8 = read_gprof_v8_nc(v8_file)

        lat7 = v7["lat"]
        lon7 = v7["lon"]
        p7 = v7["precip"]

        lat8 = v8["lat"]
        lon8 = v8["lon"]
        p8 = v8["precip"]

        # Basic shape check
        if p7.shape != p8.shape:
            raise ValueError(f"Shape mismatch for {key}: V7 {p7.shape}, V8 {p8.shape}")

        # Use V8 latitude/longitude for binning, but check geolocation closeness.
        lat_diff_mean = safe_nanmean(np.abs(lat8 - lat7))
        lon_diff_mean = safe_nanmean(np.abs(lon8 - lon7))

        # Common valid mask for direct V8-V7 comparison
        common_valid = (
            np.isfinite(p7) &
            np.isfinite(p8) &
            np.isfinite(lat8) &
            np.isfinite(lon8)
        )

        diff = np.full_like(p8, np.nan, dtype=float)
        diff[common_valid] = p8[common_valid] - p7[common_valid]

        percent_diff = np.full_like(p8, np.nan, dtype=float)
        pct_mask = common_valid & (p7 >= PERCENT_DENOM_THRESHOLD)
        percent_diff[pct_mask] = 100.0 * (p8[pct_mask] - p7[pct_mask]) / p7[pct_mask]

        wet7 = common_valid & (p7 >= WET_THRESHOLD)
        wet8 = common_valid & (p8 >= WET_THRESHOLD)

        # Land/ocean mask sampled to swath pixels
        if lsm is not None:
            swath_lsm = sample_lsm_to_swath(lat8, lon8, lsm)
        else:
            swath_lsm = None

        # ---------------------------------------------------------------------
        # Summary statistics
        # ---------------------------------------------------------------------
        row = {
            "key": key,
            "v7_file": os.path.basename(v7_file),
            "v8_file": os.path.basename(v8_file),
            "wet_threshold_mmhr": WET_THRESHOLD,
            "shape_nscan": p7.shape[0],
            "shape_npixel": p7.shape[1],
            "lat_diff_mean_abs": lat_diff_mean,
            "lon_diff_mean_abs": lon_diff_mean,
            "common_valid_count": int(np.sum(common_valid)),
            "v7_wet_count": int(np.sum(wet7)),
            "v8_wet_count": int(np.sum(wet8)),
            "v7_wet_fraction_percent": 100.0 * np.sum(wet7) / np.sum(common_valid) if np.sum(common_valid) > 0 else np.nan,
            "v8_wet_fraction_percent": 100.0 * np.sum(wet8) / np.sum(common_valid) if np.sum(common_valid) > 0 else np.nan,
            "wet_fraction_diff_v8_minus_v7_percent": (
                100.0 * np.sum(wet8) / np.sum(common_valid)
                - 100.0 * np.sum(wet7) / np.sum(common_valid)
            ) if np.sum(common_valid) > 0 else np.nan,
        }

        row.update(compute_basic_summary(p7[common_valid], "v7_precip"))
        row.update(compute_basic_summary(p8[common_valid], "v8_precip"))
        row.update(compute_basic_summary(diff[common_valid], "diff_v8_minus_v7"))
        row.update(compute_basic_summary(percent_diff[np.isfinite(percent_diff)], "percent_diff"))

        summary_df = pd.DataFrame([row])

        # ---------------------------------------------------------------------
        # Zonal stats
        # ---------------------------------------------------------------------
        zonal_list = []

        class_masks = {
            "all": common_valid,
        }

        if swath_lsm is not None:
            class_masks["land"] = common_valid & (swath_lsm == 1)
            class_masks["ocean"] = common_valid & (swath_lsm == 0)

        for class_name, cmask in class_masks.items():
            zonal_list.append(
                compute_zonal_stats(
                    lat=lat8[cmask],
                    precip=p7[cmask],
                    wet_threshold=WET_THRESHOLD,
                    lat_bins=LAT_BINS,
                    class_name=class_name,
                    product_name="V7",
                )
            )

            zonal_list.append(
                compute_zonal_stats(
                    lat=lat8[cmask],
                    precip=p8[cmask],
                    wet_threshold=WET_THRESHOLD,
                    lat_bins=LAT_BINS,
                    class_name=class_name,
                    product_name="V8",
                )
            )

            zonal_list.append(
                compute_zonal_stats(
                    lat=lat8[cmask],
                    precip=diff[cmask],
                    wet_threshold=0.0,
                    lat_bins=LAT_BINS,
                    class_name=class_name,
                    product_name="V8_minus_V7",
                )
            )

        zonal_df = pd.concat(zonal_list, ignore_index=True)
        zonal_df.insert(0, "key", key)

        # ---------------------------------------------------------------------
        # PDFs
        # ---------------------------------------------------------------------
        pdf_list = []

        hemi_masks = {
            "ALL_HEMI": common_valid,
            "NH": common_valid & (lat8 >= 0),
            "SH": common_valid & (lat8 < 0),
        }

        for hemi_name, hmask in hemi_masks.items():
            for class_name, cmask in class_masks.items():

                final_mask = hmask & cmask

                tmp_v7 = compute_pdf_elements_from_array(
                    p7[final_mask],
                    PDF_BINS,
                    "V7"
                )

                tmp_v8 = compute_pdf_elements_from_array(
                    p8[final_mask],
                    PDF_BINS,
                    "V8"
                )

                tmp_v7.insert(0, "hemisphere", hemi_name)
                tmp_v8.insert(0, "hemisphere", hemi_name)

                tmp_v7.insert(1, "surface_class", class_name)
                tmp_v8.insert(1, "surface_class", class_name)

                pdf_list.extend([tmp_v7, tmp_v8])

        pdf_df = pd.concat(pdf_list, ignore_index=True)
        pdf_df.insert(0, "key", key)
        return summary_df, zonal_df, pdf_df

    except Exception as e:
        error_df = pd.DataFrame([{
            "key": key,
            "v7_file": os.path.basename(v7_file),
            "v8_file": os.path.basename(v8_file),
            "error": str(e),
        }])

        return error_df, None, None


# =============================================================================
# Aggregation helpers
# =============================================================================

def aggregate_zonal_stats(zonal_all, method="pooled"):
    """
    Aggregate zonal statistics across many matched orbit pairs.

    Parameters
    ----------
    zonal_all : DataFrame
        Per-orbit zonal statistics.
    method : str
        "pooled":
            Pixel-count weighted aggregation.
            Mean precipitation is weighted by n_valid.
            Wet fraction is total n_wet / total n_valid.

        "granule_mean":
            Each granule/bin contributes equally.
            Mean precipitation is mean of per-granule bin means.
            Wet fraction is mean of per-granule wet fractions.

    Returns
    -------
    DataFrame
    """
    if method not in ["pooled", "granule_mean"]:
        raise ValueError("method must be 'pooled' or 'granule_mean'")

    df = zonal_all.copy()

    # Keep only bins with valid data
    df = df[df["n_valid"] > 0].copy()

    if df.empty:
        return pd.DataFrame()

    group_cols = ["product", "surface_class", "lat_min", "lat_max", "lat_mid"]

    rows = []

    for keys, g in df.groupby(group_cols):
        product, surface_class, lat_min, lat_max, lat_mid = keys

        n_valid_total = int(g["n_valid"].sum())
        n_wet_total = int(g["n_wet"].sum())
        n_granules = int(g["key"].nunique()) if "key" in g.columns else int(len(g))

        if method == "pooled":
            # -------------------------------------------------------------
            # Pixel-weighted / pooled-pixel aggregation
            # -------------------------------------------------------------
            if n_valid_total > 0:
                mean_precip = np.average(
                    g["mean_precip_mmhr"],
                    weights=g["n_valid"]
                )
                wet_fraction = 100.0 * n_wet_total / n_valid_total
            else:
                mean_precip = np.nan
                wet_fraction = np.nan

            g_wet = g[g["n_wet"] > 0].copy()

            if not g_wet.empty and g_wet["n_wet"].sum() > 0:
                wet_mean_precip = np.average(
                    g_wet["wet_mean_precip_mmhr"],
                    weights=g_wet["n_wet"]
                )
            else:
                wet_mean_precip = np.nan

        elif method == "granule_mean":
            # -------------------------------------------------------------
            # Granule-weighted aggregation
            # Each granule/bin receives equal weight.
            # -------------------------------------------------------------
            mean_precip = g["mean_precip_mmhr"].mean()
            wet_fraction = g["wet_fraction_percent"].mean()

            g_wet = g[np.isfinite(g["wet_mean_precip_mmhr"])].copy()

            if not g_wet.empty:
                wet_mean_precip = g_wet["wet_mean_precip_mmhr"].mean()
            else:
                wet_mean_precip = np.nan

        rows.append({
            "aggregation_method": method,
            "product": product,
            "surface_class": surface_class,
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lat_mid": lat_mid,
            "n_granules_contributing": n_granules,
            "n_valid": n_valid_total,
            "n_wet": n_wet_total,
            "mean_precip_mmhr": float(mean_precip) if np.isfinite(mean_precip) else np.nan,
            "wet_fraction_percent": float(wet_fraction) if np.isfinite(wet_fraction) else np.nan,
            "wet_mean_precip_mmhr": float(wet_mean_precip) if np.isfinite(wet_mean_precip) else np.nan,
            "wet_threshold_mmhr": WET_THRESHOLD,
        })

    return pd.DataFrame(rows)


def aggregate_pdfs(pdf_all, method="pooled"):
    """
    Aggregate PDF counts and volumes across many matched orbit pairs.

    Parameters
    ----------
    pdf_all : DataFrame
        Per-orbit PDF statistics.
    method : str
        "pooled":
            Pool counts and volumes first, then compute PDFc/PDFv.

        "granule_mean":
            Average the per-granule PDFc/PDFv values by bin.
            Each granule receives equal weight.

    Returns
    -------
    DataFrame
    """
    if method not in ["pooled", "granule_mean"]:
        raise ValueError("method must be 'pooled' or 'granule_mean'")

    df = pdf_all.copy()

    group_cols = ["product", "hemisphere", "surface_class", "bin_low", "bin_high", "bin_label"]

    if method == "pooled":
        g = df.groupby(group_cols, as_index=False).agg({
            "count": "sum",
            "volume": "sum",
        })

        result_list = []

        for (product, hemisphere, surface_class), gg in g.groupby(["product", "hemisphere", "surface_class"]):
            gg = gg.copy()

            total_count = gg["count"].sum()
            total_volume = gg["volume"].sum()

            gg["pdfc"] = 100.0 * gg["count"] / total_count if total_count > 0 else np.nan
            gg["pdfv"] = 100.0 * gg["volume"] / total_volume if total_volume > 0 else np.nan
            gg["aggregation_method"] = method
            gg["wet_threshold_mmhr"] = WET_THRESHOLD

            result_list.append(gg)

        return pd.concat(result_list, ignore_index=True)

    elif method == "granule_mean":
        g = df.groupby(group_cols, as_index=False).agg({
            "pdfc": "mean",
            "pdfv": "mean",
            "count": "sum",
            "volume": "sum",
        })

        g["aggregation_method"] = method
        g["wet_threshold_mmhr"] = WET_THRESHOLD
        

        return g

# =============================================================================
# Main
# =============================================================================

def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    print("Finding matched V7/V8 GPROF orbit files...")
    pairs = find_matched_pairs(V7_DIR, V8_DIR)

    print(f"Total matched pairs found: {len(pairs)}")

    if MAX_PAIRS is not None:
        pairs = pairs[:MAX_PAIRS]
        print(f"Processing first {len(pairs)} matched pairs because MAX_PAIRS={MAX_PAIRS}")

    if len(pairs) == 0:
        raise RuntimeError("No matched V7/V8 pairs found. Check V7_DIR and V8_DIR.")

    # Save matched-pair inventory
    inventory_df = pd.DataFrame(pairs)
    inventory_path = os.path.join(OUT_DIR, "matched_v7_v8_pairs_used.csv")
    inventory_df.to_csv(inventory_path, index=False)
    print(f"Saved matched-pair inventory: {inventory_path}")

    # Load land/ocean mask once
    if USE_LAND_OCEAN_MASK:
        print("Loading land/ocean mask...")
        lsm = load_land_ocean_mask(LAND_SEA_MASK_PATH, LAND_SEA_MASK_VAR)
        print("Land/ocean mask loaded.")
    else:
        lsm = None
        print("Land/ocean mask not used.")

    print(f"Processing orbit pairs with N_JOBS={N_JOBS}...")

    results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(process_one_pair)(pair, lsm=lsm)
        for pair in pairs
    )

    summary_list = []
    zonal_list = []
    pdf_list = []
    error_list = []

    for summary_df, zonal_df, pdf_df in results:
        if "error" in summary_df.columns:
            error_list.append(summary_df)
        else:
            summary_list.append(summary_df)

        if zonal_df is not None:
            zonal_list.append(zonal_df)

        if pdf_df is not None:
            pdf_list.append(pdf_df)

    # -------------------------------------------------------------------------
    # Save per-orbit summary
    # -------------------------------------------------------------------------
    if summary_list:
        summary_all = pd.concat(summary_list, ignore_index=True)
        summary_path = os.path.join(OUT_DIR, "per_orbit_summary.csv")
        summary_all.to_csv(summary_path, index=False)
        print(f"Saved: {summary_path}")
    else:
        summary_all = pd.DataFrame()

    # -------------------------------------------------------------------------
    # Save per-orbit zonal stats and aggregated zonal stats
    # -------------------------------------------------------------------------
    if zonal_list:
        zonal_all = pd.concat(zonal_list, ignore_index=True)

        zonal_path = os.path.join(OUT_DIR, "per_orbit_zonal_stats.csv")
        zonal_all.to_csv(zonal_path, index=False)
        print(f"Saved: {zonal_path}")

        zonal_agg_list = []

        if AGGREGATION_METHOD in ["pooled", "both"]:
            zonal_agg_pooled = aggregate_zonal_stats(zonal_all, method="pooled")
            zonal_agg_pooled_path = os.path.join(OUT_DIR, "aggregated_zonal_stats_pooled.csv")
            zonal_agg_pooled.to_csv(zonal_agg_pooled_path, index=False)
            print(f"Saved: {zonal_agg_pooled_path}")
            zonal_agg_list.append(zonal_agg_pooled)

        if AGGREGATION_METHOD in ["granule_mean", "both"]:
            zonal_agg_granule = aggregate_zonal_stats(zonal_all, method="granule_mean")
            zonal_agg_granule_path = os.path.join(OUT_DIR, "aggregated_zonal_stats_granule_mean.csv")
            zonal_agg_granule.to_csv(zonal_agg_granule_path, index=False)
            print(f"Saved: {zonal_agg_granule_path}")
            zonal_agg_list.append(zonal_agg_granule)

        if zonal_agg_list:
            zonal_agg = pd.concat(zonal_agg_list, ignore_index=True)
            zonal_agg_path = os.path.join(OUT_DIR, f"aggregated_zonal_stats_{sve_fle_part}.csv")
            zonal_agg.to_csv(zonal_agg_path, index=False)
            print(f"Saved combined aggregation file: {zonal_agg_path}")

            zonal_agg_stable_path = os.path.join(OUT_DIR, "aggregated_zonal_stats.csv")
            zonal_agg.to_csv(zonal_agg_stable_path, index=False)
            print(f"Saved stable combined aggregation file: {zonal_agg_stable_path}")
        else:
            zonal_agg = pd.DataFrame()

    else:
        zonal_all = pd.DataFrame()
        zonal_agg = pd.DataFrame()

    # -------------------------------------------------------------------------
    # Save per-orbit PDFs and aggregated PDFs
    # -------------------------------------------------------------------------
    if pdf_list:
        pdf_all = pd.concat(pdf_list, ignore_index=True)
        pdf_path = os.path.join(OUT_DIR, "per_orbit_pdf_stats.csv")
        pdf_all.to_csv(pdf_path, index=False)
        print(f"Saved: {pdf_path}")

        pdf_agg_list = []

        if AGGREGATION_METHOD in ["pooled", "both"]:
            pdf_agg_pooled = aggregate_pdfs(pdf_all, method="pooled")
            pdf_agg_pooled_path = os.path.join(OUT_DIR, "aggregated_pdf_stats_pooled.csv")
            pdf_agg_pooled.to_csv(pdf_agg_pooled_path, index=False)
            print(f"Saved: {pdf_agg_pooled_path}")
            pdf_agg_list.append(pdf_agg_pooled)

        if AGGREGATION_METHOD in ["granule_mean", "both"]:
            pdf_agg_granule = aggregate_pdfs(pdf_all, method="granule_mean")
            pdf_agg_granule_path = os.path.join(OUT_DIR, "aggregated_pdf_stats_granule_mean.csv")
            pdf_agg_granule.to_csv(pdf_agg_granule_path, index=False)
            print(f"Saved: {pdf_agg_granule_path}")
            pdf_agg_list.append(pdf_agg_granule)

        if pdf_agg_list:
            pdf_agg = pd.concat(pdf_agg_list, ignore_index=True)
            pdf_agg_path = os.path.join(OUT_DIR, f"aggregated_pdf_stats_{sve_fle_part}.csv")
            pdf_agg.to_csv(pdf_agg_path, index=False)
            print(f"Saved combined aggregation file: {pdf_agg_path}")

            pdf_agg_stable_path = os.path.join(OUT_DIR, "aggregated_pdf_stats.csv")
            pdf_agg.to_csv(pdf_agg_stable_path, index=False)
            print(f"Saved stable combined aggregation file: {pdf_agg_stable_path}")
        else:
            pdf_agg = pd.DataFrame()
    else:
        pdf_all = pd.DataFrame()
        pdf_agg = pd.DataFrame()

    # -------------------------------------------------------------------------
    # Save errors
    # -------------------------------------------------------------------------
    if error_list:
        errors_all = pd.concat(error_list, ignore_index=True)
        error_path = os.path.join(OUT_DIR, f"processing_errors_{sve_fle_part}.csv")
        errors_all.to_csv(error_path, index=False)
        print(f"Saved errors: {error_path}")
    else:
        print("No processing errors.")

    print("Done.")


if __name__ == "__main__":
    main()
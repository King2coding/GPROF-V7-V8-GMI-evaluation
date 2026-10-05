# %%
"""
Interactive QC for collocated GMI GPROF V7/V8 + ERA5 + MERRA2 + AutoSnow files.

This file is designed for VS Code interactive use.
Run one # %% cell at a time so you can inspect outputs and ask questions.

Main goals:
1. List currently available collocated files safely while production may still be running.
2. Open one file and inspect its structure.
3. QC a small sample.
4. QC all currently safe files.
5. Build QC summaries:
   - common-valid V7/V8/ERA5 comparison
   - V7/V8 retrieval-coverage diagnostics
   - V8-only retrieval diagnostics
   - monthly count tables
6. Save dated QC outputs.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl


# %%
# -----------------------------
# 1. Project paths and settings
# -----------------------------

PROJECT_DIR = Path("/home/kkumah/Projects/Comparing_V7_V8_GPROF-GMI_data")
COLLOCATED_DIR = Path("/scratch/kkumah/Collocated_GMI_ERA5_MERRA_Autosnow")

QC_TABLE_DIR = PROJECT_DIR / "QC_outputs" / "tables"
QC_FIGURE_DIR = PROJECT_DIR / "QC_outputs" / "figures"
QC_LOG_DIR = PROJECT_DIR / "QC_outputs" / "logs"

QC_TABLE_DIR.mkdir(parents=True, exist_ok=True)
QC_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
QC_LOG_DIR.mkdir(parents=True, exist_ok=True)

RUN_DATE = datetime.now(timezone.utc).strftime("%Y%m%d")

# GMI is mm/hr. ERA5 is currently stored as mm for matched hourly field.
WET_THRESHOLD = 0.1

# Interactive testing size
MAX_FILES = 20

# Safety while collocation production is still running
SKIP_LAST_FILE = True
MIN_FILE_AGE_SECONDS = 120

print("Project directory:", PROJECT_DIR)
print("Collocated input directory:", COLLOCATED_DIR)
print("QC table output directory:", QC_TABLE_DIR)
print("QC figure output directory:", QC_FIGURE_DIR)
print("Run date:", RUN_DATE)

# -----------------------------
# Consistent product colors
# -----------------------------

COLOR_V7 = "#0000FF"      # blue
COLOR_V8 = "#FF0000"      # red
COLOR_ERA5 = "#2CA02C"    # green
COLOR_IMERG = "#FF7F0E"   # orange, if used later
COLOR_BOTH = "#000000"    # black / neutral diagnostic
COLOR_V8_ONLY = COLOR_V8
COLOR_V7_ONLY = COLOR_V7

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['DejaVu Serif', 'Times', 'serif']
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12


# %%
# -----------------------------
# 2. List available files safely
# -----------------------------

def list_collocated_files(
    input_dir: Path,
    skip_last_file: bool = True,
    min_file_age_seconds: int = 120,
) -> list[Path]:
    """
    List collocated NetCDF files safely while production may still be running.

    Safety logic:
    1. Sort files by modification time.
    2. Exclude files modified very recently.
    3. Optionally skip the newest remaining file.

    This reduces the chance of reading a file that is still being written.
    """
    files = sorted(
        input_dir.glob("GMI_GPROF_V7_V8_ERA5_MERRA2_AutoSnow_*.nc"),
        key=lambda p: p.stat().st_mtime,
    )

    now = datetime.now().timestamp()

    safe_files = []
    skipped_recent = []

    for f in files:
        age = now - f.stat().st_mtime
        if age < min_file_age_seconds:
            skipped_recent.append(f)
        else:
            safe_files.append(f)

    skipped_last = None
    if skip_last_file and len(safe_files) > 0:
        skipped_last = safe_files[-1]
        safe_files = safe_files[:-1]

    safe_files = sorted(safe_files)

    print("Total .nc files found:", len(files))
    print("Skipped recently modified files:", len(skipped_recent))
    if skipped_last is not None:
        print("Skipped newest safe file as extra precaution:", skipped_last.name)
    print("Safe files available for QC:", len(safe_files))

    return safe_files


all_files = list_collocated_files(
    COLLOCATED_DIR,
    skip_last_file=SKIP_LAST_FILE,
    min_file_age_seconds=MIN_FILE_AGE_SECONDS,
)

for f in all_files[:5]:
    print(f.name)

if len(all_files) > 5:
    print("...")


# %%
# -----------------------------
# 3. Open one sample file
# -----------------------------

sample_file = all_files[0]
print("Sample file:", sample_file)

ds = xr.open_dataset(sample_file)
ds


# %%
# -----------------------------
# 4. Inspect sample file
# -----------------------------

print("Dimensions:")
print(ds.sizes)

print("\nVariables:")
for v in ds.variables:
    print(v, ds[v].dims, ds[v].shape, ds[v].dtype)

print("\nGlobal attributes:")
for k, v in ds.attrs.items():
    print(f"{k}: {v}")


# %%
# -----------------------------
# 5a. Quick validity check for sample file
# -----------------------------

required_vars = [
    "lat",
    "lon",
    "gmi_v7_surface_precipitation",
    "gmi_v8_surface_precipitation",
    "era5_tp",
    "merra2_t2m",
    "merra2_ts",
    "autosnow_class",
    "gmi_scan_time",
    "gmi_scan_year",
    "gmi_scan_doy",
    "gmi_scan_hour",
]

missing = [v for v in required_vars if v not in ds.variables]
print("Missing required variables:", missing)

lat = ds["lat"].values
lon = ds["lon"].values
v7 = ds["gmi_v7_surface_precipitation"].values
v8 = ds["gmi_v8_surface_precipitation"].values
era5 = ds["era5_tp"].values
t2m = ds["merra2_t2m"].values
ts = ds["merra2_ts"].values
autosnow = ds["autosnow_class"].values

valid_latlon = np.isfinite(lat) & np.isfinite(lon)
valid_v7 = np.isfinite(v7) & (v7 >= 0)
valid_v8 = np.isfinite(v8) & (v8 >= 0)
valid_era5 = np.isfinite(era5) & (era5 >= 0)
valid_t2m = np.isfinite(t2m)
valid_ts = np.isfinite(ts)
valid_autosnow = autosnow != 255

valid_full = (
    valid_latlon
    & valid_v7
    & valid_v8
    & valid_era5
    & valid_t2m
    & valid_ts
    & valid_autosnow
)

print("Total footprints:", lat.size)
print("Valid lat/lon:", valid_latlon.sum())
print("Valid V7:", valid_v7.sum())
print("Valid V8:", valid_v8.sum())
print("Valid ERA5:", valid_era5.sum())
print("Valid MERRA2 T2M:", valid_t2m.sum())
print("Valid MERRA2 TS:", valid_ts.sum())
print("Valid AutoSnow:", valid_autosnow.sum())
print("Valid full comparison:", valid_full.sum())
print("Fraction full comparison:", valid_full.sum() / lat.size)
print("AutoSnow classes present:", np.unique(autosnow[autosnow != 255]))

ds.close()

# %%
# -----------------------------
# 5b. Inspect precipitation value ranges and possible fill values
# -----------------------------

for name, arr in {
    "V7": v7,
    "V8": v8,
    "ERA5": era5,
}.items():
    finite = np.isfinite(arr)
    print(f"\n{name}")
    print("  total:", arr.size)
    print("  finite:", finite.sum())
    print("  nan:", np.isnan(arr).sum())
    if finite.sum() > 0:
        vals = arr[finite]
        print("  min:", np.nanmin(vals))
        print("  p0.1, p1, p5, p50, p95, p99, p99.9:", np.nanpercentile(vals, [0.1, 1, 5, 50, 95, 99, 99.9]))
        print("  max:", np.nanmax(vals))
        print("  count < 0:", np.sum(vals < 0))
        print("  unique negative values:", np.unique(vals[vals < 0])[:20])
        print("  count == 0:", np.sum(vals == 0))

# %%
# -----------------------------
# 5c. Inspect V8-only pixels in one sample file
# -----------------------------

valid_v7 = np.isfinite(v7) & (v7 >= 0)
valid_v8 = np.isfinite(v8) & (v8 >= 0)

valid_both_gmi = valid_v7 & valid_v8
valid_v8_only = valid_v8 & ~valid_v7
valid_v7_only = valid_v7 & ~valid_v8

print("Total footprints:", v7.size)
print("Both valid:", valid_both_gmi.sum())
print("V8 only:", valid_v8_only.sum())
print("V7 only:", valid_v7_only.sum())

print("\nV7 values where V8 is valid but V7 is missing/invalid:")
v7_at_v8_only = v7[valid_v8_only]
print("  finite count:", np.isfinite(v7_at_v8_only).sum())
print("  nan count:", np.isnan(v7_at_v8_only).sum())
if np.isfinite(v7_at_v8_only).sum() > 0:
    print("  unique finite values:", np.unique(v7_at_v8_only[np.isfinite(v7_at_v8_only)])[:50])

print("\nV8 values where V8-only:")
v8_at_v8_only = v8[valid_v8_only]
print("  min:", np.nanmin(v8_at_v8_only))
print("  median:", np.nanmedian(v8_at_v8_only))
print("  mean:", np.nanmean(v8_at_v8_only))
print("  max:", np.nanmax(v8_at_v8_only))
print("  fraction zero:", np.mean(v8_at_v8_only == 0))
print("  fraction > 0:", np.mean(v8_at_v8_only > 0))
print("  fraction > wet threshold:", np.mean(v8_at_v8_only > WET_THRESHOLD))

# %%
# -----------------------------
# 5d. V8-only pixels by AutoSnow class in sample file
# -----------------------------

valid_autosnow = autosnow != 255

classes = sorted([int(c) for c in np.unique(autosnow[valid_autosnow])])

for c in classes:
    m = autosnow == c
    total_c = m.sum()
    both_c = (valid_both_gmi & m).sum()
    v8_only_c = (valid_v8_only & m).sum()
    v7_only_c = (valid_v7_only & m).sum()

    print(
        f"AutoSnow class {c}: "
        f"total={total_c}, "
        f"both={both_c}, "
        f"v8_only={v8_only_c}, "
        f"v7_only={v7_only_c}, "
        f"frac_v8_only={v8_only_c / total_c if total_c > 0 else np.nan:.4f}"
    )

# %%
# -----------------------------
# 5e. Temperature distribution for V8-only vs both-valid pixels
# -----------------------------

for label, mask in {
    "both_valid": valid_both_gmi & valid_t2m,
    "v8_only": valid_v8_only & valid_t2m,
    "v7_only": valid_v7_only & valid_t2m,
}.items():
    vals = t2m[mask]
    print(f"\n{label}")
    print("  n:", vals.size)
    if vals.size > 0:
        print("  T2M percentiles:", np.nanpercentile(vals, [1, 5, 25, 50, 75, 95, 99]))
# %%
# -----------------------------
# 6. Helper functions
# -----------------------------

REQUIRED_VARS = [
    "lat",
    "lon",
    "gmi_v7_surface_precipitation",
    "gmi_v8_surface_precipitation",
    "era5_tp",
    "merra2_t2m",
    "merra2_ts",
    "autosnow_class",
    "gmi_scan_time",
    "gmi_scan_year",
    "gmi_scan_doy",
    "gmi_scan_hour",
]


def parse_file_metadata(path: Path) -> dict:
    """
    Parse orbit/date information from filename.

    Expected:
    GMI_GPROF_V7_V8_ERA5_MERRA2_AutoSnow_000079_20140304_20140304.nc
    """
    pattern = (
        r"GMI_GPROF_V7_V8_ERA5_MERRA2_AutoSnow_"
        r"(?P<orbit>\d+)_"
        r"(?P<date1>\d{8})_"
        r"(?P<date2>\d{8})\.nc"
    )

    match = re.match(pattern, path.name)

    if match is None:
        return {
            "orbit_number": None,
            "start_date": None,
            "end_date": None,
            "year": None,
            "month": None,
            "day": None,
        }

    date1 = match.group("date1")

    return {
        "orbit_number": match.group("orbit"),
        "start_date": date1,
        "end_date": match.group("date2"),
        "year": int(date1[:4]),
        "month": int(date1[4:6]),
        "day": int(date1[6:8]),
    }


def finite_mean(values: np.ndarray) -> float:
    """Mean over finite values only."""
    finite = np.isfinite(values)
    if finite.sum() == 0:
        return np.nan
    return float(np.nanmean(values[finite]))


def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    """Mean over values where mask is True and values are finite."""
    selected = values[mask]
    finite = np.isfinite(selected)
    if finite.sum() == 0:
        return np.nan
    return float(np.nanmean(selected[finite]))


def wet_fraction_common(values: np.ndarray, valid_full: np.ndarray, threshold: float) -> float:
    """Wet fraction using only the common-valid comparison mask."""
    selected = values[valid_full]
    if selected.size == 0:
        return np.nan
    return float((selected > threshold).mean())


def retrieval_value_stats(values: np.ndarray, wet_threshold: float) -> dict:
    """Stats for retrieval values in a selected mask."""
    if values.size == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
            "frac_zero": np.nan,
            "frac_positive": np.nan,
            "frac_wet": np.nan,
        }

    return {
        "mean": float(np.nanmean(values)),
        "median": float(np.nanmedian(values)),
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
        "frac_zero": float(np.mean(values == 0)),
        "frac_positive": float(np.mean(values > 0)),
        "frac_wet": float(np.mean(values > wet_threshold)),
    }


def class_string(values: np.ndarray) -> str:
    """Comma-separated AutoSnow class values, excluding 255."""
    vals = np.unique(values)
    vals = [int(v) for v in vals if int(v) != 255]
    return ",".join(map(str, vals))


def qc_one_file(path: Path, wet_threshold: float = 0.1) -> dict:
    """
    QC one collocated NetCDF file.

    Main comparison uses common-valid footprints:
    lat/lon, V7, V8, ERA5, MERRA2 T2M, MERRA2 TS, and AutoSnow all valid.

    Returns one dictionary row.
    """
    meta = parse_file_metadata(path)

    row = {
        "filename": path.name,
        "path": str(path),
        "file_size_mb": path.stat().st_size / 1024**2,
        **meta,
        "status": "unknown",
        "opens_with_xarray": False,
        "required_variables_present": False,
        "missing_required_variables": "",
        "error_message": "",
    }

    try:
        ds = xr.open_dataset(path)
        row["opens_with_xarray"] = True
    except Exception as exc:
        row["status"] = "failed_open"
        row["error_message"] = repr(exc)
        return row

    try:
        missing = [v for v in REQUIRED_VARS if v not in ds.variables]
        if missing:
            row["status"] = "missing_vars"
            row["missing_required_variables"] = ",".join(missing)
            return row

        row["required_variables_present"] = True

        lat = ds["lat"].values
        lon = ds["lon"].values
        v7 = ds["gmi_v7_surface_precipitation"].values
        v8 = ds["gmi_v8_surface_precipitation"].values
        era5 = ds["era5_tp"].values
        t2m = ds["merra2_t2m"].values
        ts = ds["merra2_ts"].values
        autosnow = ds["autosnow_class"].values

        n_scan = int(ds.sizes.get("scan", 0))
        n_pixel = int(ds.sizes.get("pixel", 0))
        n_total = int(lat.size)

        valid_latlon = np.isfinite(lat) & np.isfinite(lon)
        valid_v7 = np.isfinite(v7) & (v7 >= 0)
        valid_v8 = np.isfinite(v8) & (v8 >= 0)
        valid_era5 = np.isfinite(era5) & (era5 >= 0)
        valid_t2m = np.isfinite(t2m)
        valid_ts = np.isfinite(ts)
        valid_autosnow = autosnow != 255

        valid_full = (
            valid_latlon
            & valid_v7
            & valid_v8
            & valid_era5
            & valid_t2m
            & valid_ts
            & valid_autosnow
        )

        valid_both_gmi = valid_v7 & valid_v8
        valid_v7_only = valid_v7 & ~valid_v8
        valid_v8_only = valid_v8 & ~valid_v7

        v8_only_stats = retrieval_value_stats(v8[valid_v8_only], wet_threshold)
        v7_only_stats = retrieval_value_stats(v7[valid_v7_only], wet_threshold)

        v8_common_stats = retrieval_value_stats(v8[valid_full], wet_threshold)
        v7_common_stats = retrieval_value_stats(v7[valid_full], wet_threshold)
        era5_common_stats = retrieval_value_stats(era5[valid_full], wet_threshold)

        row.update(
            {
                "status": "ok",
                "n_scan": n_scan,
                "n_pixel": n_pixel,
                "n_total_footprints": n_total,

                "lat_min": float(np.nanmin(lat)),
                "lat_max": float(np.nanmax(lat)),
                "lon_min": float(np.nanmin(lon)),
                "lon_max": float(np.nanmax(lon)),

                # Product-native valid counts
                "n_valid_latlon": int(valid_latlon.sum()),
                "n_valid_v7": int(valid_v7.sum()),
                "n_valid_v8": int(valid_v8.sum()),
                "n_valid_era5": int(valid_era5.sum()),
                "n_valid_merra2_t2m": int(valid_t2m.sum()),
                "n_valid_merra2_ts": int(valid_ts.sum()),
                "n_valid_autosnow": int(valid_autosnow.sum()),

                # Common-valid comparison count
                "n_valid_full_comparison": int(valid_full.sum()),
                "frac_valid_full_comparison": float(valid_full.sum() / n_total),

                # Product-native means, diagnostic only
                "mean_v7_all_valid": finite_mean(v7),
                "mean_v8_all_valid": finite_mean(v8),
                "mean_era5_all_valid": finite_mean(era5),
                "mean_t2m_all_valid": finite_mean(t2m),
                "mean_ts_all_valid": finite_mean(ts),

                # Main common-valid comparison metrics
                "mean_v7_full_comparison": masked_mean(v7, valid_full),
                "mean_v8_full_comparison": masked_mean(v8, valid_full),
                "mean_era5_full_comparison": masked_mean(era5, valid_full),

                "median_v7_full_comparison": v7_common_stats["median"],
                "median_v8_full_comparison": v8_common_stats["median"],
                "median_era5_full_comparison": era5_common_stats["median"],

                "wet_fraction_v7_full_comparison": wet_fraction_common(v7, valid_full, wet_threshold),
                "wet_fraction_v8_full_comparison": wet_fraction_common(v8, valid_full, wet_threshold),
                "wet_fraction_era5_full_comparison": wet_fraction_common(era5, valid_full, wet_threshold),

                # AutoSnow diagnostic
                "autosnow_classes_present": class_string(autosnow),

                # V7/V8 retrieval-coverage diagnostics
                "n_valid_both_gmi": int(valid_both_gmi.sum()),
                "n_valid_v7_only": int(valid_v7_only.sum()),
                "n_valid_v8_only": int(valid_v8_only.sum()),

                "frac_valid_both_gmi": float(valid_both_gmi.sum() / n_total),
                "frac_valid_v7_only": float(valid_v7_only.sum() / n_total),
                "frac_valid_v8_only": float(valid_v8_only.sum() / n_total),

                # V8-only retrieval-value diagnostics
                "mean_v8_only": v8_only_stats["mean"],
                "median_v8_only": v8_only_stats["median"],
                "min_v8_only": v8_only_stats["min"],
                "max_v8_only": v8_only_stats["max"],
                "frac_v8_only_zero": v8_only_stats["frac_zero"],
                "frac_v8_only_positive": v8_only_stats["frac_positive"],
                "frac_v8_only_wet": v8_only_stats["frac_wet"],

                # V7-only retrieval-value diagnostics
                "mean_v7_only": v7_only_stats["mean"],
                "median_v7_only": v7_only_stats["median"],
                "min_v7_only": v7_only_stats["min"],
                "max_v7_only": v7_only_stats["max"],
                "frac_v7_only_zero": v7_only_stats["frac_zero"],
                "frac_v7_only_positive": v7_only_stats["frac_positive"],
                "frac_v7_only_wet": v7_only_stats["frac_wet"],
            }
        )

    except Exception as exc:
        row["status"] = "failed_qc"
        row["error_message"] = repr(exc)

    finally:
        ds.close()

    return row

# %%
# -----------------------------
# 7. QC a small sample
# -----------------------------

files_to_scan = all_files[:MAX_FILES]

rows = []
for i, f in enumerate(files_to_scan, start=1):
    print(f"[{i}/{len(files_to_scan)}] {f.name}")
    rows.append(qc_one_file(f, wet_threshold=WET_THRESHOLD))

qc_df = pd.DataFrame(rows)

print("QC dataframe shape:", qc_df.shape)
qc_df.head()

# %%
# -----------------------------
# 8. Inspect small-sample QC results
# -----------------------------

display_cols = [
    "filename",
    "status",
    "year",
    "month",
    "day",
    "n_total_footprints",
    "n_valid_v7",
    "n_valid_v8",
    "n_valid_both_gmi",
    "n_valid_v7_only",
    "n_valid_v8_only",
    "n_valid_full_comparison",
    "frac_valid_full_comparison",
    "mean_v7_full_comparison",
    "mean_v8_full_comparison",
    "mean_era5_full_comparison",
    "wet_fraction_v7_full_comparison",
    "wet_fraction_v8_full_comparison",
    "wet_fraction_era5_full_comparison",
    "mean_v8_only",
    "median_v8_only",
    "frac_v8_only_positive",
    "frac_v8_only_wet",
    "autosnow_classes_present",
]

qc_df[display_cols].head(20)

# %%
# -----------------------------
# 9. Small-sample sanity checks
# -----------------------------

print(qc_df["status"].value_counts(dropna=False))

print(
    "n_valid_full_comparison <= n_valid_both_gmi:",
    (qc_df["n_valid_full_comparison"] <= qc_df["n_valid_both_gmi"]).all(),
)

qc_df[
    [
        "frac_valid_full_comparison",
        "mean_v7_full_comparison",
        "mean_v8_full_comparison",
        "mean_era5_full_comparison",
        "wet_fraction_v7_full_comparison",
        "wet_fraction_v8_full_comparison",
        "wet_fraction_era5_full_comparison",
        "frac_valid_both_gmi",
        "frac_valid_v7_only",
        "frac_valid_v8_only",
        "mean_v8_only",
        "median_v8_only",
        "frac_v8_only_positive",
        "frac_v8_only_wet",
    ]
].describe()

# %%
# -----------------------------
# 10. Run QC on all currently safe files
# -----------------------------

RUN_ALL = True

if RUN_ALL:
    rows_all = []

    for i, f in enumerate(all_files, start=1):
        if i == 1 or i % 100 == 0 or i == len(all_files):
            print(f"[{i}/{len(all_files)}] {f.name}", flush=True)

        rows_all.append(qc_one_file(f, wet_threshold=WET_THRESHOLD))

    qc_all_df = pd.DataFrame(rows_all)
    print("All-file QC shape:", qc_all_df.shape)

else:
    print("RUN_ALL is False.")

# %%
# -----------------------------
# 11. Full QC quick checks
# -----------------------------

qc = qc_all_df.copy()
ok_all = qc[qc["status"] == "ok"].copy()

print("Number of QC files:", len(qc))

print("\nStatus counts:")
print(qc["status"].value_counts(dropna=False))

print("\nYear/month coverage:")
print(qc[["year", "month"]].value_counts().sort_index())

print(
    "\nCommon-valid mask sanity check:",
    (ok_all["n_valid_full_comparison"] <= ok_all["n_valid_both_gmi"]).all(),
)

# %%
# -----------------------------
# 12. Full QC descriptive summaries
# -----------------------------

print("Common-valid comparison summary:")
common_cols = [
    "frac_valid_full_comparison",
    "mean_v7_full_comparison",
    "mean_v8_full_comparison",
    "mean_era5_full_comparison",
    "wet_fraction_v7_full_comparison",
    "wet_fraction_v8_full_comparison",
    "wet_fraction_era5_full_comparison",
]
ok_all[common_cols].describe()

print("V7/V8 retrieval-coverage diagnostic:")
coverage_cols = [
    "frac_valid_both_gmi",
    "frac_valid_v7_only",
    "frac_valid_v8_only",
    "frac_valid_full_comparison",
]
ok_all[coverage_cols].describe()

print("V8-only retrieval-value diagnostic:")
v8_only_cols = [
    "mean_v8_only",
    "median_v8_only",
    "frac_v8_only_zero",
    "frac_v8_only_positive",
    "frac_v8_only_wet",
]
ok_all[v8_only_cols].describe()

# %%
# -----------------------------
# 13. Mean preliminary differences
# -----------------------------

print("Mean common-valid precipitation differences:")
print("Mean V8 - V7:", (ok_all["mean_v8_full_comparison"] - ok_all["mean_v7_full_comparison"]).mean())
print("Mean V7 - ERA5:", (ok_all["mean_v7_full_comparison"] - ok_all["mean_era5_full_comparison"]).mean())
print("Mean V8 - ERA5:", (ok_all["mean_v8_full_comparison"] - ok_all["mean_era5_full_comparison"]).mean())

print("\nMean common-valid wet-fraction differences:")
print(
    "Wet fraction V8 - V7:",
    (
        ok_all["wet_fraction_v8_full_comparison"]
        - ok_all["wet_fraction_v7_full_comparison"]
    ).mean(),
)
print(
    "Wet fraction V7 - ERA5:",
    (
        ok_all["wet_fraction_v7_full_comparison"]
        - ok_all["wet_fraction_era5_full_comparison"]
    ).mean(),
)
print(
    "Wet fraction V8 - ERA5:",
    (
        ok_all["wet_fraction_v8_full_comparison"]
        - ok_all["wet_fraction_era5_full_comparison"]
    ).mean(),
)

# %%
# -----------------------------
# 14. Monthly summary: common-valid comparison
# -----------------------------

coverage_ym = (
    ok_all.groupby(["year", "month"], dropna=False)
    .agg(
        n_files=("filename", "count"),
        mean_frac_valid_full=("frac_valid_full_comparison", "mean"),
        mean_frac_valid_both_gmi=("frac_valid_both_gmi", "mean"),
        mean_frac_v7_only=("frac_valid_v7_only", "mean"),
        mean_frac_v8_only=("frac_valid_v8_only", "mean"),
        mean_v7=("mean_v7_full_comparison", "mean"),
        mean_v8=("mean_v8_full_comparison", "mean"),
        mean_era5=("mean_era5_full_comparison", "mean"),
        wet_fraction_v7=("wet_fraction_v7_full_comparison", "mean"),
        wet_fraction_v8=("wet_fraction_v8_full_comparison", "mean"),
        wet_fraction_era5=("wet_fraction_era5_full_comparison", "mean"),
    )
    .reset_index()
)

coverage_ym["mean_v8_minus_v7"] = coverage_ym["mean_v8"] - coverage_ym["mean_v7"]
coverage_ym["mean_v7_minus_era5"] = coverage_ym["mean_v7"] - coverage_ym["mean_era5"]
coverage_ym["mean_v8_minus_era5"] = coverage_ym["mean_v8"] - coverage_ym["mean_era5"]

coverage_ym["date"] = pd.to_datetime(
    coverage_ym["year"].astype(int).astype(str)
    + "-"
    + coverage_ym["month"].astype(int).astype(str).str.zfill(2)
    + "-01"
)

coverage_ym

# %%
# -----------------------------
# 15a. Monthly summary: actual valid retrieval counts
# -----------------------------

count_ym = (
    ok_all.groupby(["year", "month"], dropna=False)
    .agg(
        n_files=("filename", "count"),
        n_total_footprints=("n_total_footprints", "sum"),
        n_valid_v7=("n_valid_v7", "sum"),
        n_valid_v8=("n_valid_v8", "sum"),
        n_valid_both_gmi=("n_valid_both_gmi", "sum"),
        n_valid_v7_only=("n_valid_v7_only", "sum"),
        n_valid_v8_only=("n_valid_v8_only", "sum"),
        n_valid_full_comparison=("n_valid_full_comparison", "sum"),
    )
    .reset_index()
)

count_ym["frac_valid_v7"] = count_ym["n_valid_v7"] / count_ym["n_total_footprints"]
count_ym["frac_valid_v8"] = count_ym["n_valid_v8"] / count_ym["n_total_footprints"]
count_ym["frac_valid_both_gmi"] = count_ym["n_valid_both_gmi"] / count_ym["n_total_footprints"]
count_ym["frac_valid_v7_only"] = count_ym["n_valid_v7_only"] / count_ym["n_total_footprints"]
count_ym["frac_valid_v8_only"] = count_ym["n_valid_v8_only"] / count_ym["n_total_footprints"]
count_ym["frac_valid_full_comparison"] = (
    count_ym["n_valid_full_comparison"] / count_ym["n_total_footprints"]
)

count_ym["date"] = pd.to_datetime(
    count_ym["year"].astype(int).astype(str)
    + "-"
    + count_ym["month"].astype(int).astype(str).str.zfill(2)
    + "-01"
)

count_ym[
    [
        "year",
        "month",
        "n_files",
        "n_total_footprints",
        "n_valid_v7",
        "n_valid_v8",
        "n_valid_both_gmi",
        "n_valid_v7_only",
        "n_valid_v8_only",
        "n_valid_full_comparison",
        "frac_valid_v7",
        "frac_valid_v8",
        "frac_valid_v8_only",
    ]
]

# %%
# -----------------------------
# 15b. Monthly count balance checks
# -----------------------------

count_ym["v7_balance_error"] = (
    count_ym["n_valid_v7"]
    - count_ym["n_valid_both_gmi"]
    - count_ym["n_valid_v7_only"]
)

count_ym["v8_balance_error"] = (
    count_ym["n_valid_v8"]
    - count_ym["n_valid_both_gmi"]
    - count_ym["n_valid_v8_only"]
)

print("Max absolute V7 balance error:", count_ym["v7_balance_error"].abs().max())
print("Max absolute V8 balance error:", count_ym["v8_balance_error"].abs().max())

count_ym[
    [
        "date",
        "n_valid_v7",
        "n_valid_v8",
        "n_valid_both_gmi",
        "n_valid_v7_only",
        "n_valid_v8_only",
        "v7_balance_error",
        "v8_balance_error",
    ]
].head()

# %%
# -----------------------------
# 16. Monthly summary: V8-only retrieval diagnostics
# -----------------------------

v8_only_ym = (
    ok_all.groupby(["year", "month"], dropna=False)
    .agg(
        n_files=("filename", "count"),
        mean_frac_v8_only=("frac_valid_v8_only", "mean"),
        mean_v8_only=("mean_v8_only", "mean"),
        median_v8_only=("median_v8_only", "mean"),
        mean_frac_v8_only_zero=("frac_v8_only_zero", "mean"),
        mean_frac_v8_only_positive=("frac_v8_only_positive", "mean"),
        mean_frac_v8_only_wet=("frac_v8_only_wet", "mean"),
    )
    .reset_index()
)

v8_only_ym["date"] = pd.to_datetime(
    v8_only_ym["year"].astype(int).astype(str)
    + "-"
    + v8_only_ym["month"].astype(int).astype(str).str.zfill(2)
    + "-01"
)

v8_only_ym

# %%
# -----------------------------
# 17. Plot V7/V8 valid retrieval fractions by month
# -----------------------------

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    coverage_ym["date"],
    coverage_ym["mean_frac_valid_both_gmi"],
    marker="x",
    linestyle="--",
    linewidth=1.5,
    color=COLOR_BOTH,
    label="Both V7 and V8 valid",
)

ax.plot(
    coverage_ym["date"],
    coverage_ym["mean_frac_v8_only"],
    marker="o",
    linewidth=2,
    color=COLOR_V8_ONLY,
    label="V8 valid where V7 is missing",
)

ax.plot(
    coverage_ym["date"],
    coverage_ym["mean_frac_v7_only"],
    marker="o",
    linewidth=2,
    color=COLOR_V7_ONLY,
    label="V7 valid where V8 is missing",
)

ax.set_title("Monthly mean fraction of GMI footprints with valid retrievals")
ax.set_xlabel("Month")
ax.set_ylabel("Mean fraction of GMI swath")
ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()
plt.show()

# %%
# -----------------------------
# 18. Plot actual valid retrieval counts by month
# -----------------------------

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    count_ym["date"],
    count_ym["n_valid_both_gmi"],
    marker="x",
    linestyle="--",
    linewidth=1.5,
    color=COLOR_BOTH,
    label="Both V7 and V8 valid",
)

ax.plot(
    count_ym["date"],
    count_ym["n_valid_v8_only"],
    marker="o",
    linewidth=2,
    color=COLOR_V8_ONLY,
    label="V8 valid where V7 is missing",
)

ax.plot(
    count_ym["date"],
    count_ym["n_valid_v7_only"],
    marker="o",
    linewidth=2,
    color=COLOR_V7_ONLY,
    label="V7 valid where V8 is missing",
)

ax.set_title("Monthly count of version-specific valid GMI retrievals")
ax.set_xlabel("Month")
ax.set_ylabel("Number of GMI footprints")
ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()
plt.show()

# %%
# -----------------------------
# 19. Plot common-valid precipitation means by month
# -----------------------------

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    coverage_ym["date"],
    coverage_ym["mean_v7"],
    marker="o",
    linewidth=2,
    color=COLOR_V7,
    label="GMI V7",
)

ax.plot(
    coverage_ym["date"],
    coverage_ym["mean_v8"],
    marker="o",
    linewidth=2,
    color=COLOR_V8,
    label="GMI V8",
)

ax.plot(
    coverage_ym["date"],
    coverage_ym["mean_era5"],
    marker="o",
    linewidth=2,
    color=COLOR_ERA5,
    label="ERA5",
)

ax.set_title("Common-valid monthly orbit-mean precipitation")
ax.set_xlabel("Month")
ax.set_ylabel("Mean precipitation")
ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()
plt.show()

# %%
# %%
# -----------------------------
# 20. Plot monthly product differences
# -----------------------------

fig, ax = plt.subplots(figsize=(10, 5))

ax.axhline(0, linewidth=1, color="black")

ax.plot(
    coverage_ym["date"],
    coverage_ym["mean_v8_minus_v7"],
    marker="o",
    linewidth=2,
    color=COLOR_V8,
    label="V8 - V7",
)

ax.plot(
    coverage_ym["date"],
    coverage_ym["mean_v7_minus_era5"],
    marker="o",
    linewidth=2,
    color=COLOR_V7,
    label="V7 - ERA5",
)

ax.plot(
    coverage_ym["date"],
    coverage_ym["mean_v8_minus_era5"],
    marker="o",
    linewidth=2,
    color=COLOR_ERA5,
    label="V8 - ERA5",
)

ax.set_title("Monthly common-valid precipitation differences")
ax.set_xlabel("Month")
ax.set_ylabel("Mean difference")
ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()
plt.show()

# %%
# -----------------------------
# 21a. Plot actual valid retrieval counts by product
# -----------------------------

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    count_ym["date"],
    count_ym["n_valid_v7"],
    marker="o",
    linewidth=2,
    color=COLOR_V7,
    label="GMI V7 valid",
)

ax.plot(
    count_ym["date"],
    count_ym["n_valid_v8"],
    marker="o",
    linewidth=2,
    color=COLOR_V8,
    label="GMI V8 valid",
)

ax.plot(
    count_ym["date"],
    count_ym["n_valid_both_gmi"],
    marker="x",
    linestyle="--",
    linewidth=1.5,
    color=COLOR_BOTH,
    label="Both V7 and V8 valid",
)

ax.set_title("Monthly count of GMI footprints with valid retrievals in V7 and V8")
ax.set_xlabel("Month")
ax.set_ylabel("Number of valid GMI footprints")
ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()
plt.show()

# -----------------------------
# 21b. Plot valid retrieval count mismatch by product
# -----------------------------

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    count_ym["date"],
    count_ym["n_valid_v8_only"],
    marker="o",
    linewidth=2,
    color=COLOR_V8,
    label="V8 valid where V7 is missing",
)

ax.plot(
    count_ym["date"],
    count_ym["n_valid_v7_only"],
    marker="o",
    linewidth=2,
    color=COLOR_V7,
    label="V7 valid where V8 is missing",
)

ax.set_title("Monthly count of GMI footprints with valid retrieval in only one version")
ax.set_xlabel("Month")
ax.set_ylabel("Number of GMI footprints")
ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()
plt.show()

# %%

# %%
# -----------------------------
# 22. Plot valid retrieval fractions by product
# -----------------------------

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    count_ym["date"],
    count_ym["frac_valid_v7"],
    marker="o",
    linewidth=2,
    color=COLOR_V7,
    label="GMI V7 valid",
)

ax.plot(
    count_ym["date"],
    count_ym["frac_valid_v8"],
    marker="o",
    linewidth=2,
    color=COLOR_V8,
    label="GMI V8 valid",
)

# ax.plot(
#     count_ym["date"],
#     count_ym["frac_valid_both_gmi"],
#     marker="x",
#     linestyle="--",
#     linewidth=1.5,
#     color=COLOR_BOTH,
#     label="Both V7 and V8 valid",
# )

ax.set_title("Monthly fraction of GMI footprints with valid retrievals in V7 and V8")
ax.set_xlabel("Month")
ax.set_ylabel("Fraction of GMI swath")
ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()
plt.show()


# %%
# -----------------------------
# 22a. Plot V8-only and V7-only valid retrieval fractions
# -----------------------------

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    count_ym["date"],
    count_ym["frac_valid_v8_only"],
    marker="o",
    linewidth=2,
    color=COLOR_V8,
    label="V8 valid where V7 is missing",
)

ax.plot(
    count_ym["date"],
    count_ym["frac_valid_v7_only"],
    marker="o",
    linewidth=2,
    color=COLOR_V7,
    label="V7 valid where V8 is missing",
)

ax.set_title("Monthly fraction of version-specific valid GMI retrievals")
ax.set_xlabel("Month")
ax.set_ylabel("Fraction of GMI swath")
ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()
plt.show()

# %%
# -----------------------------
# 22b. Plot monthly wet fraction among V8-only retrievals
# -----------------------------

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    v8_only_ym["date"],
    v8_only_ym["mean_frac_v8_only_wet"],
    marker="o",
    linewidth=2,
    color=COLOR_V8,
    label=f"V8-only retrievals > {WET_THRESHOLD} mm/hr",
)

ax.set_title("Monthly wet fraction among V8-only GMI retrievals")
ax.set_xlabel("Month")
ax.set_ylabel("Fraction of V8-only footprints")
ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()
plt.show()


# %%
# -----------------------------
# 23. Save full QC outputs
# -----------------------------

n_scanned = len(qc_all_df)

inventory_full_csv = QC_TABLE_DIR / f"qc_collocated_file_inventory_FULL_{RUN_DATE}_n{n_scanned}.csv"
failed_full_csv = QC_TABLE_DIR / f"qc_collocated_failed_files_FULL_{RUN_DATE}_n{n_scanned}.csv"
coverage_ym_csv = QC_TABLE_DIR / f"qc_monthly_coverage_and_precip_summary_{RUN_DATE}_n{n_scanned}.csv"
count_ym_csv = QC_TABLE_DIR / f"qc_valid_retrieval_counts_by_year_month_{RUN_DATE}_n{n_scanned}.csv"
v8_only_ym_csv = QC_TABLE_DIR / f"qc_v8_only_retrieval_diagnostics_by_year_month_{RUN_DATE}_n{n_scanned}.csv"

qc_all_df.to_csv(inventory_full_csv, index=False)
qc_all_df[qc_all_df["status"] != "ok"].to_csv(failed_full_csv, index=False)
coverage_ym.to_csv(coverage_ym_csv, index=False)
count_ym.to_csv(count_ym_csv, index=False)
v8_only_ym.to_csv(v8_only_ym_csv, index=False)

print("Saved:")
print(inventory_full_csv)
print(failed_full_csv)
print(coverage_ym_csv)
print(count_ym_csv)
print(v8_only_ym_csv)

# %%
# -----------------------------
# 24. Box plot of positive precipitation values
#     using common-valid V7/V8/ERA5 footprints
# -----------------------------

# To avoid loading too much into memory, sample positive values from each file.
# Increase or decrease this depending on speed/memory.
MAX_VALUES_PER_PRODUCT = 2_000_000
RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)

v7_positive_all = []
v8_positive_all = []
era5_positive_all = []

files_for_boxplot = all_files  # or all_files[:1000] for quick testing first

for i, f in enumerate(files_for_boxplot, start=1):
    if i == 1 or i % 500 == 0 or i == len(files_for_boxplot):
        print(f"[{i}/{len(files_for_boxplot)}] {f.name}", flush=True)

    try:
        with xr.open_dataset(f) as ds:
            v7 = ds["gmi_v7_surface_precipitation"].values
            v8 = ds["gmi_v8_surface_precipitation"].values
            era5 = ds["era5_tp"].values

            common_valid = (
                np.isfinite(v7) & (v7 >= 0)
                & np.isfinite(v8) & (v8 >= 0)
                & np.isfinite(era5) & (era5 >= 0)
            )

            # Positive values only, from the same common-valid footprint pool.
            v7_pos = v7[common_valid & (v7 > 0)]
            v8_pos = v8[common_valid & (v8 > 0)]
            era5_pos = era5[common_valid & (era5 > 0)]

            v7_positive_all.append(v7_pos.astype("float32"))
            v8_positive_all.append(v8_pos.astype("float32"))
            era5_positive_all.append(era5_pos.astype("float32"))

    except Exception as exc:
        print(f"Skipping {f.name}: {exc}")

v7_positive_all = np.concatenate(v7_positive_all)
v8_positive_all = np.concatenate(v8_positive_all)
era5_positive_all = np.concatenate(era5_positive_all)

print("Positive common-valid sample sizes:")
print("V7:", v7_positive_all.size)
print("V8:", v8_positive_all.size)
print("ERA5:", era5_positive_all.size)


# %%
# -----------------------------
# Box plot: precipitation values above threshold
# Common-valid V7/V8/ERA5 footprints
# -----------------------------

BOX_MIN_PRECIP = 0.02  # mm/hr
MAX_VALUES_PER_PRODUCT = 10_000_000
RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)

def random_sample(values, max_n, rng):
    """Randomly sample values if array is larger than max_n."""
    values = values[np.isfinite(values)]
    if values.size <= max_n:
        return values
    idx = rng.choice(values.size, size=max_n, replace=False)
    return values[idx]

# Use already-created positive arrays, but threshold them more strictly
v7_box = random_sample(v7_positive_all[v7_positive_all >= BOX_MIN_PRECIP], MAX_VALUES_PER_PRODUCT, rng)
v8_box = random_sample(v8_positive_all[v8_positive_all >= BOX_MIN_PRECIP], MAX_VALUES_PER_PRODUCT, rng)
era5_box = random_sample(era5_positive_all[era5_positive_all >= BOX_MIN_PRECIP], MAX_VALUES_PER_PRODUCT, rng)

def fmt_n(n):
    """Format large sample counts for plot labels."""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}k"
    return str(n)


box_labels = [
    f"GMI V7\nN={fmt_n(v7_box.size)}",
    f"GMI V8\nN={fmt_n(v8_box.size)}",
    f"ERA5\nN={fmt_n(era5_box.size)}",
]

box_data = [v7_box, v8_box, era5_box]
box_labels = ["GMI V7", "GMI V8", "ERA5"]
box_colors = [COLOR_V7, COLOR_V8, COLOR_ERA5]

print(f"Box plot threshold: >= {BOX_MIN_PRECIP} mm/hr")
print("Sample sizes:")
print("V7:", v7_box.size)
print("V8:", v8_box.size)
print("ERA5:", era5_box.size)
fig, ax = plt.subplots(figsize=(8, 5))

bp = ax.boxplot(
    box_data,
    labels=box_labels,
    patch_artist=True,
    showfliers=False,
    widths=0.55,
)

for patch, color in zip(bp["boxes"], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.45)
    patch.set_edgecolor(color)
    patch.set_linewidth(1.5)

for median in bp["medians"]:
    median.set_color("black")
    median.set_linewidth(2)

for whisker in bp["whiskers"]:
    whisker.set_color("black")
    whisker.set_linewidth(1.3)

for cap in bp["caps"]:
    cap.set_color("black")
    cap.set_linewidth(1.3)

ax.set_title(f"Precipitation distribution from subsampled\ncommon-valid footprints ≥ {BOX_MIN_PRECIP} mm/hr")
ax.set_ylabel("Precipitation rate [mm/hr]")
ax.grid(True, axis="y", alpha=0.3)
ax.text(
    0.25,
    0.93,
    f"N = {fmt_n(v7_box.size)} per product",
    transform=ax.transAxes,
    ha="center",
    va="center",
    fontsize=14,
    fontweight="bold",
    color="k",
)
fig.tight_layout()
plt.show()

print(f"Threshold: >= {BOX_MIN_PRECIP} mm/hr")
print("Full counts before sampling:")
print("V7:", np.sum(v7_positive_all >= BOX_MIN_PRECIP))
print("V8:", np.sum(v8_positive_all >= BOX_MIN_PRECIP))
print("ERA5:", np.sum(era5_positive_all >= BOX_MIN_PRECIP))
print("Plotted counts after sampling:")
print("V7:", fmt_n(v7_box.size))
print("V8:", fmt_n(v8_box.size))
print("ERA5:", fmt_n(era5_box.size))

# %%
# -----------------------------
# 25. Log-scale box plot of positive precipitation values
#     using common-valid V7/V8/ERA5 footprints
# -----------------------------

fig, ax = plt.subplots(figsize=(8, 5))

bp = ax.boxplot(
    box_data,
    labels=box_labels,
    patch_artist=True,
    showfliers=False,
    widths=0.55,
)

for patch, color in zip(bp["boxes"], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.45)
    patch.set_edgecolor(color)

for median in bp["medians"]:
    median.set_color("black")
    median.set_linewidth(2)

for whisker in bp["whiskers"]:
    whisker.set_color("black")

for cap in bp["caps"]:
    cap.set_color("black")

ax.set_yscale("log")
ax.set_title("Positive precipitation distribution from common-valid footprints")
ax.set_ylabel("Precipitation, log scale")
ax.grid(True, axis="y", alpha=0.3, which="both")

fig.tight_layout()
plt.show()

# %%
# -----------------------------
# PDF by precipitation volume using existing positive arrays
# -----------------------------

PDF_BINS = np.array(
    [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0],
    dtype=float,
)

PDF_BIN_CENTERS = np.sqrt(PDF_BINS[:-1] * PDF_BINS[1:])
PDF_BIN_CENTERS[0] = 0.05

PDF_BIN_LABELS = [
    "0.1", "0.2", "0.5", "1", "2", "4", "8", "16", "32", "64", "128", "256"
]


def pdf_by_precip_volume(values, bins):
    """
    Compute precipitation-volume PDF [%].

    Each bin is weighted by the sum of precipitation values in that bin.
    """
    values = values[np.isfinite(values) & (values > 0)]

    volume_sum, _ = np.histogram(values, bins=bins, weights=values)

    total_volume = np.nansum(volume_sum)

    if total_volume == 0:
        return np.full(len(bins) - 1, np.nan), volume_sum

    pdf_pct = 100.0 * volume_sum / total_volume

    return pdf_pct, volume_sum


pdf_volume_pct_v7, volume_sum_v7 = pdf_by_precip_volume(v7_positive_all, PDF_BINS)
pdf_volume_pct_v8, volume_sum_v8 = pdf_by_precip_volume(v8_positive_all, PDF_BINS)
pdf_volume_pct_era5, volume_sum_era5 = pdf_by_precip_volume(era5_positive_all, PDF_BINS)

print("PDF by volume sums:")
print("V7:", np.nansum(pdf_volume_pct_v7))
print("V8:", np.nansum(pdf_volume_pct_v8))
print("ERA5:", np.nansum(pdf_volume_pct_era5))

# %%
# -----------------------------
# Plot PDF by precipitation volume
# -----------------------------

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    PDF_BIN_CENTERS,
    pdf_volume_pct_v7,
    marker="o",
    linewidth=2,
    color=COLOR_V7,
    label="GMI V7",
)

ax.plot(
    PDF_BIN_CENTERS,
    pdf_volume_pct_v8,
    marker="o",
    linewidth=2,
    color=COLOR_V8,
    label="GMI V8",
)

ax.plot(
    PDF_BIN_CENTERS,
    pdf_volume_pct_era5,
    marker="o",
    linewidth=2,
    color=COLOR_ERA5,
    label="ERA5",
)

ax.set_xscale("log", base=2)
ax.set_xticks(PDF_BIN_CENTERS)
ax.set_xticklabels(PDF_BIN_LABELS, rotation=45)

ax.set_title("PDF by precipitation volume: common-valid positive footprints")
ax.set_xlabel("Precipitation rate [mm/hr]")
ax.set_ylabel("PDF by volume [%]")
ax.grid(True, alpha=0.3, which="both")
ax.legend()

fig.tight_layout()
plt.show()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_gprof_v7_v8_orbit_diagnostics.py

Plot diagnostics from compare_gprof_v7_v8_orbit.py outputs.

This script does not reread orbital HDF5/NetCDF files.
It reads the CSV outputs already produced by the orbital comparison script.

Expected input files
--------------------
OUT_DIR/
    per_orbit_summary.csv
    aggregated_zonal_stats.csv
    aggregated_pdf_stats.csv

Main outputs
------------
FIG_DIR/
    zonal_mean_precip_v7_v8.png
    zonal_mean_precip_difference.png
    zonal_wet_fraction_v7_v8.png
    zonal_wet_fraction_difference.png
    pdfc_v7_v8.png
    pdfv_v7_v8.png

TABLE_DIR/
    overall_summary_table.csv
    percent_difference_summary_table.csv
    land_ocean_summary_table.csv
"""

# =============================================================================
# Imports
# =============================================================================

import os
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# =============================================================================
# User settings
# =============================================================================

OUT_DIR = "/home/kkumah/Projects/Comparing_V7_V8_GPROF-GMI_data/Outs/orbit_v7_v8_comparison"


FIG_DIR = os.path.join(OUT_DIR, "figures")
TABLE_DIR = os.path.join(OUT_DIR, "tables")

SUMMARY_CSV = os.path.join(OUT_DIR, "per_orbit_summary.csv")
ZONAL_CSV = os.path.join(OUT_DIR, "aggregated_zonal_stats.csv")
PDF_CSV = os.path.join(OUT_DIR, "aggregated_pdf_stats.csv")

# Plot settings
DPI = 300

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['DejaVu Serif', 'Times', 'serif']
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18

# Convert mm/hr to mm/day for plot labels if desired.
# Since GPROF surfacePrecipitation is in mm/hr, multiplying by 24 gives mm/day-equivalent.
CONVERT_TO_MMDAY = False

# Surface classes expected from processing script
SURFACE_CLASSES = ["all", "ocean", "land"]

SURFACE_CLASS_TITLES = {
    "all": "COMBINED",
    "ocean": "OCEAN",
    "land": "LAND",
}

PRODUCT_COLORS = {
    "V7": "blue",
    "V8": "red",
    "V8_minus_V7": "black",
    "IMERG V07": "tab:orange",
    "ERA5": "tab:green",
}

PRODUCT_LINESTYLES = {
    "V7": "-",
    "V8": "--",
    "V8_minus_V7": "-",
    "IMERG V07": "-.",
    "ERA5": ":",
}

GPROF_PRODUCTS = ["V7", "V8"]
REFERENCE_PRODUCTS = ["IMERG V07", "ERA5"]
PLOT_PRODUCTS_WITH_REFS = GPROF_PRODUCTS + REFERENCE_PRODUCTS

# Hemisphere latitude limits
NH_LAT_MIN = 50
NH_LAT_MAX = 90
SH_LAT_MIN = -90
SH_LAT_MAX = -50

# Which aggregation method to plot from combined CSV files:
# "pooled" or "granule_mean"
PLOT_AGGREGATION_METHOD = "granule_mean"

# Use native GPROF rate units for orbital analysis.
# Recommended False for orbital files.
CONVERT_TO_MMDAY = False

MAX_PAIRS = 100000
cde_run_date = date.today().strftime('%Y%m%d')

sve_fle_part = f"{MAX_PAIRS}_{cde_run_date}"

# =============================================================================
# Optional gridded reference diagnostics: IMERG / ERA5
# =============================================================================

ADD_GRIDDED_REFERENCES = True

REFERENCE_DIAG_DIR = (
    "/home/kkumah/Projects/Comparing_V7_V8_GPROF-GMI_data/"
    "Outs/gridded_reference_diagnostics_wet002"
)

REFERENCE_ZONAL_CSV = os.path.join(
    REFERENCE_DIAG_DIR, "reference_zonal_stats.csv"
)

REFERENCE_PDF_CSV = os.path.join(
    REFERENCE_DIAG_DIR, "reference_pdf_stats.csv"
)

REFERENCE_SUMMARY_CSV = os.path.join(
    REFERENCE_DIAG_DIR, "reference_summary_stats.csv"
)

REFERENCE_PRODUCT_LABELS = {
    "IMERG": "IMERG V07",
    "ERA5": "ERA5",
}

# =============================================================================
# Helpers
# =============================================================================

def ensure_dirs():
    Path(FIG_DIR).mkdir(parents=True, exist_ok=True)
    Path(TABLE_DIR).mkdir(parents=True, exist_ok=True)

def aggregate_reference_zonal_years(df):
    """
    Aggregate multi-year IMERG/ERA5 zonal reference diagnostics across years.

    Weighted by n_valid.
    """
    if "year" not in df.columns:
        return df

    group_cols = [
        "product",
        "surface_class",
        "lat_min",
        "lat_max",
        "lat_mid",
        "aggregation_method",
        "source",
    ]

    rows = []

    for keys, g in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))

        n_valid = g["n_valid"].sum()
        n_wet = g["n_wet"].sum()

        row["n_valid"] = int(n_valid)
        row["n_wet"] = int(n_wet)

        row["mean_precip_mmhr"] = (
            np.average(g["mean_precip_mmhr"], weights=g["n_valid"])
            if n_valid > 0 else np.nan
        )

        row["wet_fraction_percent"] = (
            100.0 * n_wet / n_valid
            if n_valid > 0 else np.nan
        )

        if "wet_mean_precip_mmhr" in g.columns and g["n_wet"].sum() > 0:
            valid_wet = g[g["n_wet"] > 0].copy()
            row["wet_mean_precip_mmhr"] = np.average(
                valid_wet["wet_mean_precip_mmhr"],
                weights=valid_wet["n_wet"]
            )
        else:
            row["wet_mean_precip_mmhr"] = np.nan

        row["year"] = "2010_2016_2018"

        if "wet_threshold_mmhr" in g.columns:
            row["wet_threshold_mmhr"] = g["wet_threshold_mmhr"].iloc[0]

        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_reference_pdf_years(df):
    """
    Aggregate multi-year IMERG/ERA5 PDF reference diagnostics across years.

    Counts and volumes are pooled across years.
    """
    if "year" not in df.columns:
        return df

    group_cols = [
        "product",
        "hemisphere",
        "surface_class",
        "bin_low",
        "bin_high",
        "bin_label",
        "aggregation_method",
        "source",
    ]

    g = df.groupby(group_cols, as_index=False, dropna=False).agg({
        "count": "sum",
        "volume": "sum",
    })

    out = []

    for (product, hemisphere, surface_class), sub in g.groupby(
        ["product", "hemisphere", "surface_class"]
    ):
        sub = sub.copy()

        total_count = sub["count"].sum()
        total_volume = sub["volume"].sum()

        sub["pdfc"] = 100.0 * sub["count"] / total_count if total_count > 0 else np.nan
        sub["pdfv"] = 100.0 * sub["volume"] / total_volume if total_volume > 0 else np.nan
        sub["year"] = "2010_2016_2018"

        if "bin_mid" not in sub.columns:
            sub["bin_mid"] = np.where(
                sub["bin_low"] == 0,
                sub["bin_high"],
                np.sqrt(sub["bin_low"] * sub["bin_high"])
            )

        out.append(sub)

    return pd.concat(out, ignore_index=True)

def filter_aggregation_method(df, method):
    """
    Filter combined aggregation CSV to selected aggregation method.
    If the column does not exist, return the DataFrame unchanged for backward compatibility.
    """
    if "aggregation_method" in df.columns:
        return df[df["aggregation_method"] == method].copy()
    return df.copy()

def load_gridded_reference_zonal(reference_zonal_csv=REFERENCE_ZONAL_CSV):
    """
    Load IMERG/ERA5 gridded-reference zonal diagnostics and format them
    to match the GPROF plotting dataframe as closely as possible.
    """
    if not ADD_GRIDDED_REFERENCES:

        return pd.DataFrame()

    if not os.path.exists(reference_zonal_csv):
        print(f"[REF] Zonal reference file not found: {reference_zonal_csv}")
        return pd.DataFrame()

    df = pd.read_csv(reference_zonal_csv)

    # Rename product labels for plotting
    df["product"] = df["product"].replace(REFERENCE_PRODUCT_LABELS)

    # Ensure expected columns exist
    if "wet_fraction_percent" not in df.columns and "wet_fraction" in df.columns:
        df["wet_fraction_percent"] = df["wet_fraction"] * 100.0

    if "mean_precip_mmhr" not in df.columns and "mean_precip" in df.columns:
        df["mean_precip_mmhr"] = df["mean_precip"]

    if "lat_mid" not in df.columns:
        if {"lat_min", "lat_max"}.issubset(df.columns):
            df["lat_mid"] = 0.5 * (df["lat_min"] + df["lat_max"])

    df["source"] = df.get("source", "gridded_reference")

    df = aggregate_reference_zonal_years(df)

    print(f"[REF] Loaded/aggregated zonal reference diagnostics: {df.shape}")

    return df


def load_gridded_reference_pdf(reference_pdf_csv=REFERENCE_PDF_CSV):
    """
    Load IMERG/ERA5 gridded-reference PDF diagnostics and format them
    for GPROF-style PDF plotting.
    """

    if not ADD_GRIDDED_REFERENCES:
        return pd.DataFrame()

    if not os.path.exists(reference_pdf_csv):
        print(f"[REF] PDF reference file not found: {reference_pdf_csv}")
        return pd.DataFrame()

    df = pd.read_csv(reference_pdf_csv)

    df["product"] = df["product"].replace(REFERENCE_PRODUCT_LABELS)

    # New reference code already writes pdfc/pdfv in percent.
    # If older file had fractions, convert only if maximum is <= 1.
    if "pdfc" in df.columns and df["pdfc"].max(skipna=True) <= 1.0:
        df["pdfc"] = df["pdfc"] * 100.0

    if "pdfv" in df.columns and df["pdfv"].max(skipna=True) <= 1.0:
        df["pdfv"] = df["pdfv"] * 100.0

    if "bin_mid" not in df.columns:
        if {"bin_low", "bin_high"}.issubset(df.columns):
            df["bin_mid"] = np.sqrt(df["bin_low"] * df["bin_high"])

    df["source"] = df.get("source", "gridded_reference")

    df = aggregate_reference_pdf_years(df)

    print(f"[REF] Loaded/aggregated PDF reference diagnostics: {df.shape}")

    return df


def load_gridded_reference_summary(reference_summary_csv=REFERENCE_SUMMARY_CSV):
    """
    Load IMERG/ERA5 gridded-reference summary diagnostics.
    """

    if not os.path.exists(reference_summary_csv):
        print(f"[REF] Summary reference file not found: {reference_summary_csv}")
        return pd.DataFrame()

    df = pd.read_csv(reference_summary_csv)

    df["product"] = df["product"].replace(REFERENCE_PRODUCT_LABELS)

    print(f"[REF] Loaded summary reference diagnostics: {df.shape}")

    return df


def load_inputs():
    summary = pd.read_csv(SUMMARY_CSV)
    zonal = pd.read_csv(ZONAL_CSV)
    pdf = pd.read_csv(PDF_CSV)

    zonal = filter_aggregation_method(zonal, PLOT_AGGREGATION_METHOD)
    pdf = filter_aggregation_method(pdf, PLOT_AGGREGATION_METHOD)

    # Make sure GPROF PDF has a hemisphere column for compatibility.
    if "hemisphere" not in pdf.columns:
        pdf["hemisphere"] = "ALL_HEMI"

    if ADD_GRIDDED_REFERENCES:
        ref_zonal = load_gridded_reference_zonal()
        ref_pdf = load_gridded_reference_pdf()

        if not ref_zonal.empty:
            zonal = pd.concat([zonal, ref_zonal], ignore_index=True)

        if not ref_pdf.empty:
            pdf = pd.concat([pdf, ref_pdf], ignore_index=True)

    print("\n[LOAD CHECK]")
    print("Summary:", summary.shape)
    print("Zonal:", zonal.shape)
    print("PDF:", pdf.shape)

    if "product" in zonal.columns:
        print("Zonal products:", sorted(zonal["product"].dropna().unique()))

    if "product" in pdf.columns:
        print("PDF products:", sorted(pdf["product"].dropna().unique()))

    return summary, zonal, pdf


def precip_unit_factor_and_label():
    if CONVERT_TO_MMDAY:
        return 24.0, "mm/day"
    return 1.0, "mm/hr"


def filter_hemi(df, hemi):
    if hemi == "NH":
        return df[(df["lat_mid"] >= NH_LAT_MIN) & (df["lat_mid"] <= NH_LAT_MAX)].copy()
    elif hemi == "SH":
        return df[(df["lat_mid"] >= SH_LAT_MIN) & (df["lat_mid"] <= SH_LAT_MAX)].copy()
    else:
        raise ValueError("hemi must be 'NH' or 'SH'")


def get_class_df(df, product, surface_class, hemi):
    tmp = df[
        (df["product"] == product) &
        (df["surface_class"] == surface_class)
    ].copy()

    tmp = filter_hemi(tmp, hemi)
    tmp = tmp.sort_values("lat_mid")

    return tmp

def style_axes(ax):
    ax.grid(True, alpha=0.35)
    ax.tick_params(axis="both", which="major", labelsize=11)
    ax.minorticks_on()

def robust_xlim_from_values(values, pad_frac=0.12, symmetric=False, min_span=None):
    """
    Compute robust x-axis limits from finite values.

    Parameters
    ----------
    values : array-like
        Values used for x-axis.
    pad_frac : float
        Fractional padding around data range.
    symmetric : bool
        If True, make limits symmetric around zero.
    min_span : float or None
        Minimum x-axis span.

    Returns
    -------
    xmin, xmax
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return None, None

    # Use robust percentiles to avoid one noisy bin stretching the axis
    lo = np.nanpercentile(values, 2)
    hi = np.nanpercentile(values, 98)

    # Fallback if percentiles collapse
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = np.nanmin(values)
        hi = np.nanmax(values)

    if symmetric:
        max_abs = np.nanmax(np.abs([lo, hi]))
        if min_span is not None:
            max_abs = max(max_abs, min_span / 2)
        xmin, xmax = -max_abs, max_abs
    else:
        span = hi - lo
        if min_span is not None:
            span = max(span, min_span)
        if span == 0:
            span = abs(hi) if hi != 0 else 1.0

        pad = pad_frac * span
        xmin = lo - pad
        xmax = hi + pad

        # For positive-only plots, start near zero
        if xmin > 0:
            xmin = 0

    return xmin, xmax


def apply_robust_xlim(ax, values, symmetric=False, min_span=None):
    """
    Apply robust x-limits to an axis.
    """
    xmin, xmax = robust_xlim_from_values(
        values,
        symmetric=symmetric,
        min_span=min_span
    )

    if xmin is not None and xmax is not None:
        ax.set_xlim(xmin, xmax)

# =============================================================================
# Zonal mean precipitation plots
# =============================================================================

def plot_zonal_mean_precip(zonal):
    """
    Plot V7 and V8 zonal mean precipitation for NH/SH and all/ocean/land.
    """
    factor, unit_label = precip_unit_factor_and_label()

    fig, axes = plt.subplots(
    2, 3,
    figsize=(14, 9),
    sharex=False,
    sharey=False
    )

    layout = [
        ("NH", "all", axes[0, 0]),
        ("NH", "ocean", axes[0, 1]),
        ("NH", "land", axes[0, 2]),
        ("SH", "all", axes[1, 0]),
        ("SH", "ocean", axes[1, 1]),
        ("SH", "land", axes[1, 2]),
    ]

    for hemi, surface_class, ax in layout:
        x_values_for_xlim = []

        for product in PLOT_PRODUCTS_WITH_REFS:
            tmp = get_class_df(zonal, product, surface_class, hemi)

            if tmp.empty:
                continue

            x = tmp["mean_precip_mmhr"].values * factor
            y = tmp["lat_mid"].values

            x_values_for_xlim.extend(x[np.isfinite(x)])

            ax.plot(
                    x, y,
                    color=PRODUCT_COLORS.get(product, None),
                    linestyle=PRODUCT_LINESTYLES.get(product, "-"),
                    linewidth=2.5 if product in GPROF_PRODUCTS else 2.0,
                    label=product
                )

        apply_robust_xlim(
            ax,
            x_values_for_xlim,
            symmetric=False,
            min_span=1.0 if CONVERT_TO_MMDAY else 0.05
        )

        title = f"{hemi} - {SURFACE_CLASS_TITLES.get(surface_class, surface_class.upper())}"
        ax.set_title(title, fontsize=15, fontweight="bold")
        style_axes(ax)

    axes[0, 0].set_ylabel("Latitude", fontsize=13, fontweight="bold")
    axes[1, 0].set_ylabel("Latitude", fontsize=13, fontweight="bold")

    for ax in axes[1, :]:
        ax.set_xlabel(f"Mean precipitation [{unit_label}]", fontsize=13, fontweight="bold")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=13,
        bbox_to_anchor=(0.5, -0.01)
    )

    fig.suptitle(
        "Zonal mean precipitation: GPROF V7/V8 with IMERG and ERA5 reference",
        fontsize=17,
        fontweight="bold",
        y=0.98
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    out = os.path.join(FIG_DIR, f"zonal_mean_precip_v7_v8_{PLOT_AGGREGATION_METHOD}_{sve_fle_part}.png")
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out}")


def plot_zonal_mean_precip_difference(zonal):
    """
    Plot V8 - V7 zonal mean precipitation difference.

    Difference is computed directly from aggregated V8 and V7 zonal means,
    not from the product='V8_minus_V7' row, because the latter represents
    mean pixelwise difference and may be weighted slightly differently.
    """
    factor, unit_label = precip_unit_factor_and_label()

    fig, axes = plt.subplots(
        2, 3,
        figsize=(14, 9),
        sharex=False,
        sharey=False
    )

    layout = [
        ("NH", "all", axes[0, 0]),
        ("NH", "ocean", axes[0, 1]),
        ("NH", "land", axes[0, 2]),
        ("SH", "all", axes[1, 0]),
        ("SH", "ocean", axes[1, 1]),
        ("SH", "land", axes[1, 2]),
    ]

    for hemi, surface_class, ax in layout:
        v7 = get_class_df(zonal, "V7", surface_class, hemi)
        v8 = get_class_df(zonal, "V8", surface_class, hemi)

        if v7.empty or v8.empty:
            continue

        merged = pd.merge(
            v7[["lat_mid", "mean_precip_mmhr"]],
            v8[["lat_mid", "mean_precip_mmhr"]],
            on="lat_mid",
            suffixes=("_v7", "_v8")
        )

        x = (merged["mean_precip_mmhr_v8"] - merged["mean_precip_mmhr_v7"]) * factor
        y = merged["lat_mid"]

        ax.plot(x, y, color="black", linewidth=2.5)
        ax.axvline(0, color="gray", linewidth=1.2)

        apply_robust_xlim(
            ax,
            x,
            symmetric=True,
            min_span=0.5 if CONVERT_TO_MMDAY else 0.02
        )

        title = f"{hemi} - {SURFACE_CLASS_TITLES.get(surface_class, surface_class.upper())}"
        ax.set_title(title, fontsize=15, fontweight="bold")
        style_axes(ax)

    axes[0, 0].set_ylabel("Latitude", fontsize=13, fontweight="bold")
    axes[1, 0].set_ylabel("Latitude", fontsize=13, fontweight="bold")

    for ax in axes[1, :]:
        ax.set_xlabel(f"V8 - V7 [{unit_label}]", fontsize=13, fontweight="bold")

    fig.suptitle(
        "Orbit-sampled zonal mean precipitation difference: V8 - V7",
        fontsize=17,
        fontweight="bold",
        y=0.98
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out = os.path.join(FIG_DIR, f"zonal_mean_precip_difference_v8_minus_v7_{PLOT_AGGREGATION_METHOD}_{sve_fle_part}.png")
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out}")


# =============================================================================
# Zonal wet-fraction plots
# =============================================================================

def plot_zonal_wet_fraction(zonal):
    """
    Plot V7 and V8 zonal wet-pixel fraction.
    """
    fig, axes = plt.subplots(
        2, 3,
        figsize=(14, 9),
        sharex=False,
        sharey=False
    )

    layout = [
        ("NH", "all", axes[0, 0]),
        ("NH", "ocean", axes[0, 1]),
        ("NH", "land", axes[0, 2]),
        ("SH", "all", axes[1, 0]),
        ("SH", "ocean", axes[1, 1]),
        ("SH", "land", axes[1, 2]),
    ]

    for hemi, surface_class, ax in layout:
        x_values_for_xlim = []

        for product in PLOT_PRODUCTS_WITH_REFS:
            tmp = get_class_df(zonal, product, surface_class, hemi)

            if tmp.empty:
                continue

            x = tmp["wet_fraction_percent"].values
            y = tmp["lat_mid"].values

            x_values_for_xlim.extend(x[np.isfinite(x)])

            ax.plot(
                    x, y,
                    color=PRODUCT_COLORS.get(product, None),
                    linestyle=PRODUCT_LINESTYLES.get(product, "-"),
                    linewidth=2.5 if product in GPROF_PRODUCTS else 2.0,
                    label=product
                )

        apply_robust_xlim(
            ax,
            x_values_for_xlim,
            symmetric=False,
            min_span=5.0
        )

        title = f"{hemi} - {SURFACE_CLASS_TITLES.get(surface_class, surface_class.upper())}"
        ax.set_title(title, fontsize=15, fontweight="bold")
        style_axes(ax)

    axes[0, 0].set_ylabel("Latitude", fontsize=13, fontweight="bold")
    axes[1, 0].set_ylabel("Latitude", fontsize=13, fontweight="bold")

    for ax in axes[1, :]:
        ax.set_xlabel("Wet-pixel fraction [%]", fontsize=13, fontweight="bold")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=13,
        bbox_to_anchor=(0.5, -0.01)
    )

    fig.suptitle(
        "Zonal precipitation fraction: GPROF V7/V8 with IMERG and ERA5 reference",
        fontsize=17,
        fontweight="bold",
        y=0.98
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    out = os.path.join(FIG_DIR, f"zonal_wet_fraction_v7_v8_{PLOT_AGGREGATION_METHOD}_{sve_fle_part}.png")
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out}")


def plot_zonal_wet_fraction_difference(zonal):
    """
    Plot V8 - V7 zonal wet-pixel fraction difference.
    """
    fig, axes = plt.subplots(
        2, 3,
        figsize=(14, 9),
        sharex=False,
        sharey=False
    )

    layout = [
        ("NH", "all", axes[0, 0]),
        ("NH", "ocean", axes[0, 1]),
        ("NH", "land", axes[0, 2]),
        ("SH", "all", axes[1, 0]),
        ("SH", "ocean", axes[1, 1]),
        ("SH", "land", axes[1, 2]),
    ]

    for hemi, surface_class, ax in layout:
        v7 = get_class_df(zonal, "V7", surface_class, hemi)
        v8 = get_class_df(zonal, "V8", surface_class, hemi)

        if v7.empty or v8.empty:
            continue

        merged = pd.merge(
            v7[["lat_mid", "wet_fraction_percent"]],
            v8[["lat_mid", "wet_fraction_percent"]],
            on="lat_mid",
            suffixes=("_v7", "_v8")
        )

        x = merged["wet_fraction_percent_v8"] - merged["wet_fraction_percent_v7"]
        y = merged["lat_mid"]

        ax.plot(x, y, color="black", linewidth=2.5)
        ax.axvline(0, color="gray", linewidth=1.2)

        apply_robust_xlim(
            ax,
            x,
            symmetric=True,
            min_span=5.0
        )

        title = f"{hemi} - {SURFACE_CLASS_TITLES.get(surface_class, surface_class.upper())}"
        ax.set_title(title, fontsize=15, fontweight="bold")
        style_axes(ax)

    axes[0, 0].set_ylabel("Latitude", fontsize=13, fontweight="bold")
    axes[1, 0].set_ylabel("Latitude", fontsize=13, fontweight="bold")

    for ax in axes[1, :]:
        ax.set_xlabel("V8 - V7 wet fraction [% points]", fontsize=13, fontweight="bold")

    fig.suptitle(
        "Orbit-sampled zonal wet-pixel fraction difference: V8 - V7",
        fontsize=17,
        fontweight="bold",
        y=0.98
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out = os.path.join(FIG_DIR, f"zonal_wet_fraction_difference_v8_minus_v7_{PLOT_AGGREGATION_METHOD}_{sve_fle_part}.png")
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out}")

# =============================================================================
# PDF plots
# =============================================================================

def plot_pdf(pdf, pdf_type="pdfc"):
    """
    Plot PDFc or PDFv for all/ocean/land.

    pdf_type:
        "pdfc" = occurrence PDF
        "pdfv" = volume PDF
    """
    if pdf_type not in ["pdfc", "pdfv"]:
        raise ValueError("pdf_type must be 'pdfc' or 'pdfv'")

    ylabel = "PDF by occurrence [%]" if pdf_type == "pdfc" else "PDF by volume [%]"
    title_label = "PDFc" if pdf_type == "pdfc" else "PDFv"

    fig, axes = plt.subplots(
        1, 3,
        figsize=(15, 4.8),
        sharey=True
    )

    for ax, surface_class in zip(axes, SURFACE_CLASSES):
        tmp = pdf[pdf["surface_class"] == surface_class].copy()

        # For the general PDF plot, use only all-hemisphere rows.
        # This prevents reference products from being counted three times.
        if "hemisphere" in tmp.columns:
            tmp = tmp[
                (tmp["hemisphere"].isna()) |
                (tmp["hemisphere"] == "ALL_HEMI")
            ].copy()

        if tmp.empty:
            ax.set_title(SURFACE_CLASS_TITLES.get(surface_class, surface_class.upper()))
            continue

        for product in PLOT_PRODUCTS_WITH_REFS:
            pp = tmp[tmp["product"] == product].copy()
            pp = pp.sort_values("bin_high")

            if pp.empty:
                continue

            x = pp["bin_high"].values
            y = pp[pdf_type].values

            ax.plot(
                    x,
                    y,
                    marker="o",
                    color=PRODUCT_COLORS.get(product, None),
                    linestyle=PRODUCT_LINESTYLES.get(product, "-"),
                    linewidth=2.5 if product in GPROF_PRODUCTS else 2.0,
                    label=product
                )

        ax.set_xscale("log", base=2)
        ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256])
        ax.set_xticklabels(["0.1", "0.2", "0.5", "1", "2", "4", "8", "16", "32", "64", "128", "256"], rotation=45)

        ax.set_title(SURFACE_CLASS_TITLES.get(surface_class, surface_class.upper()), fontsize=14, fontweight="bold")
        ax.set_xlabel("Precipitation rate [mm/hr]", fontsize=12, fontweight="bold")
        style_axes(ax)

    axes[0].set_ylabel(ylabel, fontsize=13, fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=13,
        bbox_to_anchor=(0.5, -0.08)
    )

    fig.suptitle(
        f"{title_label}: GPROF V7/V8 with IMERG and ERA5 reference",
        fontsize=16,
        fontweight="bold",
        y=1.02
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    out = os.path.join(FIG_DIR, f"{pdf_type}_v7_v8_{PLOT_AGGREGATION_METHOD}_{sve_fle_part}.png")
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out}")


# =============================================================================
# Summary Bar plots
# =============================================================================
def make_overall_summary_from_per_orbit(summary):
    """
    Build an overall descriptive-stat table from per-orbit summary values.

    This uses mean of per-orbit summary metrics, so it is mainly for quick
    visual comparison rather than pooled-pixel exact statistics.
    """
    rows = []

    metrics = [
        ("mean", "Mean"),
        ("median", "Median"),
        ("p95", "P95"),
        ("p99", "P99"),
    ]

    for metric_key, metric_label in metrics:
        v7_col = f"v7_precip_{metric_key}"
        v8_col = f"v8_precip_{metric_key}"

        if v7_col not in summary.columns or v8_col not in summary.columns:
            continue

        v7_val = summary[v7_col].mean()
        v8_val = summary[v8_col].mean()

        rows.append({
            "metric": metric_label,
            "V7": v7_val,
            "V8": v8_val,
            "diff_v8_minus_v7": v8_val - v7_val,
            "percent_change_v8_minus_v7": 100.0 * (v8_val - v7_val) / v7_val if v7_val > 0 else np.nan,
        })

    return pd.DataFrame(rows)

def plot_summary_descriptive_bars(summary):
    """
    Bar plot comparing V7 and V8 descriptive precipitation statistics.
    """
    table = make_overall_summary_from_per_orbit(summary)

    if table.empty:
        print("No descriptive summary table available for plotting.")
        return

    factor, unit_label = precip_unit_factor_and_label()

    metrics = table["metric"].values
    x = np.arange(len(metrics))
    width = 0.35

    v7 = table["V7"].values * factor
    v8 = table["V8"].values * factor

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    ax.bar(x - width / 2, v7, width, label="V7", color="blue")
    ax.bar(x + width / 2, v8, width, label="V8", color="red")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12, fontweight="bold")
    ax.set_ylabel(f"Precipitation rate [{unit_label}]", fontsize=13, fontweight="bold")
    ax.set_title("GPROF V7 vs V8 descriptive precipitation statistics", fontsize=15, fontweight="bold")

    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(frameon=False, fontsize=12)

    plt.tight_layout()

    out = os.path.join(FIG_DIR, f"summary_descriptive_stats_bars_{PLOT_AGGREGATION_METHOD}_{sve_fle_part}.png")
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out}")

    table_out = os.path.join(TABLE_DIR, f"summary_descriptive_stats_table_{PLOT_AGGREGATION_METHOD}_{sve_fle_part}.csv")
    table.to_csv(table_out, index=False)
    print(f"Saved: {table_out}")


def plot_summary_difference_bars(summary):
    """
    Bar plot of V8 - V7 differences for descriptive statistics.
    """
    table = make_overall_summary_from_per_orbit(summary)

    if table.empty:
        print("No summary difference table available for plotting.")
        return

    factor, unit_label = precip_unit_factor_and_label()

    metrics = table["metric"].values
    x = np.arange(len(metrics))
    diff = table["diff_v8_minus_v7"].values * factor

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    ax.bar(x, diff, width=0.55, color="black")

    ax.axhline(0, color="gray", linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12, fontweight="bold")
    ax.set_ylabel(f"V8 - V7 [{unit_label}]", fontsize=13, fontweight="bold")
    ax.set_title("Absolute change in descriptive precipitation statistics", fontsize=15, fontweight="bold")

    ax.grid(True, axis="y", alpha=0.35)

    plt.tight_layout()

    out = os.path.join(FIG_DIR, f"summary_descriptive_stats_difference_bars_{PLOT_AGGREGATION_METHOD}_{sve_fle_part}.png")
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out}")

def plot_summary_percent_change_bars(summary):
    """
    Bar plot of percent change in descriptive precipitation statistics.
    """
    table = make_overall_summary_from_per_orbit(summary)

    if table.empty:
        print("No percent-change summary table available for plotting.")
        return

    metrics = table["metric"].values
    x = np.arange(len(metrics))
    pct = table["percent_change_v8_minus_v7"].values

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    ax.bar(x, pct, width=0.55, color="black")

    ax.axhline(0, color="gray", linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12, fontweight="bold")
    ax.set_ylabel("Percent change V8 - V7 [%]", fontsize=13, fontweight="bold")
    ax.set_title("Percent change in descriptive precipitation statistics", fontsize=15, fontweight="bold")

    ax.grid(True, axis="y", alpha=0.35)

    plt.tight_layout()

    out = os.path.join(FIG_DIR, f"summary_descriptive_stats_percent_change_bars_{PLOT_AGGREGATION_METHOD}_{sve_fle_part}.png")
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out}")

LAND_OCEAN_SUMMARY_CSV = os.path.join(TABLE_DIR, "land_ocean_summary_table.csv")
def plot_land_ocean_summary_bars():
    """
    Plot land/ocean summary table if it exists.

    Expected columns:
        surface_class
        v7_mean_precip_mmhr
        v8_mean_precip_mmhr
        percent_diff_v8_minus_v7
        wet_fraction_diff_percent_points
    """
    if not os.path.exists(LAND_OCEAN_SUMMARY_CSV):
        print(f"Land/ocean summary table not found: {LAND_OCEAN_SUMMARY_CSV}")
        return

    df = pd.read_csv(LAND_OCEAN_SUMMARY_CSV)

    if df.empty:
        print("Land/ocean summary table is empty.")
        return

    factor, unit_label = precip_unit_factor_and_label()

    # Keep preferred order
    order = ["all", "ocean", "land"]
    df["surface_class"] = pd.Categorical(df["surface_class"], categories=order, ordered=True)
    df = df.sort_values("surface_class")

    x = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    ax.bar(
        x - width / 2,
        df["v7_mean_precip_mmhr"].values * factor,
        width,
        label="V7",
        color="blue"
    )

    ax.bar(
        x + width / 2,
        df["v8_mean_precip_mmhr"].values * factor,
        width,
        label="V8",
        color="red"
    )

    ax.set_xticks(x)
    ax.set_xticklabels([str(s).upper() for s in df["surface_class"]], fontsize=12, fontweight="bold")
    ax.set_ylabel(f"Mean precipitation rate [{unit_label}]", fontsize=13, fontweight="bold")
    ax.set_title("GPROF V7 vs V8 mean precipitation by surface class", fontsize=15, fontweight="bold")

    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(frameon=False, fontsize=12)

    plt.tight_layout()

    out = os.path.join(FIG_DIR, f"land_ocean_mean_precip_bars_{PLOT_AGGREGATION_METHOD}_{sve_fle_part}.png")
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out}")

    # Percent/difference panel
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    axes[0].bar(
        x,
        df["percent_diff_v8_minus_v7"].values,
        color="black",
        width=0.55
    )
    axes[0].axhline(0, color="gray", linewidth=1.2)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(s).upper() for s in df["surface_class"]], fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Mean precip percent difference [%]", fontsize=12, fontweight="bold")
    axes[0].set_title("V8 - V7 percent difference", fontsize=13, fontweight="bold")
    axes[0].grid(True, axis="y", alpha=0.35)

    axes[1].bar(
        x,
        df["wet_fraction_diff_percent_points"].values,
        color="black",
        width=0.55
    )
    axes[1].axhline(0, color="gray", linewidth=1.2)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(s).upper() for s in df["surface_class"]], fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Wet-fraction difference [% points]", fontsize=12, fontweight="bold")
    axes[1].set_title("V8 - V7 wet-fraction difference", fontsize=13, fontweight="bold")
    axes[1].grid(True, axis="y", alpha=0.35)

    plt.tight_layout()

    out = os.path.join(FIG_DIR, f"land_ocean_percent_and_wet_fraction_change_bars_{PLOT_AGGREGATION_METHOD}_{sve_fle_part}.png")
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out}")

def plot_pdf_by_hemisphere_surface(pdf, pdf_type="pdfv"):
    """
    Plot PDFc or PDFv separated by hemisphere and surface class.

    Rows:
        NH, SH
    Columns:
        all, ocean, land
    """
    if pdf_type not in ["pdfc", "pdfv"]:
        raise ValueError("pdf_type must be 'pdfc' or 'pdfv'")

    if "hemisphere" not in pdf.columns:
        print("No 'hemisphere' column found in PDF table. Re-run processing with hemispheric PDF support.")
        return

    ylabel = "PDF by occurrence [%]" if pdf_type == "pdfc" else "PDF by volume [%]"
    title_label = "PDFc" if pdf_type == "pdfc" else "PDFv"

    hemis = ["NH", "SH"]

    fig, axes = plt.subplots(
        2, 3,
        figsize=(15, 8.5),
        sharex=True,
        sharey=True
    )

    for i, hemi in enumerate(hemis):
        for j, surface_class in enumerate(SURFACE_CLASSES):
            ax = axes[i, j]

            tmp = pdf[
                (pdf["hemisphere"] == hemi) &
                (pdf["surface_class"] == surface_class)
            ].copy()

            if tmp.empty:
                ax.set_title(f"{hemi} - {SURFACE_CLASS_TITLES.get(surface_class, surface_class.upper())}")
                style_axes(ax)
                continue

            for product in PLOT_PRODUCTS_WITH_REFS:
                pp = tmp[tmp["product"] == product].copy()
                pp = pp.sort_values("bin_high")

                if pp.empty:
                    continue

                x = pp["bin_high"].values
                y = pp[pdf_type].values

                ax.plot(
                        x,
                        y,
                        marker="o",
                        color=PRODUCT_COLORS.get(product, None),
                        linestyle=PRODUCT_LINESTYLES.get(product, "-"),
                        linewidth=2.3 if product in GPROF_PRODUCTS else 1.9,
                        label=product
                    )

            ax.set_xscale("log", base=2)
            ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256])
            ax.set_xticklabels(
                ["0.1", "0.2", "0.5", "1", "2", "4", "8", "16", "32", "64", "128", "256"],
                rotation=45
            )

            ax.set_title(
                f"{hemi} - {SURFACE_CLASS_TITLES.get(surface_class, surface_class.upper())}",
                fontsize=14,
                fontweight="bold"
            )

            if i == 1:
                ax.set_xlabel("Precipitation rate [mm/hr]", fontsize=12, fontweight="bold")

            if j == 0:
                ax.set_ylabel(ylabel, fontsize=13, fontweight="bold")

            style_axes(ax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=13,
        bbox_to_anchor=(0.5, -0.01)
    )

    fig.suptitle(
        f"{title_label} by hemisphere and surface class: GPROF V7/V8 with references",
        fontsize=16,
        fontweight="bold",
        y=0.99
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    out = os.path.join(
        FIG_DIR,
        f"{pdf_type}_v7_v8_by_hemi_surface_{PLOT_AGGREGATION_METHOD}_{sve_fle_part}.png"
    )

    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out}")


def plot_land_ocean_summary_bars_with_refs(zonal):
    """
    Plot mean precipitation by surface class for GPROF V7/V8 plus
    IMERG/ERA5 gridded references.

    Uses zonal table and area/sample-weighted mean from n_valid.
    """
    factor, unit_label = precip_unit_factor_and_label()

    products = PLOT_PRODUCTS_WITH_REFS
    surface_order = ["all", "ocean", "land"]

    rows = []

    for product in products:
        for surface_class in surface_order:
            tmp = zonal[
                (zonal["product"] == product) &
                (zonal["surface_class"] == surface_class) &
                (zonal["n_valid"] > 0)
            ].copy()

            if tmp.empty:
                continue

            mean_precip = np.average(
                tmp["mean_precip_mmhr"],
                weights=tmp["n_valid"]
            )

            wet_fraction = 100.0 * tmp["n_wet"].sum() / tmp["n_valid"].sum()

            rows.append({
                "product": product,
                "surface_class": surface_class,
                "mean_precip_mmhr": mean_precip,
                "wet_fraction_percent": wet_fraction,
                "n_valid": int(tmp["n_valid"].sum()),
                "n_wet": int(tmp["n_wet"].sum()),
            })

    df = pd.DataFrame(rows)

    if df.empty:
        print("No land/ocean/reference summary available for plotting.")
        return

    df["surface_class"] = pd.Categorical(
        df["surface_class"],
        categories=surface_order,
        ordered=True
    )

    x = np.arange(len(surface_order))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10.5, 5.6))

    offsets = np.linspace(
        -width * (len(products) - 1) / 2,
         width * (len(products) - 1) / 2,
         len(products)
    )

    for off, product in zip(offsets, products):
        sub = df[df["product"] == product].sort_values("surface_class")

        vals = []
        for surface_class in surface_order:
            row = sub[sub["surface_class"] == surface_class]
            vals.append(row["mean_precip_mmhr"].iloc[0] * factor if not row.empty else np.nan)

        ax.bar(
            x + off,
            vals,
            width,
            label=product,
            color=PRODUCT_COLORS.get(product, None)
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [s.upper() for s in surface_order],
        fontsize=12,
        fontweight="bold"
    )

    ax.set_ylabel(f"Mean precipitation [{unit_label}]", fontsize=13, fontweight="bold")
    ax.set_title(
        "Mean precipitation by surface class: GPROF V7/V8 with references",
        fontsize=15,
        fontweight="bold"
    )

    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(frameon=False, fontsize=11, ncol=2)

    plt.tight_layout()

    out = os.path.join(
        FIG_DIR,
        f"land_ocean_mean_precip_bars_with_refs_{PLOT_AGGREGATION_METHOD}_{sve_fle_part}.png"
    )

    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out}")

    table_out = os.path.join(
        TABLE_DIR,
        f"land_ocean_mean_precip_with_refs_table_{PLOT_AGGREGATION_METHOD}_{sve_fle_part}.csv"
    )

    df.to_csv(table_out, index=False)
    print(f"Saved: {table_out}")

# =============================================================================
# Tables
# =============================================================================

def make_summary_tables(summary, zonal, pdf):
    """
    Create concise CSV tables summarizing major V8-V7 differences.
    """
    zonal_gprof = zonal[zonal["product"].isin(["V7", "V8"])].copy()
    pdf_gprof = pdf[pdf["product"].isin(["V7", "V8"])].copy()
    # -------------------------------------------------------------------------
    # Overall per-orbit summary aggregated across processed granules
    # -------------------------------------------------------------------------
    table_rows = []

    n_orbits = len(summary)

    total_common_valid = summary["common_valid_count"].sum()
    total_v7_wet = summary["v7_wet_count"].sum()
    total_v8_wet = summary["v8_wet_count"].sum()

    v7_wet_frac = 100.0 * total_v7_wet / total_common_valid if total_common_valid > 0 else np.nan
    v8_wet_frac = 100.0 * total_v8_wet / total_common_valid if total_common_valid > 0 else np.nan

    table_rows.append({
        "n_orbits": n_orbits,
        "common_valid_pixels": int(total_common_valid),
        "v7_mean_precip_mmhr_mean_of_orbits": summary["v7_precip_mean"].mean(),
        "v8_mean_precip_mmhr_mean_of_orbits": summary["v8_precip_mean"].mean(),
        "mean_diff_v8_minus_v7_mmhr": summary["diff_v8_minus_v7_mean"].mean(),
        "mean_percent_diff_where_v7_ge_threshold": summary["percent_diff_mean"].mean(),
        "v7_wet_fraction_percent_total": v7_wet_frac,
        "v8_wet_fraction_percent_total": v8_wet_frac,
        "wet_fraction_difference_percent_points": v8_wet_frac - v7_wet_frac,
    })

    overall_table = pd.DataFrame(table_rows)

    overall_path = os.path.join(TABLE_DIR, f"overall_summary_table_{PLOT_AGGREGATION_METHOD}_{sve_fle_part}.csv")
    overall_table.to_csv(overall_path, index=False)
    print(f"Saved: {overall_path}") 

    # -------------------------------------------------------------------------
    # Percent-difference summary table
    # -------------------------------------------------------------------------
    pct_cols = [
        "percent_diff_valid_count",
        "percent_diff_mean",
        "percent_diff_median",
        "percent_diff_std",
        "percent_diff_p95",
        "percent_diff_p99",
    ]

    existing_pct_cols = [c for c in pct_cols if c in summary.columns]

    pct_table = summary[["key"] + existing_pct_cols].copy()
    pct_table_path = os.path.join(TABLE_DIR, f"percent_difference_summary_by_orbit_{PLOT_AGGREGATION_METHOD}_{sve_fle_part}.csv")
    pct_table.to_csv(pct_table_path, index=False)
    print(f"Saved: {pct_table_path}")

    # -------------------------------------------------------------------------
    # Land/ocean aggregated summary from zonal stats
    # -------------------------------------------------------------------------
    land_ocean_rows = []

    for surface_class in SURFACE_CLASSES:
        v7 = zonal_gprof[
            (zonal_gprof["product"] == "V7") &
            (zonal_gprof["surface_class"] == surface_class) &
            (zonal_gprof["n_valid"] > 0)
        ].copy()

        v8 = zonal_gprof[
            (zonal_gprof["product"] == "V8") &
            (zonal_gprof["surface_class"] == surface_class) &
            (zonal_gprof["n_valid"] > 0)
        ].copy()

        if v7.empty or v8.empty:
            continue

        v7_mean = np.average(v7["mean_precip_mmhr"], weights=v7["n_valid"])
        v8_mean = np.average(v8["mean_precip_mmhr"], weights=v8["n_valid"])

        v7_wet = 100.0 * v7["n_wet"].sum() / v7["n_valid"].sum()
        v8_wet = 100.0 * v8["n_wet"].sum() / v8["n_valid"].sum()

        land_ocean_rows.append({
            "surface_class": surface_class,
            "v7_mean_precip_mmhr": v7_mean,
            "v8_mean_precip_mmhr": v8_mean,
            "diff_v8_minus_v7_mmhr": v8_mean - v7_mean,
            "percent_diff_v8_minus_v7": 100.0 * (v8_mean - v7_mean) / v7_mean if v7_mean > 0 else np.nan,
            "v7_wet_fraction_percent": v7_wet,
            "v8_wet_fraction_percent": v8_wet,
            "wet_fraction_diff_percent_points": v8_wet - v7_wet,
            "v7_valid_pixels": int(v7["n_valid"].sum()),
            "v8_valid_pixels": int(v8["n_valid"].sum()),
        })

    land_ocean_table = pd.DataFrame(land_ocean_rows)

    land_ocean_path = os.path.join(
        TABLE_DIR,
        f"land_ocean_summary_table_{PLOT_AGGREGATION_METHOD}_{sve_fle_part}.csv"
    )
    land_ocean_table.to_csv(land_ocean_path, index=False)
    print(f"Saved: {land_ocean_path}")

    land_ocean_stable_path = os.path.join(TABLE_DIR, "land_ocean_summary_table.csv")
    land_ocean_table.to_csv(land_ocean_stable_path, index=False)
    print(f"Saved stable land/ocean table: {land_ocean_stable_path}")

    pdf_gprof_allhemi = pdf_gprof[
    (pdf_gprof["hemisphere"].isna()) |
    (pdf_gprof["hemisphere"] == "ALL_HEMI")
        ].copy()

    # -------------------------------------------------------------------------
    # PDF difference table
    # -------------------------------------------------------------------------
    pdf_rows = []

    for surface_class in SURFACE_CLASSES:
        p7 = pdf_gprof_allhemi[
        (pdf_gprof_allhemi["product"] == "V7") &
        (pdf_gprof_allhemi["surface_class"] == surface_class)
            ].copy()

        p8 = pdf_gprof_allhemi[
        (pdf_gprof_allhemi["product"] == "V8") &
        (pdf_gprof_allhemi["surface_class"] == surface_class)
        ].copy()

        if p7.empty or p8.empty:
            continue

        merged = pd.merge(
            p7[["bin_low", "bin_high", "bin_label", "pdfc", "pdfv"]],
            p8[["bin_low", "bin_high", "bin_label", "pdfc", "pdfv"]],
            on=["bin_low", "bin_high", "bin_label"],
            suffixes=("_v7", "_v8")
        )

        merged["surface_class"] = surface_class
        merged["pdfc_diff_v8_minus_v7"] = merged["pdfc_v8"] - merged["pdfc_v7"]
        merged["pdfv_diff_v8_minus_v7"] = merged["pdfv_v8"] - merged["pdfv_v7"]

        pdf_rows.append(merged)

    if pdf_rows:
        pdf_diff_table = pd.concat(pdf_rows, ignore_index=True)
        pdf_diff_path = os.path.join(TABLE_DIR, f"pdf_difference_table_{PLOT_AGGREGATION_METHOD}_{sve_fle_part}.csv")
        pdf_diff_table.to_csv(pdf_diff_path, index=False)
        print(f"Saved: {pdf_diff_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    ensure_dirs()

    print("Loading CSV outputs...")
    summary, zonal, pdf = load_inputs()

    print("Making summary tables...")
    make_summary_tables(summary, zonal, pdf)
    print("Making summary bar plots...")
    plot_summary_descriptive_bars(summary)
    plot_summary_difference_bars(summary)
    plot_summary_percent_change_bars(summary)
    plot_land_ocean_summary_bars()
    plot_land_ocean_summary_bars_with_refs(zonal)

    print("Making zonal mean precipitation plots...")
    plot_zonal_mean_precip(zonal)
    plot_zonal_mean_precip_difference(zonal)

    print("Making zonal wet-fraction plots...")
    plot_zonal_wet_fraction(zonal)
    plot_zonal_wet_fraction_difference(zonal)

    print("Making PDF plots...")
    plot_pdf(pdf, pdf_type="pdfc")
    plot_pdf(pdf, pdf_type="pdfv")
    plot_pdf_by_hemisphere_surface(pdf, pdf_type="pdfc")
    plot_pdf_by_hemisphere_surface(pdf, pdf_type="pdfv")

    print("Done.")


if __name__ == "__main__":
    main()
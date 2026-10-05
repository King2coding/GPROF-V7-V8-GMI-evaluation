#!/usr/bin/env python3
"""Preliminary, snapshot-based GMI GPROF V7/V8 comparison against MRMS.

The production matchup archive is read only.  Completed NetCDF filenames are
snapshotted at startup, validated, and processed one orbit at a time.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import xarray as xr


# =============================================================================
# CENTRAL SCIENTIFIC CONFIGURATION
# All thresholds, masks, bins, minimum sample sizes, formulas, and plot limits
# are declared here rather than inside plotting functions.
# =============================================================================

STATUS_LABEL = "PRELIMINARY — incomplete production archive"

DEFAULT_INPUT_ROOT = Path(
    "/scratch/kkumah/GPROF_V7_V8_MRMS_RAQI_ERA5_MERRA2_AutoSnow_matchups/matchups/GMI"
)
DEFAULT_OUTPUT_PARENT = Path(
    "/scratch/kkumah/GPROF_V7_V8_MRMS_RAQI_ERA5_MERRA2_AutoSnow_matchups/preliminary_analyses"
)

REQUIRED_VARIABLES = (
    "surfacePrecipitation_V7",
    "surfacePrecipitation_V8",
    "MRMS_Pass2",
    "RAQI",
    "ERA5_precipitation",
    "MERRA2_T2M",
    "AutoSnow",
    "mrms_valid_flag",
)

PRODUCTS = {
    "V7": {"variable": "surfacePrecipitation_V7", "label": "GPROF V7", "color": "#2563a6"},
    "V8": {"variable": "surfacePrecipitation_V8", "label": "GPROF V8", "color": "#dd8a22"},
    "MRMS": {"variable": "MRMS_Pass2", "label": "MRMS", "color": "#202936"},
    "ERA5": {"variable": "ERA5_precipitation", "label": "ERA5", "color": "#8a6f00"},
}

RAQI_SELECTIONS = (
    ("raqi_all", None, "all finite RAQI"),
    ("raqi_ge_0p5", 0.5, "RAQI >= 0.5"),
    ("raqi_ge_0p8", 0.8, "RAQI >= 0.8"),
)

WET_THRESHOLD = 0.1
MIN_FULL_RAQI_08_SAMPLE = 10_000
MIN_MEANINGFUL_SURFACE_SAMPLE = 1_000
MIN_TEMPERATURE_PDF_SAMPLE = 1_000

TEMPERATURE_EDGES = np.array([-np.inf, 263.15, 273.15, 283.15, np.inf])
TEMPERATURE_LABELS = (
    "T2M < 263.15 K",
    "263.15 <= T2M < 273.15 K",
    "273.15 <= T2M < 283.15 K",
    "T2M >= 283.15 K",
)

# Exact zeros are counted separately. These common edges apply to positive
# precipitation occurrence PDFs. The final interval is an overflow bin.
PDF_POSITIVE_EDGES = np.array(
    [0.0, 0.01, 0.03, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0,
     8.0, 16.0, 32.0, 64.0, 128.0, np.inf]
)

SCATTER_HISTOGRAM_BINS = 160

MASK_DEFINITIONS = {
    "v7_v8_mrms_common": (
        "finite(V7) AND finite(V8) AND finite(MRMS) AND finite(RAQI) "
        "AND mrms_valid_flag == 1 AND selected RAQI condition"
    ),
    "four_product_common": (
        "finite(V7) AND finite(V8) AND finite(MRMS) AND finite(ERA5) "
        "AND finite(RAQI) AND mrms_valid_flag == 1 AND selected RAQI condition"
    ),
    "mrms_wet_v7_v8_common": (
        "v7_v8_mrms_common AND MRMS >= 0.1 mm in the selected one-hour accumulation"
    ),
    "mrms_wet_four_product_common": (
        "four_product_common AND MRMS >= 0.1 mm in the selected one-hour accumulation"
    ),
    "temperature_four_product_common": (
        "four_product_common AND finite(MERRA2_T2M) AND selected temperature-bin condition"
    ),
    "autosnow_four_product_common": (
        "four_product_common AND finite(AutoSnow) AND AutoSnow == selected class"
    ),
    "product_wet_pdf": (
        "four_product_common AND selected product >= 0.1; the base footprint population "
        "is common, while wet conditioning is product-specific and N is reported"
    ),
}

FORMULAS = {
    "mean_bias": "mean(product - MRMS)",
    "relative_bias_percent": "100 * sum(product - MRMS) / sum(MRMS)",
    "mae": "mean(abs(product - MRMS))",
    "rmse": "sqrt(mean((product - MRMS)^2))",
    "pearson_r": "cov(product, MRMS) / (population_std(product) * population_std(MRMS))",
    "standard_deviation": "population standard deviation (ddof=0)",
    "centered_rmse": "sqrt(mean(((product-mean(product))-(MRMS-mean(MRMS)))^2))",
    "pod": "hits / (hits + misses)",
    "far": "false_alarms / (hits + false_alarms)",
    "success_ratio": "hits / (hits + false_alarms) = 1 - FAR",
    "csi": "hits / (hits + misses + false_alarms)",
    "frequency_bias": "(hits + false_alarms) / (hits + misses)",
    "ets": "(hits-random_hits)/(hits+misses+false_alarms-random_hits)",
    "random_hits": "(hits+misses)*(hits+false_alarms)/N",
    "occurrence_pdf": "100 * bin_count / selected_sample_count",
    "taylor_radius": "population_std(product) / population_std(MRMS)",
    "taylor_angle": "arccos(Pearson correlation)",
}

AUTOSNOW_NAMES = {
    0: "clear water",
    1: "snow-free land",
    2: "snow-covered land",
    3: "ice-covered water",
}

ORBIT_RE = re.compile(r"_(\d{6})_(\d{8})\.nc$")


def safe_divide(numerator, denominator):
    return float(numerator / denominator) if denominator != 0 else np.nan


@dataclass
class PairStats:
    """Incremental exact pair statistics, refactored from IEPPO_utils.py."""

    n: int = 0
    sum_ref: float = 0.0
    sum_product: float = 0.0
    sum_ref2: float = 0.0
    sum_product2: float = 0.0
    sum_cross: float = 0.0
    sum_error: float = 0.0
    sum_abs_error: float = 0.0
    sum_sq_error: float = 0.0

    def update(self, reference, product):
        reference = np.asarray(reference, dtype=np.float64)
        product = np.asarray(product, dtype=np.float64)
        if reference.size != product.size:
            raise ValueError("PairStats inputs differ in size")
        if reference.size == 0:
            return
        if not (np.all(np.isfinite(reference)) and np.all(np.isfinite(product))):
            raise ValueError("PairStats received nonfinite values")
        error = product - reference
        self.n += int(reference.size)
        self.sum_ref += float(np.sum(reference, dtype=np.float64))
        self.sum_product += float(np.sum(product, dtype=np.float64))
        self.sum_ref2 += float(np.sum(reference * reference, dtype=np.float64))
        self.sum_product2 += float(np.sum(product * product, dtype=np.float64))
        self.sum_cross += float(np.sum(reference * product, dtype=np.float64))
        self.sum_error += float(np.sum(error, dtype=np.float64))
        self.sum_abs_error += float(np.sum(np.abs(error), dtype=np.float64))
        self.sum_sq_error += float(np.sum(error * error, dtype=np.float64))

    def result(self):
        if self.n == 0:
            return {name: np.nan for name in (
                "reference_mean", "product_mean", "mean_bias", "relative_bias_percent",
                "mae", "rmse", "pearson_r", "reference_std", "product_std",
                "centered_rmse", "normalized_std", "normalized_centered_rmse"
            )} | {"N": 0}
        n = float(self.n)
        mean_ref = self.sum_ref / n
        mean_product = self.sum_product / n
        var_ref = max(self.sum_ref2 / n - mean_ref * mean_ref, 0.0)
        var_product = max(self.sum_product2 / n - mean_product * mean_product, 0.0)
        covariance = self.sum_cross / n - mean_ref * mean_product
        std_ref = math.sqrt(var_ref)
        std_product = math.sqrt(var_product)
        correlation = safe_divide(covariance, std_ref * std_product)
        if np.isfinite(correlation):
            correlation = float(np.clip(correlation, -1.0, 1.0))
        mean_error = self.sum_error / n
        centered_error_variance = max(self.sum_sq_error / n - mean_error * mean_error, 0.0)
        centered_rmse = math.sqrt(centered_error_variance)
        return {
            "N": self.n,
            "reference_mean": mean_ref,
            "product_mean": mean_product,
            "mean_bias": mean_error,
            "relative_bias_percent": safe_divide(100.0 * self.sum_error, self.sum_ref),
            "mae": self.sum_abs_error / n,
            "rmse": math.sqrt(self.sum_sq_error / n),
            "pearson_r": correlation,
            "reference_std": std_ref,
            "product_std": std_product,
            "centered_rmse": centered_rmse,
            "normalized_std": safe_divide(std_product, std_ref),
            "normalized_centered_rmse": safe_divide(centered_rmse, std_ref),
        }


@dataclass
class ContingencyStats:
    """Incremental contingency metrics, refactored from IEPPO_utils.py."""

    hits: int = 0
    misses: int = 0
    false_alarms: int = 0
    correct_negatives: int = 0

    def update(self, reference, product, threshold=WET_THRESHOLD):
        reference_event = np.asarray(reference) >= threshold
        product_event = np.asarray(product) >= threshold
        self.hits += int(np.count_nonzero(reference_event & product_event))
        self.misses += int(np.count_nonzero(reference_event & ~product_event))
        self.false_alarms += int(np.count_nonzero(~reference_event & product_event))
        self.correct_negatives += int(np.count_nonzero(~reference_event & ~product_event))

    def result(self):
        h, m, f, c = self.hits, self.misses, self.false_alarms, self.correct_negatives
        n = h + m + f + c
        random_hits = safe_divide((h + m) * (h + f), n)
        ets_denominator = h + m + f - random_hits if np.isfinite(random_hits) else np.nan
        return {
            "N": n, "hits": h, "misses": m, "false_alarms": f,
            "correct_negatives": c,
            "POD": safe_divide(h, h + m),
            "FAR": safe_divide(f, h + f),
            "success_ratio": safe_divide(h, h + f),
            "CSI": safe_divide(h, h + m + f),
            "frequency_bias": safe_divide(h + f, h + m),
            "ETS": safe_divide(h - random_hits, ets_denominator),
            "random_hits": random_hits,
        }


@dataclass
class MeanStats:
    n: int = 0
    total: float = 0.0

    def update(self, values):
        values = np.asarray(values, dtype=np.float64)
        self.n += int(values.size)
        self.total += float(np.sum(values, dtype=np.float64))

    @property
    def mean(self):
        return safe_divide(self.total, self.n)


@dataclass
class OccurrenceDistribution:
    """Incremental occurrence PDF with an explicit exact-zero category."""

    zero_count: int = 0
    positive_counts: np.ndarray = field(
        default_factory=lambda: np.zeros(len(PDF_POSITIVE_EDGES) - 1, dtype=np.int64)
    )
    total_count: int = 0

    def update(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return
        if not np.all(np.isfinite(values)):
            raise ValueError("OccurrenceDistribution received nonfinite values")
        self.zero_count += int(np.count_nonzero(values == 0.0))
        positive = values[values > 0.0]
        self.positive_counts += np.histogram(positive, bins=PDF_POSITIVE_EDGES)[0]
        self.total_count += int(values.size)

    def records(self, product, selection, distribution, subgroup="all surfaces"):
        labels = ["exact zero"]
        for left, right in zip(PDF_POSITIVE_EDGES[:-1], PDF_POSITIVE_EDGES[1:]):
            labels.append(f"> {left:g}" if np.isinf(right) else f"{left:g}–{right:g}")
        counts = np.concatenate(([self.zero_count], self.positive_counts))
        return [{
            "status_label": STATUS_LABEL,
            "raqi_selection": selection,
            "distribution": distribution,
            "subgroup": subgroup,
            "product": product,
            "bin_order": index,
            "bin_label": label,
            "count": int(count),
            "selected_sample_count": self.total_count,
            "occurrence_percent": safe_divide(100.0 * count, self.total_count),
        } for index, (label, count) in enumerate(zip(labels, counts))]


def new_mean_group():
    return {product: MeanStats() for product in PRODUCTS}


def new_distribution_group(products=tuple(PRODUCTS)):
    return {product: OccurrenceDistribution() for product in products}


def new_selection_state():
    return {
        "counts": {"v7v8_common": 0, "four_product_common": 0,
                   "mrms_wet_v7v8": 0, "mrms_wet_four": 0},
        "pair_v7v8": {
            "all": {"V7": PairStats(), "V8": PairStats()},
            "mrms_wet": {"V7": PairStats(), "V8": PairStats()},
        },
        "pair_four": {
            "all": {"V7": PairStats(), "V8": PairStats(), "ERA5": PairStats()},
            "mrms_wet": {"V7": PairStats(), "V8": PairStats(), "ERA5": PairStats()},
        },
        "categorical": {"V7": ContingencyStats(), "V8": ContingencyStats()},
        "means": {"all surfaces": new_mean_group()},
        "surface_counts": {value: 0 for value in AUTOSNOW_NAMES},
        "temperature_counts": {label: 0 for label in TEMPERATURE_LABELS},
        "temperature_means": {label: new_mean_group() for label in TEMPERATURE_LABELS},
        "pdf_all": new_distribution_group(),
        "pdf_wet": new_distribution_group(),
        "temperature_pdf": {
            label: new_distribution_group(("V7", "V8", "MRMS"))
            for label in TEMPERATURE_LABELS
        },
        "scatter_max": {"all": 0.0, "mrms_wet": 0.0},
    }


def selection_mask(base_mask, raqi, threshold):
    if threshold is None:
        return base_mask
    return base_mask & (raqi >= threshold)


def load_required_arrays(path):
    with xr.open_dataset(path, decode_times=False, decode_timedelta=False) as ds:
        missing = [name for name in REQUIRED_VARIABLES if name not in ds]
        if missing:
            raise KeyError(f"missing required variables: {', '.join(missing)}")
        shapes = {name: ds[name].shape for name in REQUIRED_VARIABLES}
        if len(set(shapes.values())) != 1:
            raise ValueError(f"required-variable shapes differ: {shapes}")
        arrays = {name: ds[name].values.reshape(-1) for name in REQUIRED_VARIABLES}
        attrs = dict(ds.attrs)
    return arrays, attrs


def orbit_metadata(path, attrs):
    match = ORBIT_RE.search(path.name)
    orbit = match.group(1) if match else "unknown"
    date = match.group(2) if match else "unknown"
    return orbit, date, attrs.get("code_version", "unknown")


def base_masks(arrays):
    v7 = arrays[PRODUCTS["V7"]["variable"]]
    v8 = arrays[PRODUCTS["V8"]["variable"]]
    mrms = arrays[PRODUCTS["MRMS"]["variable"]]
    era5 = arrays[PRODUCTS["ERA5"]["variable"]]
    raqi = arrays["RAQI"]
    mrms_flag = arrays["mrms_valid_flag"]
    v7v8 = (np.isfinite(v7) & np.isfinite(v8) & np.isfinite(mrms) &
            np.isfinite(raqi) & (mrms_flag == 1))
    four = v7v8 & np.isfinite(era5)
    return v7v8, four


def update_selection_state(state, arrays, threshold):
    values = {product: arrays[meta["variable"]] for product, meta in PRODUCTS.items()}
    raqi = arrays["RAQI"]
    snow = arrays["AutoSnow"]
    temperature = arrays["MERRA2_T2M"]
    base_v7v8, base_four = base_masks(arrays)
    mask_v7v8 = selection_mask(base_v7v8, raqi, threshold)
    mask_four = selection_mask(base_four, raqi, threshold)
    wet_v7v8 = mask_v7v8 & (values["MRMS"] >= WET_THRESHOLD)
    wet_four = mask_four & (values["MRMS"] >= WET_THRESHOLD)

    state["counts"]["v7v8_common"] += int(np.count_nonzero(mask_v7v8))
    state["counts"]["four_product_common"] += int(np.count_nonzero(mask_four))
    state["counts"]["mrms_wet_v7v8"] += int(np.count_nonzero(wet_v7v8))
    state["counts"]["mrms_wet_four"] += int(np.count_nonzero(wet_four))

    for product in ("V7", "V8"):
        state["pair_v7v8"]["all"][product].update(values["MRMS"][mask_v7v8], values[product][mask_v7v8])
        state["pair_v7v8"]["mrms_wet"][product].update(values["MRMS"][wet_v7v8], values[product][wet_v7v8])
        state["categorical"][product].update(values["MRMS"][mask_v7v8], values[product][mask_v7v8])

    for product in ("V7", "V8", "ERA5"):
        state["pair_four"]["all"][product].update(values["MRMS"][mask_four], values[product][mask_four])
        state["pair_four"]["mrms_wet"][product].update(values["MRMS"][wet_four], values[product][wet_four])

    for product in PRODUCTS:
        selected = values[product][mask_four]
        state["means"]["all surfaces"][product].update(selected)
        state["pdf_all"][product].update(selected)
        state["pdf_wet"][product].update(selected[selected >= WET_THRESHOLD])

    for snow_class in AUTOSNOW_NAMES:
        group_mask = mask_four & np.isfinite(snow) & (snow == snow_class)
        count = int(np.count_nonzero(group_mask))
        state["surface_counts"][snow_class] += count
        group_name = f"AutoSnow {snow_class}: {AUTOSNOW_NAMES[snow_class]}"
        if group_name not in state["means"]:
            state["means"][group_name] = new_mean_group()
        for product in PRODUCTS:
            state["means"][group_name][product].update(values[product][group_mask])

    finite_temperature = np.isfinite(temperature)
    for index, label in enumerate(TEMPERATURE_LABELS):
        group_mask = (mask_four & finite_temperature &
                      (temperature >= TEMPERATURE_EDGES[index]) &
                      (temperature < TEMPERATURE_EDGES[index + 1]))
        state["temperature_counts"][label] += int(np.count_nonzero(group_mask))
        for product in PRODUCTS:
            state["temperature_means"][label][product].update(values[product][group_mask])
        for product in ("V7", "V8", "MRMS"):
            state["temperature_pdf"][label][product].update(values[product][group_mask])

    if np.any(mask_four):
        state["scatter_max"]["all"] = max(
            state["scatter_max"]["all"],
            *(float(np.max(values[product][mask_four])) for product in PRODUCTS),
        )
    if np.any(wet_four):
        state["scatter_max"]["mrms_wet"] = max(
            state["scatter_max"]["mrms_wet"],
            *(float(np.max(values[product][wet_four])) for product in PRODUCTS),
        )


def write_csv(records, path):
    frame = records if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    if "status_label" not in frame.columns:
        frame.insert(0, "status_label", STATUS_LABEL)
    frame.to_csv(path, index=False)
    return frame


def preliminary_title(ax, title, subtitle=None):
    ax.set_title(title, loc="left", fontsize=12, color="#202936", pad=12)
    if subtitle:
        ax.text(0.0, 1.01, subtitle, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=8.5, color="#596273")


def save_figure(fig, path):
    # Keep the preliminary status outside the plotting area so it cannot obscure
    # axis labels, tick labels, legends, or scientific annotations.
    fig.text(0.5, 1.015, STATUS_LABEL, ha="center", va="bottom",
             fontsize=9, fontweight="bold", color="#8a4f00")
    fig.savefig(path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def mean_records(state, selection):
    records = []
    for group, product_stats in state["means"].items():
        n = product_stats["MRMS"].n
        if group not in ("all surfaces", "AutoSnow 1: snow-free land", "AutoSnow 2: snow-covered land"):
            if n < MIN_MEANINGFUL_SURFACE_SAMPLE:
                continue
        means = {product: product_stats[product].mean for product in PRODUCTS}
        difference = means["V8"] - means["V7"]
        records.append({
            "raqi_selection": selection, "surface_group": group, "N": n,
            "GPROF_V7_mean": means["V7"], "GPROF_V8_mean": means["V8"],
            "MRMS_mean": means["MRMS"], "ERA5_mean": means["ERA5"],
            "V8_minus_V7": difference,
            "V8_percent_change_from_V7": safe_divide(100.0 * difference, means["V7"]),
            "mask_definition": MASK_DEFINITIONS["four_product_common"],
        })
    return records


def plot_means(records, selection_label, output_path):
    frame = pd.DataFrame(records)
    groups = frame["surface_group"].tolist()
    x = np.arange(len(groups), dtype=float)
    width = 0.19
    fig, ax = plt.subplots(figsize=(max(10, 2.2 * len(groups)), 6), constrained_layout=True)
    for offset, product in enumerate(PRODUCTS):
        column = {"V7": "GPROF_V7_mean", "V8": "GPROF_V8_mean",
                  "MRMS": "MRMS_mean", "ERA5": "ERA5_mean"}[product]
        ax.bar(x + (offset - 1.5) * width, frame[column], width,
               label=PRODUCTS[product]["label"], color=PRODUCTS[product]["color"],
               edgecolor="#202936", linewidth=.45)
    mean_values = frame[["GPROF_V7_mean", "GPROF_V8_mean", "MRMS_mean", "ERA5_mean"]].to_numpy(float)
    finite_means = mean_values[np.isfinite(mean_values)]
    ymax = max(float(np.max(finite_means)), .01) if finite_means.size else .01
    for index, n in enumerate(frame["N"]):
        ax.text(index, ymax * 1.035, f"N={int(n):,}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, ymax * 1.17)
    ax.set_xticks(x); ax.set_xticklabels(groups, rotation=18, ha="right")
    ax.set_ylabel("Mean precipitation (native product units)")
    if not finite_means.size:
        ax.text(.5, .5, "No samples satisfy this selection", transform=ax.transAxes,
                ha="center", va="center", color="#596273")
    preliminary_title(ax, "Mean precipitation by surface group",
                      f"{selection_label}; identical four-product common footprints")
    ax.grid(axis="y", color="#d8dde5", linewidth=.6); ax.legend(ncol=4, frameon=False)
    save_figure(fig, output_path)


def pdf_records(state, selection):
    records = []
    for product in PRODUCTS:
        records.extend(state["pdf_all"][product].records(product, selection, "including zero"))
        records.extend(state["pdf_wet"][product].records(product, selection, "product-wet >= 0.1"))
    for label in TEMPERATURE_LABELS:
        if state["temperature_counts"][label] < MIN_TEMPERATURE_PDF_SAMPLE:
            continue
        for product in ("V7", "V8", "MRMS"):
            records.extend(state["temperature_pdf"][label][product].records(
                product, selection, "including zero", subgroup=label
            ))
    return records


def plot_pdf(frame, selection_label, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), constrained_layout=True)
    for ax, distribution in zip(axes, ("including zero", "product-wet >= 0.1")):
        subset = frame[(frame["distribution"] == distribution) & (frame["subgroup"] == "all surfaces")]
        for product in PRODUCTS:
            product_frame = subset[subset["product"] == product].sort_values("bin_order")
            label = f"{PRODUCTS[product]['label']} (N={int(product_frame['selected_sample_count'].iloc[0]):,})"
            occurrence = product_frame["occurrence_percent"].to_numpy(float)
            occurrence = np.where(occurrence > 0, occurrence, np.nan)
            ax.plot(product_frame["bin_order"], occurrence,
                    marker="o", ms=3.5, lw=1.7, color=PRODUCTS[product]["color"], label=label)
        labels = subset[subset["product"] == "V7"].sort_values("bin_order")["bin_label"].tolist()
        ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=8)
        positive = subset["occurrence_percent"].to_numpy(float)
        if np.any(positive > 0):
            ax.set_yscale("log")
        ax.set_ylabel("Occurrence frequency (%)")
        title = "Common-bin precipitation PDF including exact zero" if distribution == "including zero" else "Product-wet precipitation PDF"
        preliminary_title(ax, title, f"{selection_label}; logarithmic frequency axis")
        ax.grid(axis="y", which="both", color="#d8dde5", linewidth=.5)
        ax.legend(frameon=False, fontsize=8)
    save_figure(fig, output_path)


def continuous_records(state, selection):
    records = []
    for sample_name, stats_by_product in state["pair_v7v8"].items():
        for product, stats in stats_by_product.items():
            records.append({
                "raqi_selection": selection,
                "sample": "all common-valid" if sample_name == "all" else "MRMS-wet >= 0.1",
                "product": product,
                "mask_definition": MASK_DEFINITIONS[
                    "v7_v8_mrms_common" if sample_name == "all" else "mrms_wet_v7_v8_common"
                ],
                **stats.result(),
            })
    return records


def categorical_records(state, selection):
    return [{
        "raqi_selection": selection, "product": product,
        "event_definition": "MRMS >= 0.1 and product >= 0.1 use inclusive thresholds",
        "mask_definition": MASK_DEFINITIONS["v7_v8_mrms_common"],
        **stats.result(),
    } for product, stats in state["categorical"].items()]


def draw_performance_background(ax):
    """CSI and bias geometry adapted from IEPPO_utils.draw_perf_background."""
    sr = np.linspace(0.005, 0.995, 400)
    pod = np.linspace(0.005, 0.995, 400)
    SR, POD = np.meshgrid(sr, pod)
    with np.errstate(divide="ignore", invalid="ignore"):
        csi = 1.0 / (1.0 / SR + 1.0 / POD - 1.0)
    levels = np.arange(0.1, 1.0, 0.1)
    contours = ax.contour(SR, POD, csi, levels=levels, colors="#8a4f00",
                          linewidths=.8, alpha=.75)
    ax.clabel(contours, fmt="%.1f", fontsize=7)
    for bias in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        y = bias * sr
        valid = y <= 1
        ax.plot(sr[valid], y[valid], "--", color="#667085", lw=.8)
        ylab = .96 if bias >= 1 else bias * .92
        xlab = ylab / bias
        ax.text(xlab, ylab, f"B={bias:g}", fontsize=7, color="#596273",
                ha="center", va="center")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Success ratio (1 − FAR)", ylabel="POD")
    ax.grid(color="#d8dde5", linewidth=.5)


def plot_roebber(records, selection_label, output_path):
    frame = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(7.5, 7), constrained_layout=True)
    draw_performance_background(ax)
    markers = {"V7": "o", "V8": "s"}
    for _, row in frame.iterrows():
        product = row["product"]
        ax.scatter(row["success_ratio"], row["POD"], s=95, marker=markers[product],
                   color=PRODUCTS[product]["color"], edgecolor="#202936", zorder=5,
                   label=f"{PRODUCTS[product]['label']} (N={int(row['N']):,})")
    preliminary_title(ax, "Roebber precipitation performance diagram",
                      f"{selection_label}; event threshold >= 0.1")
    ax.legend(loc="lower right", frameon=True)
    save_figure(fig, output_path)


def temperature_records(state, selection):
    records = []
    for label in TEMPERATURE_LABELS:
        stats = state["temperature_means"][label]
        records.append({
            "raqi_selection": selection, "temperature_bin": label,
            "N": state["temperature_counts"][label],
            **{f"{product}_mean": stats[product].mean for product in PRODUCTS},
            "mask_definition": MASK_DEFINITIONS["temperature_four_product_common"],
            "sparse_for_interpretation": state["temperature_counts"][label] < MIN_TEMPERATURE_PDF_SAMPLE,
        })
    return records


def plot_temperature_means(records, selection_label, output_path):
    frame = pd.DataFrame(records)
    x = np.arange(len(frame)); width = .19
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    for index, product in enumerate(PRODUCTS):
        ax.bar(x + (index - 1.5) * width, frame[f"{product}_mean"], width,
               color=PRODUCTS[product]["color"], edgecolor="#202936", linewidth=.45,
               label=PRODUCTS[product]["label"])
    mean_values = frame[[f"{product}_mean" for product in PRODUCTS]].to_numpy(float)
    finite_means = mean_values[np.isfinite(mean_values)]
    ymax = max(float(np.max(finite_means)), .01) if finite_means.size else .01
    for index, n in enumerate(frame["N"]):
        ax.text(index, ymax * 1.035, f"N={int(n):,}", ha="center", fontsize=8)
    ax.set_ylim(0, ymax * 1.17); ax.set_xticks(x); ax.set_xticklabels(frame["temperature_bin"], rotation=15)
    ax.set_ylabel("Mean precipitation (native product units)")
    if not finite_means.size:
        ax.text(.5, .5, "No samples satisfy this selection", transform=ax.transAxes,
                ha="center", va="center", color="#596273")
    preliminary_title(ax, "Mean precipitation by MERRA-2 2-m temperature",
                      f"{selection_label}; identical four-product footprints with finite T2M")
    ax.grid(axis="y", color="#d8dde5", linewidth=.6); ax.legend(ncol=4, frameon=False)
    save_figure(fig, output_path)


def plot_temperature_pdfs(frame, state, selection_label, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    plotted = False
    for ax, label in zip(axes.flat, TEMPERATURE_LABELS):
        n = state["temperature_counts"][label]
        subset = frame[(frame["subgroup"] == label) & (frame["distribution"] == "including zero")]
        if n < MIN_TEMPERATURE_PDF_SAMPLE or subset.empty:
            ax.text(.5, .5, f"Sparse bin — N={n:,}\nNo PDF generated", ha="center", va="center",
                    transform=ax.transAxes, color="#596273")
            ax.set_axis_off()
            continue
        plotted = True
        for product in ("V7", "V8", "MRMS"):
            product_frame = subset[subset["product"] == product].sort_values("bin_order")
            occurrence = product_frame["occurrence_percent"].to_numpy(float)
            occurrence = np.where(occurrence > 0, occurrence, np.nan)
            ax.plot(product_frame["bin_order"], occurrence,
                    marker="o", ms=3, lw=1.5, color=PRODUCTS[product]["color"],
                    label=PRODUCTS[product]["label"])
        labels = subset[subset["product"] == "V7"].sort_values("bin_order")["bin_label"].tolist()
        ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
        ax.set_yscale("log"); ax.set_ylabel("Occurrence frequency (%)")
        preliminary_title(ax, label, f"N={n:,}; {selection_label}")
        ax.grid(axis="y", which="both", color="#d8dde5", linewidth=.5); ax.legend(frameon=False)
    if plotted:
        save_figure(fig, output_path)
    else:
        plt.close(fig)


def plot_taylor(state, selection_label, output_path):
    results = {product: state["pair_four"]["mrms_wet"][product].result()
               for product in ("V7", "V8", "ERA5")}
    correlations = [value["pearson_r"] for value in results.values() if np.isfinite(value["pearson_r"])]
    theta_max = np.pi if correlations and min(correlations) < 0 else np.pi / 2
    radii = [value["normalized_std"] for value in results.values() if np.isfinite(value["normalized_std"])]
    radius_max = max(1.5, (max(radii) if radii else 1.0) * 1.2)
    fig = plt.figure(figsize=(8.5, 7.5), constrained_layout=True)
    ax = fig.add_subplot(111, polar=True)
    ax.set_thetamin(0); ax.set_thetamax(np.degrees(theta_max)); ax.set_ylim(0, radius_max)
    corr_ticks = np.array([0.0, .2, .4, .6, .8, .9, .95, .99, 1.0])
    if theta_max > np.pi / 2:
        corr_ticks = np.array([-1.0, -.8, -.4, 0, .4, .8, .9, .95, .99, 1.0])
    angles = np.arccos(np.clip(corr_ticks, -1, 1))
    keep = angles <= theta_max + 1e-9
    ax.set_xticks(angles[keep]); ax.set_xticklabels([f"{v:g}" for v in corr_ticks[keep]])
    theta_grid, radius_grid = np.meshgrid(np.linspace(0, theta_max, 240),
                                          np.linspace(0.01, radius_max, 240))
    crmse = np.sqrt(1 + radius_grid ** 2 - 2 * radius_grid * np.cos(theta_grid))
    levels = np.arange(.25, max(1.01, radius_max + .5), .25)
    cs = ax.contour(theta_grid, radius_grid, crmse, levels=levels,
                    colors="#a0a8b4", linewidths=.7)
    ax.clabel(cs, fontsize=7, fmt="%.2g")
    ax.scatter(0, 1, marker="*", s=150, color=PRODUCTS["MRMS"]["color"], label="MRMS reference")
    markers = {"V7": "o", "V8": "s", "ERA5": "^"}
    for product, result in results.items():
        if not (np.isfinite(result["pearson_r"]) and np.isfinite(result["normalized_std"])):
            continue
        ax.scatter(np.arccos(np.clip(result["pearson_r"], -1, 1)), result["normalized_std"],
                   s=90, marker=markers[product], color=PRODUCTS[product]["color"],
                   edgecolor="#202936", label=f"{PRODUCTS[product]['label']} (N={result['N']:,})")
    ax.set_ylabel("Normalized standard deviation", labelpad=28)
    ax.set_title("Taylor diagram — MRMS-wet common sample\n"
                 f"{selection_label}; angle=correlation; contours=normalized centered RMSE",
                 va="bottom", fontsize=12, color="#202936")
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.08), frameon=False)
    if not correlations:
        ax.text(.5, .5, "No MRMS-wet samples satisfy this selection",
                transform=ax.transAxes, ha="center", va="center", color="#596273")
    save_figure(fig, output_path)


def make_scatter_histograms(used_paths, active_selections, states, logger):
    histograms = {}
    for key, threshold, _ in active_selections:
        histograms[key] = {}
        for sample in ("all", "mrms_wet"):
            maximum = max(states[key]["scatter_max"][sample], 1.0e-6)
            edges = np.linspace(0.0, np.log1p(maximum), SCATTER_HISTOGRAM_BINS + 1)
            histograms[key][sample] = {
                "edges": edges,
                "hist": {product: np.zeros((SCATTER_HISTOGRAM_BINS, SCATTER_HISTOGRAM_BINS), dtype=np.int64)
                         for product in ("V7", "V8", "ERA5")},
            }
    for index, path in enumerate(used_paths, 1):
        arrays, _ = load_required_arrays(path)
        values = {product: arrays[meta["variable"]] for product, meta in PRODUCTS.items()}
        base_v7v8, base_four = base_masks(arrays)
        for key, threshold, _ in active_selections:
            mask = selection_mask(base_four, arrays["RAQI"], threshold)
            for sample in ("all", "mrms_wet"):
                selected = mask if sample == "all" else mask & (values["MRMS"] >= WET_THRESHOLD)
                edges = histograms[key][sample]["edges"]
                x = np.log1p(values["MRMS"][selected])
                for product in ("V7", "V8", "ERA5"):
                    y = np.log1p(values[product][selected])
                    histograms[key][sample]["hist"][product] += np.histogram2d(x, y, bins=(edges, edges))[0].astype(np.int64)
        if index % 25 == 0 or index == len(used_paths):
            logger.info("Scatter-density pass: %d/%d files", index, len(used_paths))
    return histograms


def plot_scatter(histogram, pair_stats, selection_label, sample_label, output_path):
    edges = histogram["edges"]
    hists = histogram["hist"]
    vmax = max(int(np.max(value)) for value in hists.values())
    norm = LogNorm(vmin=1, vmax=max(vmax, 2))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.7), constrained_layout=True, sharex=True, sharey=True)
    image = None
    for ax, product in zip(axes, ("V7", "V8", "ERA5")):
        masked = np.ma.masked_where(hists[product].T == 0, hists[product].T)
        image = ax.pcolormesh(edges, edges, masked, cmap="cividis", norm=norm, shading="auto")
        ax.plot([edges[0], edges[-1]], [edges[0], edges[-1]], color="#202936", lw=1.1, ls="--")
        result = pair_stats[product].result()
        ax.text(.04, .96,
                f"N={result['N']:,}\nBias={result['mean_bias']:.3g}\nMAE={result['mae']:.3g}\n"
                f"RMSE={result['rmse']:.3g}\nr={result['pearson_r']:.3f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
                bbox={"facecolor": "white", "edgecolor": "#d8dde5", "alpha": .9})
        ax.set_xlabel(f"log1p(MRMS [mm per selected hour])")
        preliminary_title(ax, f"MRMS versus {PRODUCTS[product]['label']}", sample_label)
    axes[0].set_ylabel("log1p(product precipitation in native units)")
    fig.colorbar(image, ax=axes, label="Footprint count per log1p density cell", shrink=.88)
    fig.suptitle(f"Matched product-versus-MRMS density panels — {selection_label}",
                 fontsize=13, color="#202936")
    save_figure(fig, output_path)


def build_mask_count_records(states, active_selections):
    records = []
    for key, _, label in active_selections:
        state = states[key]
        common_counts = [
            ("continuous metrics: all", "V7,V8", state["counts"]["v7v8_common"], "v7_v8_mrms_common"),
            ("continuous metrics: MRMS wet", "V7,V8", state["counts"]["mrms_wet_v7v8"], "mrms_wet_v7_v8_common"),
            ("categorical metrics", "V7,V8", state["counts"]["v7v8_common"], "v7_v8_mrms_common"),
            ("four-product means/PDF base", "V7,V8,MRMS,ERA5", state["counts"]["four_product_common"], "four_product_common"),
            ("Taylor and wet scatter", "V7,V8,ERA5 vs MRMS", state["counts"]["mrms_wet_four"], "mrms_wet_four_product_common"),
            ("all-valid scatter", "V7,V8,ERA5 vs MRMS", state["counts"]["four_product_common"], "four_product_common"),
        ]
        for analysis, products, count, mask_key in common_counts:
            records.append({"raqi_selection": label, "analysis_product": analysis,
                            "products": products, "sample_count": count,
                            "mask_definition": MASK_DEFINITIONS[mask_key]})
        for snow_class, count in state["surface_counts"].items():
            records.append({"raqi_selection": label,
                            "analysis_product": f"AutoSnow class {snow_class} means",
                            "products": "V7,V8,MRMS,ERA5", "sample_count": count,
                            "mask_definition": MASK_DEFINITIONS["autosnow_four_product_common"]})
        for temp_label, count in state["temperature_counts"].items():
            records.append({"raqi_selection": label,
                            "analysis_product": f"temperature means/PDF: {temp_label}",
                            "products": "V7,V8,MRMS,ERA5", "sample_count": count,
                            "mask_definition": MASK_DEFINITIONS["temperature_four_product_common"]})
        for product in PRODUCTS:
            records.append({"raqi_selection": label,
                            "analysis_product": f"product-wet PDF: {product}",
                            "products": product,
                            "sample_count": state["pdf_wet"][product].total_count,
                            "mask_definition": MASK_DEFINITIONS["product_wet_pdf"]})
    return records


def validation_records(states, active_selections):
    records = []
    def add(check, passed, detail):
        records.append({"check": check, "passed": bool(passed), "detail": detail})
    for key, _, label in active_selections:
        state = states[key]
        add(f"{label}: four-product subset", state["counts"]["four_product_common"] <= state["counts"]["v7v8_common"],
            f"four={state['counts']['four_product_common']}, v7v8={state['counts']['v7v8_common']}")
        add(f"{label}: wet subset", state["counts"]["mrms_wet_four"] <= state["counts"]["four_product_common"],
            f"wet_four={state['counts']['mrms_wet_four']}")
        n_v7 = state["pair_v7v8"]["all"]["V7"].n
        n_v8 = state["pair_v7v8"]["all"]["V8"].n
        add(f"{label}: identical V7/V8 metric sample", n_v7 == n_v8 == state["counts"]["v7v8_common"],
            f"V7={n_v7}, V8={n_v8}")
        cat_n = [state["categorical"][product].result()["N"] for product in ("V7", "V8")]
        add(f"{label}: contingency totals", all(n == state["counts"]["v7v8_common"] for n in cat_n),
            f"categorical={cat_n}")
        pdf_n = [state["pdf_all"][product].total_count for product in PRODUCTS]
        add(f"{label}: identical four-product PDF base", all(n == state["counts"]["four_product_common"] for n in pdf_n),
            f"PDF counts={pdf_n}")
        temp_sum = sum(state["temperature_counts"].values())
        add(f"{label}: temperature bins bounded by four-product sample", temp_sum <= state["counts"]["four_product_common"],
            f"temperature finite={temp_sum}")
    return records


def write_inventory_text(path, inventory, states):
    lines = [STATUS_LABEL, "", "Snapshot inventory", "------------------"]
    for key, value in inventory.items():
        lines.append(f"{key}: {value}")
    lines.extend(["", "Sample counts by RAQI selection", "--------------------------------"])
    for key, _, label in RAQI_SELECTIONS:
        counts = states[key]["counts"]
        lines.append(f"{label}: {json.dumps(counts, sort_keys=True)}")
        lines.append(f"  AutoSnow counts on four-product sample: {states[key]['surface_counts']}")
        lines.append(f"  Temperature counts on four-product sample: {states[key]['temperature_counts']}")
    path.write_text("\n".join(lines) + "\n")


def write_summary(path, inventory, states, active_selections, figures, tables, snapshot_time):
    lines = [
        f"# {STATUS_LABEL}", "",
        "## Scope", "",
        (f"This analysis uses a fixed startup snapshot of **{inventory['files_used']}** completed "
         f"matchup files available at **{snapshot_time} UTC**. It is not a final climatology and "
         "does not include files completed after the snapshot."), "",
        f"- Orbit range: {inventory['first_orbit']}–{inventory['last_orbit']}",
        f"- Date range: {inventory['first_date']}–{inventory['last_date']}",
        f"- Total native footprints inspected: {inventory['total_footprints']:,}",
        f"- V7 missing fraction: {inventory['V7_missing_fraction']:.6f}",
        f"- V8 missing fraction: {inventory['V8_missing_fraction']:.6f}", "",
        "## Sample definitions", "",
    ]
    for name, definition in MASK_DEFINITIONS.items():
        lines.append(f"- `{name}`: {definition}")
    lines.extend(["", f"Wet threshold: **{WET_THRESHOLD}**. For MRMS this is 0.1 mm in the selected "
                  "one-hour accumulation; for GPROF it is 0.1 mm h⁻¹. Categorical events use an "
                  "inclusive `>=` threshold.", ""])
    lines.extend(["## Sample counts", ""])
    for key, _, label in RAQI_SELECTIONS:
        count = states[key]["counts"]
        generated = any(key == item[0] for item in active_selections)
        lines.append(f"- {label}: V7/V8/MRMS N={count['v7v8_common']:,}; four-product N={count['four_product_common']:,}; "
                     f"MRMS-wet four-product N={count['mrms_wet_four']:,}; full outputs generated={generated}.")
    lines.extend([
        "", "## Interpretation constraints", "",
        "- MRMS and ERA5 are selected one-hour accumulations, whereas GPROF surface precipitation is a retrieval rate. "
        "Means, biases, and scatterplots therefore compare stored native quantities with different temporal support; "
        "they must not be described as strictly time-equivalent accumulation errors.",
        "- The archive was still growing. Geographic, seasonal, surface, and storm sampling are incomplete and may be uneven.",
        "- Footprints are not independent because neighboring GMI samples and repeated weather systems are spatially correlated.",
        "- RAQI sensitivity results describe different quality-selected populations and should not be interpreted as causal effects.",
        "- Sparse temperature or AutoSnow groups are reported but not interpreted; PDF generation uses the configured minimum N.",
        "", "## Scatterplot method", "",
        "All three product-versus-MRMS panels use every footprint in the identical four-product mask. Density is accumulated "
        "as a two-dimensional histogram in `log1p(MRMS)`/`log1p(product)` space, so zeros are retained and no point subsampling is used. "
        "Panel limits and density normalization are identical within each figure.",
        "", "## Outputs", "",
        f"- Figures: {len(figures)} files under `figures/`.",
        f"- Tables: {len(tables)} CSV files under `tables/`.",
        "- Exact masks and sample counts: `tables/analysis_mask_counts.csv`.",
        "- Reuse provenance: `reused_code_manifest.md`.",
        "- Validation checks: `tables/validation_checks.csv`.",
    ])
    path.write_text("\n".join(lines) + "\n")


def setup_logger(log_path):
    logger = logging.getLogger("preliminary_gmi_analysis")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler); logger.addHandler(stream_handler)
    return logger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    parser.add_argument("--output-dir", type=Path,
                        help="Exact output directory; default is preliminary_analysis_<UTC timestamp>")
    parser.add_argument("--max-files", type=int, help="Testing only: cap the startup snapshot")
    args = parser.parse_args()

    snapshot_time = dt.datetime.now(dt.timezone.utc)
    stamp = snapshot_time.strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or args.output_parent / f"preliminary_analysis_{stamp}"
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    logs_dir = output_dir / "logs"
    for directory in (figures_dir, tables_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=False if directory == figures_dir else True)
    logger = setup_logger(logs_dir / "preliminary_analysis.log")
    logger.info(STATUS_LABEL)
    logger.info("Input root: %s", args.input_root)
    logger.info("Output directory: %s", output_dir)

    candidates = sorted(path for path in args.input_root.rglob("*.nc")
                        if not path.name.endswith(".partial") and ".partial" not in path.name)
    if args.max_files is not None:
        candidates = candidates[:args.max_files]
    snapshot_records = [{"path": str(path), "size_bytes": path.stat().st_size,
                         "mtime_utc": dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc).isoformat()}
                        for path in candidates]
    write_csv(snapshot_records, tables_dir / "startup_file_snapshot.csv")
    logger.info("Startup snapshot contains %d candidate completed NetCDF files", len(candidates))

    states = {key: new_selection_state() for key, _, _ in RAQI_SELECTIONS}
    used_paths, file_rows, skipped_rows = [], [], []
    total_footprints = v7_missing = v8_missing = mrms_flag_mismatches = 0
    orbits, dates = [], []

    for index, path in enumerate(candidates, 1):
        try:
            arrays, attrs = load_required_arrays(path)
            size = len(arrays["surfacePrecipitation_V7"])
            orbit, date, code_version = orbit_metadata(path, attrs)
            used_paths.append(path); orbits.append(orbit); dates.append(date)
            total_footprints += size
            v7_missing += int(np.count_nonzero(~np.isfinite(arrays["surfacePrecipitation_V7"])))
            v8_missing += int(np.count_nonzero(~np.isfinite(arrays["surfacePrecipitation_V8"])))
            mrms_flag_mismatches += int(np.count_nonzero(
                (arrays["mrms_valid_flag"] == 1) != np.isfinite(arrays["MRMS_Pass2"])
            ))
            file_rows.append({"path": str(path), "orbit": orbit, "date": date,
                              "footprint_count": size, "code_version": code_version, "status": "used"})
            for key, threshold, _ in RAQI_SELECTIONS:
                update_selection_state(states[key], arrays, threshold)
        except Exception as exc:
            skipped_rows.append({"path": str(path), "reason": f"{type(exc).__name__}: {exc}"})
            logger.warning("Skipping %s: %s", path, exc)
        if index % 10 == 0 or index == len(candidates):
            logger.info("Primary pass: %d/%d candidate files; %d usable", index, len(candidates), len(used_paths))

    if not used_paths:
        raise SystemExit("No readable completed matchup files contained every required variable")

    active_selections = list(RAQI_SELECTIONS[:2])
    if states["raqi_ge_0p8"]["counts"]["v7v8_common"] >= MIN_FULL_RAQI_08_SAMPLE:
        active_selections.append(RAQI_SELECTIONS[2])
    logger.info("RAQI >= 0.8 full analysis adequate=%s (N=%d, minimum=%d)",
                len(active_selections) == 3,
                states["raqi_ge_0p8"]["counts"]["v7v8_common"], MIN_FULL_RAQI_08_SAMPLE)

    scatter_histograms = make_scatter_histograms(used_paths, active_selections, states, logger)

    figure_paths, table_paths = [], []
    for key, _, label in active_selections:
        state = states[key]
        means = mean_records(state, label)
        path = tables_dir / f"mean_precipitation_{key}.csv"; write_csv(means, path); table_paths.append(path)
        fig_path = figures_dir / f"mean_precipitation_{key}.png"; plot_means(means, label, fig_path); figure_paths.append(fig_path)

        pdf = write_csv(pdf_records(state, label), tables_dir / f"precipitation_pdfs_{key}.csv")
        table_paths.append(tables_dir / f"precipitation_pdfs_{key}.csv")
        fig_path = figures_dir / f"precipitation_pdfs_{key}.png"; plot_pdf(pdf, label, fig_path); figure_paths.append(fig_path)

        continuous = continuous_records(state, label)
        path = tables_dir / f"continuous_metrics_{key}.csv"; write_csv(continuous, path); table_paths.append(path)
        categorical = categorical_records(state, label)
        path = tables_dir / f"categorical_metrics_{key}.csv"; write_csv(categorical, path); table_paths.append(path)
        fig_path = figures_dir / f"roebber_performance_{key}.png"; plot_roebber(categorical, label, fig_path); figure_paths.append(fig_path)

        fig_path = figures_dir / f"taylor_diagram_{key}.png"; plot_taylor(state, label, fig_path); figure_paths.append(fig_path)
        for sample, sample_label in (("all", "all-valid four-product common sample"),
                                     ("mrms_wet", "MRMS-wet >= 0.1 four-product common sample")):
            fig_path = figures_dir / f"scatter_mrms_products_{sample}_{key}.png"
            plot_scatter(scatter_histograms[key][sample], state["pair_four"][sample],
                         label, sample_label, fig_path)
            figure_paths.append(fig_path)

        temperature = temperature_records(state, label)
        path = tables_dir / f"temperature_means_{key}.csv"; write_csv(temperature, path); table_paths.append(path)
        fig_path = figures_dir / f"temperature_means_{key}.png"; plot_temperature_means(temperature, label, fig_path); figure_paths.append(fig_path)
        fig_path = figures_dir / f"temperature_pdfs_{key}.png"
        plot_temperature_pdfs(pdf, state, label, fig_path)
        if fig_path.exists(): figure_paths.append(fig_path)

    first_orbit, last_orbit = min(orbits), max(orbits)
    first_date, last_date = min(dates), max(dates)
    inventory = {
        "snapshot_time_utc": snapshot_time.isoformat(),
        "candidate_files_snapshotted": len(candidates),
        "files_used": len(used_paths),
        "files_skipped": len(skipped_rows),
        "first_orbit": first_orbit, "last_orbit": last_orbit,
        "first_date": first_date, "last_date": last_date,
        "total_footprints": total_footprints,
        "V7_missing_fraction": safe_divide(v7_missing, total_footprints),
        "V8_missing_fraction": safe_divide(v8_missing, total_footprints),
        "mrms_valid_flag_mismatch_count": mrms_flag_mismatches,
        "RAQI_ge_0p8_full_analysis_minimum_N": MIN_FULL_RAQI_08_SAMPLE,
        "RAQI_ge_0p8_full_analysis_generated": len(active_selections) == 3,
    }
    inventory_records = [{"metric": key, "value": value} for key, value in inventory.items()]
    path = tables_dir / "sample_inventory.csv"; write_csv(inventory_records, path); table_paths.append(path)
    path = tables_dir / "file_inventory.csv"; write_csv(file_rows, path); table_paths.append(path)
    path = tables_dir / "skipped_files.csv"; write_csv(skipped_rows, path); table_paths.append(path)
    path = tables_dir / "analysis_mask_counts.csv"; write_csv(build_mask_count_records(states, active_selections), path); table_paths.append(path)
    config_records = ([{"configuration": key, "value": value} for key, value in FORMULAS.items()] +
                      [{"configuration": f"mask:{key}", "value": value} for key, value in MASK_DEFINITIONS.items()])
    path = tables_dir / "analysis_configuration.csv"; write_csv(config_records, path); table_paths.append(path)
    checks = validation_records(states, active_selections)
    checks.append({"check": "mrms_valid_flag equals finite(MRMS_Pass2) in all used files",
                   "passed": mrms_flag_mismatches == 0,
                   "detail": f"mismatches={mrms_flag_mismatches}"})
    path = tables_dir / "validation_checks.csv"; check_frame = write_csv(checks, path); table_paths.append(path)
    write_inventory_text(output_dir / "sample_inventory.txt", inventory, states)

    source_manifest = Path(__file__).with_name("reused_code_manifest.md")
    shutil.copy2(source_manifest, output_dir / "reused_code_manifest.md")
    write_summary(output_dir / "preliminary_analysis_summary.md", inventory, states,
                  active_selections, figure_paths, table_paths, snapshot_time.isoformat())

    manifest = {
        "status_label": STATUS_LABEL,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input_snapshot_utc": snapshot_time.isoformat(),
        "input_root": str(args.input_root), "output_directory": str(output_dir),
        "files_used": len(used_paths), "figures": [str(path) for path in figure_paths],
        "tables": [str(path) for path in table_paths],
        "validation_passed": bool(check_frame["passed"].all()),
    }
    (output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("Validation checks passed=%s", manifest["validation_passed"])
    logger.info("Analysis complete: %s", output_dir)
    print(json.dumps(manifest, indent=2))
    if not manifest["validation_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""
Plotting utilities for the GMI GPROF V7/V8 comparison with MRMS.

This module contains plotting functions only. Statistical calculations,
sample masking, NetCDF loading, and DataFrame preparation belong in:

    gprof_v7_v8_analysis_functions.py

Primary plotting functions:
    - plot_surface_group_means
    - plot_temperature_binned_means
    - plot_precipitation_pdf_lines
    - plot_mrms_product_scatter_density
    - plot_performance_diagram
    - plot_quantitative_metric_bars
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from matplotlib import artist
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
from matplotlib.colors import (
    ListedColormap,
    BoundaryNorm,
    LinearSegmentedColormap,
    LogNorm,
    Normalize,
    TwoSlopeNorm,
)
from matplotlib.ticker import (
    AutoMinorLocator,
    FixedLocator,
    FormatStrFormatter,
    FuncFormatter,
)


import cartopy.crs as ccrs
import cartopy.feature as cfeature

from astropy.visualization import (
    ImageNormalize,
    LogStretch,
)

from matplotlib.colors import LinearSegmentedColormap

from mpl_scatter_density import (
    ScatterDensityArtist,
)

# =============================================================================
# Default labels and plotting configuration
# =============================================================================

DEFAULT_PRODUCT_LABELS: dict[str, str] = {
    "GPROF_V7": "GPROF V7",
    "GPROF_V8": "GPROF V8",
    "MRMS": "MRMS",
    "ERA5": "ERA5",
}

DEFAULT_PRODUCT_COLORS: dict[str, str] = {
    "GPROF_V7": "tab:blue",
    "GPROF_V8": "tab:orange",
    "MRMS": "black",
    "ERA5": "tab:green",
}

DEFAULT_PRODUCT_MARKERS: dict[str, str] = {
    "GPROF_V7": "o",
    "GPROF_V8": "s",
    "MRMS": "^",
    "ERA5": "D",
}

DEFAULT_PRODUCT_LINESTYLES: dict[str, str] = {
    "GPROF_V7": "-",
    "GPROF_V8": "-",
    "MRMS": "--",
    "ERA5": "-.",
}

DEFAULT_SURFACE_LABELS: dict[str, str] = {
    "all surfaces": "All surfaces",
    "AutoSnow 0: clear water": "Clear water",
    "AutoSnow 1: snow-free land": "Snow-free land",
    "AutoSnow 2: snow-covered land": "Snow-covered land",
    "AutoSnow 3: ice-covered water": "Ice-covered water",
}

WHITE_VIRIDIS = LinearSegmentedColormap.from_list(
    "white_viridis",
    [
        (0.0, "#ffffff"),
        (1.0e-20, "#440053"),
        (0.2, "#404388"),
        (0.4, "#2a788e"),
        (0.6, "#21a784"),
        (0.8, "#78d151"),
        (1.0, "#fde624"),
    ],
    N=256,
)

# =============================================================================
# General helpers
# =============================================================================

def _resolve_mapping(
    keys: Sequence[str],
    mapping: Mapping[str, str] | None,
    default_mapping: Mapping[str, str],
) -> dict[str, str]:
    """
    Resolve labels, colors, markers, or line styles for supplied product keys.
    """

    resolved: dict[str, str] = {}

    for key in keys:
        if mapping is not None and key in mapping:
            resolved[key] = mapping[key]
        elif key in default_mapping:
            resolved[key] = default_mapping[key]
        else:
            resolved[key] = key

    return resolved


def _validate_dataframe_columns(
    data: pd.DataFrame,
    required_columns: Sequence[str],
) -> None:
    """
    Raise KeyError if required plotting columns are absent.
    """

    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise KeyError(
            "Missing required plotting columns: "
            + ", ".join(missing)
        )


def format_sample_count(value: int | float) -> str:
    """
    Format a sample count compactly.
    """

    if not np.isfinite(value):
        return "N = unavailable"

    value = int(value)

    if value >= 1_000_000:
        return f"N = {value / 1_000_000:.2f}M"

    if value >= 1_000:
        return f"N = {value / 1_000:.1f}k"

    return f"N = {value:,}"


def apply_axis_style(
    ax: plt.Axes,
    *,
    tick_fontsize: float = 11,
    label_fontsize: float = 13,
    grid: bool = True,
    minor_ticks: bool = True,
) -> None:
    """
    Apply common formatting to one Matplotlib axis.
    """

    if grid:
        ax.grid(
            True,
            linestyle="--",
            linewidth=0.6,
            alpha=0.35,
            zorder=0,
        )

    if minor_ticks:
        ax.xaxis.set_minor_locator(
            AutoMinorLocator()
        )
        ax.yaxis.set_minor_locator(
            AutoMinorLocator()
        )

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        labelsize=tick_fontsize,
        length=5,
    )

    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=2.5,
    )

    for tick_label in (
        ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        tick_label.set_fontsize(tick_fontsize)

    ax.xaxis.label.set_size(label_fontsize)
    ax.yaxis.label.set_size(label_fontsize)


def save_figure(
    fig: plt.Figure,
    output_path: str | Path | None,
    *,
    dpi: int = 150,
    close: bool = False,
    tight: bool = True,
) -> Path | None:
    """
    Save a figure if an output path is supplied.
    """

    if output_path is None:
        return None

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if tight:
        fig.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
        )
    else:
        fig.savefig(
            output_path,
            dpi=dpi,
        )

    if close:
        plt.close(fig)

    return output_path


def _add_preliminary_label(
    fig: plt.Figure,
    text: str | None,
) -> None:
    """
    Add a small preliminary-analysis label to the figure.
    """

    if not text:
        return

    fig.text(
        0.995,
        0.005,
        text,
        ha="right",
        va="bottom",
        fontsize=8,
        alpha=0.65,
    )


# =============================================================================
# Mean precipitation by surface type
# =============================================================================

def plot_surface_group_means(
    surface_mean_data: pd.DataFrame,
    *,
    product_columns: Sequence[str] = (
        "GPROF_V7",
        "GPROF_V8",
        "MRMS",
        "ERA5",
    ),
    surface_column: str = "surface_group",
    sample_count_column: str = "sample_count",
    product_labels: Mapping[str, str] | None = None,
    product_colors: Mapping[str, str] | None = None,
    surface_labels: Mapping[str, str] | None = None,
    title: str = "Mean precipitation by surface class",
    ylabel: str = "[mm h$^{-1}$]",
    units_text: str | None = None,
    annotate_sample_counts: bool = True,
    minimum_sample_count: int | None = None,
    figure_size: tuple[float, float] = (12, 6.5),
    output_path: str | Path | None = None,
    preliminary_label: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot grouped mean precipitation bars by surface class.

    Parameters
    ----------
    surface_mean_data
        Output of calculate_surface_group_means().

    minimum_sample_count
        Surface groups with fewer samples are omitted when supplied.
    """

    required_columns = [
        surface_column,
        sample_count_column,
        *product_columns,
    ]

    _validate_dataframe_columns(
        surface_mean_data,
        required_columns,
    )

    working = surface_mean_data.copy()

    if minimum_sample_count is not None:
        working = working.loc[
            working[sample_count_column]
            >= minimum_sample_count
        ].copy()

    if working.empty:
        raise ValueError(
            "No surface groups remain for plotting."
        )

    labels = _resolve_mapping(
        product_columns,
        product_labels,
        DEFAULT_PRODUCT_LABELS,
    )

    colors = _resolve_mapping(
        product_columns,
        product_colors,
        DEFAULT_PRODUCT_COLORS,
    )

    resolved_surface_labels = dict(
        DEFAULT_SURFACE_LABELS
    )

    if surface_labels is not None:
        resolved_surface_labels.update(
            surface_labels
        )

    x = np.arange(len(working))
    number_of_products = len(product_columns)

    total_width = 0.82
    bar_width = total_width / number_of_products

    fig, ax = plt.subplots(
        figsize=figure_size
    )

    for product_index, product in enumerate(
        product_columns
    ):
        offsets = (
            product_index
            - (number_of_products - 1) / 2
        ) * bar_width

        ax.bar(
            x + offsets,
            working[product].to_numpy(),
            width=bar_width * 0.92,
            label=labels[product],
            color=colors[product],
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )

    x_tick_labels = [
        resolved_surface_labels.get(
            value,
            str(value),
        )
        for value in working[surface_column]
    ]

    ax.set_xticks(x)
    ax.set_xticklabels(
        x_tick_labels,
        rotation=0,
        ha="center",
    )

    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(
            title,
            fontsize=15,
            fontweight="bold",
            pad=12,
        )

    ax.set_ylim(
        bottom=0
    )

    if units_text:
        ax.text(
            0.01,
            0.98,
            units_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            alpha=0.75,
        )

    if annotate_sample_counts:
        maximum_values = (
            working[
                list(product_columns)
            ]
            .max(axis=1)
            .to_numpy(dtype=float)
        )

        overall_maximum = float(
            np.nanmax(maximum_values)
        )

        annotation_offset = (
            0.025 * overall_maximum
            if overall_maximum > 0
            else 0.01
        )

        for position, count, maximum_value in zip(
            x,
            working[sample_count_column],
            maximum_values,
        ):
            ax.text(
                position,
                maximum_value
                + annotation_offset,
                format_sample_count(count),
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
                color="0.25",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.70,
                    "pad": 1.5,
                },
                zorder=6,
            )

        current_bottom, current_top = (
            ax.get_ylim()
        )

        required_top = (
            np.nanmax(maximum_values)
            + 4.0 * annotation_offset
        )

        ax.set_ylim(
            current_bottom,
            max(
                current_top,
                required_top,
            ),
        )

    apply_axis_style(
        ax,
        minor_ticks=False,
    )

    ax.legend(
        frameon=False,
        ncol=min(
            len(product_columns),
            4,
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.tight_layout()
    _add_preliminary_label(
        fig,
        preliminary_label,
    )

    save_figure(
        fig,
        output_path,
    )

    return fig, ax


# =============================================================================
# Temperature-binned mean precipitation
# =============================================================================

def plot_temperature_binned_means(
    temperature_mean_data: pd.DataFrame,
    *,
    product_columns: Sequence[str] = (
        "GPROF_V7",
        "GPROF_V8",
        "MRMS",
        "ERA5",
    ),
    temperature_column: str = "temperature_midpoint",
    sample_count_column: str = "sample_count",
    product_labels: Mapping[str, str] | None = None,
    product_colors: Mapping[str, str] | None = None,
    product_markers: Mapping[str, str] | None = None,
    product_linestyles: Mapping[str, str] | None = None,
    title: str | None = None,
    xlabel: str = "2-m air temperature (K)",
    ylabel: str = "[mm h$^{-1}$]",
    x_limits: tuple[float, float] | None = None,
    x_tick_interval: float = 4.0,
    figure_size: tuple[float, float] = (10, 6.5),
    output_path: str | Path | None = None,
    preliminary_label: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot mean precipitation against numeric temperature-bin midpoints.
    """

    required_columns = [
        temperature_column,
        *product_columns,
    ]

    _validate_dataframe_columns(
        temperature_mean_data,
        required_columns,
    )

    working = temperature_mean_data.copy()

    working[
        temperature_column
    ] = pd.to_numeric(
        working[temperature_column],
        errors="coerce",
    )

    working = working.dropna(
        subset=[temperature_column]
    ).sort_values(
        temperature_column
    )

    labels = _resolve_mapping(
        product_columns,
        product_labels,
        DEFAULT_PRODUCT_LABELS,
    )

    colors = _resolve_mapping(
        product_columns,
        product_colors,
        DEFAULT_PRODUCT_COLORS,
    )

    markers = _resolve_mapping(
        product_columns,
        product_markers,
        DEFAULT_PRODUCT_MARKERS,
    )

    linestyles = _resolve_mapping(
        product_columns,
        product_linestyles,
        DEFAULT_PRODUCT_LINESTYLES,
    )

    x_values = working[
        temperature_column
    ].to_numpy(dtype=float)

    fig, ax = plt.subplots(
        figsize=figure_size
    )

    for product in product_columns:
        ax.plot(
            x_values,
            working[product],
            color=colors[product],
            marker=markers[product],
            linestyle=linestyles[product],
            linewidth=2.0,
            markersize=5.5,
            label=labels[product],
            zorder=3,
        )

    if x_limits is None:
        x_min = float(
            np.nanmin(x_values)
        )

        x_max = float(
            np.nanmax(x_values)
        )

        ax.set_xlim(
            x_min,
            x_max,
        )

    else:
        x_min = float(
            x_limits[0]
        )

        x_max = float(
            x_limits[1]
        )

        ax.set_xlim(
            x_min,
            x_max,
        )

    first_tick = (
        np.ceil(
            x_min
            / x_tick_interval
        )
        * x_tick_interval
    )

    tick_values = np.arange(
        first_tick,
        x_max
        + 0.5 * x_tick_interval,
        x_tick_interval,
    )

    ax.xaxis.set_major_locator(
        FixedLocator(
            tick_values
        )
    )

    ax.xaxis.set_major_formatter(
        FuncFormatter(
            lambda value, position:
            f"{int(round(value))}"
        )
    )

    ax.xaxis.set_minor_locator(
        AutoMinorLocator(2)
    )

    ax.set_xlabel(
        xlabel,
        fontsize=13,
        fontweight="bold",
    )

    ax.set_ylabel(
        ylabel,
        fontsize=13,
        fontweight="bold",
    )

    ax.set_title(
        title,
        fontsize=15,
        fontweight="bold",
        pad=12,
    )

    ax.set_ylim(
        bottom=0
    )

    ax.grid(
        True,
        which="major",
        linestyle="--",
        linewidth=0.6,
        alpha=0.35,
    )

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        labelsize=11,
        length=5,
    )

    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=2.5,
    )

    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper left",
    )

    fig.tight_layout()

    _add_preliminary_label(
        fig,
        preliminary_label,
    )

    save_figure(
        fig,
        output_path,
    )

    return fig, ax


# =============================================================================
# Precipitation PDFs
# =============================================================================

def plot_precipitation_pdfs(
    pdf_data: pd.DataFrame,
    *,
    metric_column: str = "occurrence_percent",
    product_column: str = "product",
    bin_label_column: str = "bin_label",
    bin_index_column: str = "bin_index",
    product_order: Sequence[str] = (
        "GPROF_V7",
        "GPROF_V8",
        "MRMS",
        "ERA5",
    ),
    product_labels: Mapping[str, str] | None = None,
    product_colors: Mapping[str, str] | None = None,
    product_markers: Mapping[str, str] | None = None,
    product_linestyles: Mapping[str, str] | None = None,
    title: str = "Precipitation distribution",
    ylabel: str | None = None,
    log_y: bool = True,
    show_zero_bin: bool = True,
    x_tick_rotation: float = 45,
    figure_size: tuple[float, float] = (12, 6.5),
    output_path: str | Path | None = None,
    preliminary_label: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot precipitation occurrence or volume distributions.

    Expected input is the output from calculate_product_pdf_table().
    """

    required_columns = [
        product_column,
        bin_label_column,
        bin_index_column,
        metric_column,
    ]

    _validate_dataframe_columns(
        pdf_data,
        required_columns,
    )

    working = pdf_data.copy()

    if not show_zero_bin:
        working = working.loc[
            working[bin_label_column]
            != "exact zero"
        ].copy()

    labels = _resolve_mapping(
        product_order,
        product_labels,
        DEFAULT_PRODUCT_LABELS,
    )

    colors = _resolve_mapping(
        product_order,
        product_colors,
        DEFAULT_PRODUCT_COLORS,
    )

    markers = _resolve_mapping(
        product_order,
        product_markers,
        DEFAULT_PRODUCT_MARKERS,
    )

    linestyles = _resolve_mapping(
        product_order,
        product_linestyles,
        DEFAULT_PRODUCT_LINESTYLES,
    )

    ordered_bins = (
        working[
            [bin_index_column, bin_label_column]
        ]
        .drop_duplicates()
        .sort_values(bin_index_column)
    )

    x_positions = np.arange(
        len(ordered_bins)
    )

    bin_to_position = {
        row[bin_label_column]: position
        for position, (_, row) in enumerate(
            ordered_bins.iterrows()
        )
    }

    fig, ax = plt.subplots(
        figsize=figure_size
    )

    for product in product_order:
        subset = working.loc[
            working[product_column] == product
        ].copy()

        if subset.empty:
            continue

        subset["x_position"] = subset[
            bin_label_column
        ].map(bin_to_position)

        subset = subset.sort_values(
            "x_position"
        )

        y_values = subset[
            metric_column
        ].to_numpy(dtype=float)

        if log_y:
            y_values = np.where(
                y_values > 0,
                y_values,
                np.nan,
            )

        ax.plot(
            subset["x_position"],
            y_values,
            marker=markers[product],
            linestyle=linestyles[product],
            color=colors[product],
            linewidth=2.0,
            markersize=5.5,
            label=labels[product],
            zorder=3,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        ordered_bins[
            bin_label_column
        ],
        rotation=x_tick_rotation,
        ha="right",
    )

    if ylabel is None:
        if metric_column == "volume_percent":
            ylabel = (
                "Contribution to precipitation volume (%)"
            )
        else:
            ylabel = "Occurrence frequency (%)"

    ax.set_xlabel("Precipitation bin")
    ax.set_ylabel(ylabel)

    ax.set_title(
        title,
        fontsize=15,
        fontweight="bold",
        pad=12,
    )

    if log_y:
        ax.set_yscale("log")

    apply_axis_style(
        ax,
        minor_ticks=False,
    )

    ax.legend(
        frameon=False,
        ncol=2,
        loc="best",
    )

    fig.tight_layout()
    _add_preliminary_label(
        fig,
        preliminary_label,
    )

    save_figure(
        fig,
        output_path,
    )

    return fig, ax


def plot_occurrence_and_volume_pdfs(
    pdf_data: pd.DataFrame,
    *,
    product_order: Sequence[str] = (
        "GPROF_V7",
        "GPROF_V8",
        "MRMS",
        "ERA5",
    ),
    product_labels: Mapping[str, str] | None = None,
    product_colors: Mapping[str, str] | None = None,
    title: str = "Precipitation distributions",
    log_y: bool = True,
    show_zero_bin: bool = True,
    figure_size: tuple[float, float] = (13, 9),
    output_path: str | Path | None = None,
    preliminary_label: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot occurrence-frequency and precipitation-volume distributions together.
    """

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=figure_size,
        sharex=True,
    )

    metrics = [
        (
            "occurrence_percent",
            "Occurrence frequency (%)",
            "(a) Occurrence distribution",
        ),
        (
            "volume_percent",
            "Contribution to precipitation volume (%)",
            "(b) Precipitation-volume distribution",
        ),
    ]

    working = pdf_data.copy()

    if not show_zero_bin:
        working = working.loc[
            working["bin_label"]
            != "exact zero"
        ].copy()

    labels = _resolve_mapping(
        product_order,
        product_labels,
        DEFAULT_PRODUCT_LABELS,
    )

    colors = _resolve_mapping(
        product_order,
        product_colors,
        DEFAULT_PRODUCT_COLORS,
    )

    ordered_bins = (
        working[
            ["bin_index", "bin_label"]
        ]
        .drop_duplicates()
        .sort_values("bin_index")
    )

    positions = np.arange(
        len(ordered_bins)
    )

    bin_to_position = {
        label: position
        for position, label in enumerate(
            ordered_bins["bin_label"]
        )
    }

    for ax, (
        metric_column,
        ylabel,
        panel_title,
    ) in zip(axes, metrics):

        for product in product_order:
            subset = working.loc[
                working["product"] == product
            ].copy()

            if subset.empty:
                continue

            subset["x_position"] = subset[
                "bin_label"
            ].map(bin_to_position)

            subset = subset.sort_values(
                "x_position"
            )

            y_values = subset[
                metric_column
            ].to_numpy(dtype=float)

            if log_y:
                y_values = np.where(
                    y_values > 0,
                    y_values,
                    np.nan,
                )

            ax.plot(
                subset["x_position"],
                y_values,
                marker=DEFAULT_PRODUCT_MARKERS.get(
                    product,
                    "o",
                ),
                linestyle=(
                    DEFAULT_PRODUCT_LINESTYLES.get(
                        product,
                        "-",
                    )
                ),
                color=colors[product],
                linewidth=2.0,
                markersize=5,
                label=labels[product],
            )

        ax.set_ylabel(ylabel)

        ax.set_title(
            panel_title,
            loc="left",
            fontsize=13,
            fontweight="bold",
        )

        if log_y:
            ax.set_yscale("log")

        apply_axis_style(
            ax,
            minor_ticks=False,
        )

    axes[-1].set_xticks(positions)
    axes[-1].set_xticklabels(
        ordered_bins["bin_label"],
        rotation=45,
        ha="right",
    )

    axes[-1].set_xlabel(
        "Precipitation bin"
    )

    axes[0].legend(
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
    )

    fig.suptitle(
        title,
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    fig.tight_layout()
    _add_preliminary_label(
        fig,
        preliminary_label,
    )

    save_figure(
        fig,
        output_path,
    )

    return fig, axes


def plot_precipitation_pdf_lines(
    pdf_data: pd.DataFrame,
    *,
    metric_column: str,
    product_order: Sequence[str] = (
        "GPROF_V7",
        "GPROF_V8",
        "MRMS",
        "ERA5",
    ),
    product_labels: Mapping[str, str] | None = None,
    product_colors: Mapping[str, str] | None = None,
    product_markers: Mapping[str, str] | None = None,
    product_linestyles: Mapping[str, str] | None = None,
    title: str = "Precipitation intensity distribution",
    ylabel: str | None = None,
    show_zero_bin: bool = False,
    y_limits: tuple[float, float] | None = None,
    figure_size: tuple[float, float] = (9.5, 6.0),
    output_path: str | Path | None = None,
    preliminary_label: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot line-based precipitation PDFs using discrete intensity classes.

    The input should be generated by calculate_product_pdf_table().
    """

    required_columns = [
        "product",
        "bin_index",
        "bin_label",
        metric_column,
    ]

    _validate_dataframe_columns(
        pdf_data,
        required_columns,
    )

    working = pdf_data.copy()

    if not show_zero_bin:
        working = working.loc[
            working["bin_label"] != "exact zero"
        ].copy()

    labels = _resolve_mapping(
        product_order,
        product_labels,
        DEFAULT_PRODUCT_LABELS,
    )

    colors = _resolve_mapping(
        product_order,
        product_colors,
        DEFAULT_PRODUCT_COLORS,
    )

    markers = _resolve_mapping(
        product_order,
        product_markers,
        DEFAULT_PRODUCT_MARKERS,
    )

    linestyles = _resolve_mapping(
        product_order,
        product_linestyles,
        DEFAULT_PRODUCT_LINESTYLES,
    )

    ordered_bins = (
        working[
            [
                "bin_index",
                "bin_label",
            ]
        ]
        .drop_duplicates()
        .sort_values("bin_index")
    )

    ordered_labels = ordered_bins[
        "bin_label"
    ].tolist()

    x_positions = np.arange(
        len(ordered_labels)
    )

    bin_position = {
        label: index
        for index, label in enumerate(
            ordered_labels
        )
    }

    fig, ax = plt.subplots(
        figsize=figure_size
    )

    for product in product_order:
        subset = working.loc[
            working["product"] == product
        ].copy()

        if subset.empty:
            continue

        subset["x_position"] = subset[
            "bin_label"
        ].map(bin_position)

        subset = subset.sort_values(
            "x_position"
        )

        ax.plot(
            subset["x_position"],
            subset[metric_column],
            color=colors[product],
            marker=markers[product],
            linestyle=linestyles[product],
            linewidth=2.2,
            markersize=6.5,
            label=labels[product],
            zorder=3,
        )

    ax.set_xticks(
        x_positions
    )

    ax.set_xticklabels(
        ordered_labels,
        rotation=0,
        ha="center",
    )

    ax.set_xlabel(
        "Precipitation intensity class (mm h$^{-1}$)",
        fontsize=13,
        fontweight="bold",
    )

    if ylabel is None:
        if metric_column == "volume_percent":
            ylabel = "PDF by volume (%)"
        else:
            ylabel = "PDF by occurrence (%)"

    ax.set_ylabel(
        ylabel,
        fontsize=13,
        fontweight="bold",
    )

    ax.set_title(
        title,
        fontsize=15,
        fontweight="bold",
        pad=12,
    )

    if y_limits is not None:
        ax.set_ylim(
            y_limits
        )
    else:
        ax.set_ylim(
            bottom=0
        )

    ax.grid(
        linestyle="--",
        linewidth=0.6,
        alpha=0.35,
    )

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        labelsize=11,
        length=5,
    )

    ax.legend(
        frameon=False,
        ncol=2,
        loc="best",
    )

    fig.tight_layout()

    _add_preliminary_label(
        fig,
        preliminary_label,
    )

    save_figure(
        fig,
        output_path,
    )

    return fig, ax

def plot_pdf_bundle(
    pdf_dict: Mapping[str, pd.DataFrame],
    *,
    product_order: Sequence[str] = (
        "GPROF_V7",
        "GPROF_V8",
        "MRMS",
        "ERA5",
    ),
    product_labels: Mapping[str, str] | None = None,
    product_colors: Mapping[str, str] | None = None,
    product_markers: Mapping[str, str] | None = None,
    product_linestyles: Mapping[str, str] | None = None,
    pdf_kind: str = "pdfc",
    title: str = "Precipitation intensity PDF",
    ylabel: str | None = None,
    xlabel: str = "[mm h$^{-1}$]",
    linewidth: float = 2.5,
    markersize: float = 6.5,
    figure_size: tuple[float, float] = (9.5, 6.0),
    output_path: str | Path | None = None,
    preliminary_label: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot product precipitation PDFs from a dictionary of PDF DataFrames.

    pdf_kind must be:
        'pdfc' for occurrence
        'pdfv' for volume
    """

    if pdf_kind not in {
        "pdfc",
        "pdfv",
    }:
        raise ValueError(
            "pdf_kind must be 'pdfc' or 'pdfv'."
        )

    labels = _resolve_mapping(
        product_order,
        product_labels,
        DEFAULT_PRODUCT_LABELS,
    )

    colors = _resolve_mapping(
        product_order,
        product_colors,
        DEFAULT_PRODUCT_COLORS,
    )

    markers = _resolve_mapping(
        product_order,
        product_markers,
        DEFAULT_PRODUCT_MARKERS,
    )

    linestyles = _resolve_mapping(
        product_order,
        product_linestyles,
        DEFAULT_PRODUCT_LINESTYLES,
    )

    fig, ax = plt.subplots(
        figsize=figure_size
    )

    reference_product = next(
        product
        for product in product_order
        if product in pdf_dict
    )

    reference_pdf = pdf_dict[
        reference_product
    ]

    x_values = reference_pdf[
        "bin"
    ].to_numpy(dtype=float)

    for product in product_order:

        if product not in pdf_dict:
            continue

        product_pdf = pdf_dict[
            product
        ]

        ax.plot(
            product_pdf["bin"],
            product_pdf[pdf_kind],
            color=colors[product],
            marker=markers[product],
            linestyle=linestyles[product],
            linewidth=linewidth,
            markersize=markersize,
            label=labels[product],
            zorder=3,
        )

    ax.set_xscale(
        "log"
    )

    ax.xaxis.set_major_locator(
        FixedLocator(
            x_values.tolist()
        )
    )

    ax.xaxis.set_major_formatter(
        FuncFormatter(
            lambda value, position:
            (
                f"{value:g}"
                if value < 1
                else f"{int(round(value))}"
            )
        )
    )

    ax.xaxis.set_minor_locator(
        FixedLocator([])
    )

    ax.tick_params(
        axis="x",
        which="minor",
        bottom=False,
        top=False,
    )

    ax.set_xlabel(
        xlabel,
        fontsize=13,
        fontweight="bold",
    )

    if ylabel is None:
        ylabel = (
            "PDF by occurrence (%)"
            if pdf_kind == "pdfc"
            else "PDF by volume (%)"
        )

    ax.set_ylabel(
        ylabel,
        fontsize=13,
        fontweight="bold",
    )

    ax.set_title(
        title,
        fontsize=15,
        fontweight="bold",
        pad=12,
    )

    ax.set_ylim(
        bottom=0
    )

    ax.grid(
        True,
        which="major",
        linestyle="--",
        linewidth=0.6,
        alpha=0.45,
    )

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        labelsize=11,
        length=6,
    )

    ax.legend(
        frameon=False,
        ncol=2,
        loc="best",
    )

    fig.tight_layout()

    _add_preliminary_label(
        fig,
        preliminary_label,
    )

    save_figure(
        fig,
        output_path,
    )

    return fig, ax
# =============================================================================
# Scatter and density plots against MRMS
# =============================================================================

def _finite_pair(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract paired finite x and y values.
    """

    _validate_dataframe_columns(
        data,
        [x_column, y_column],
    )

    x = pd.to_numeric(
        data[x_column],
        errors="coerce",
    ).to_numpy(dtype=float)

    y = pd.to_numeric(
        data[y_column],
        errors="coerce",
    ).to_numpy(dtype=float)

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    return x[valid], y[valid]


def _calculate_plot_metrics(
    reference: np.ndarray,
    product: np.ndarray,
) -> dict[str, float | int]:
    """
    Calculate basic metrics for scatterplot annotation.
    """

    if reference.size == 0:
        return {
            "N": 0,
            "bias": np.nan,
            "MAE": np.nan,
            "RMSE": np.nan,
            "CC": np.nan,
        }

    residual = product - reference

    if (
        reference.size >= 2
        and np.std(reference) > 0
        and np.std(product) > 0
    ):
        correlation = float(
            np.corrcoef(
                reference,
                product,
            )[0, 1]
        )
    else:
        correlation = np.nan

    return {
        "N": int(reference.size),
        "bias": float(
            np.mean(residual)
        ),
        "MAE": float(
            np.mean(
                np.abs(residual)
            )
        ),
        "RMSE": float(
            np.sqrt(
                np.mean(
                    residual ** 2
                )
            )
        ),
        "CC": correlation,
    }


def _format_metrics_annotation(
    metrics: Mapping[str, float | int],
    *,
    precision: int = 3,
) -> str:
    """
    Format scatterplot metrics.
    """

    return (
        f"{format_sample_count(metrics['N'])}\n"
        f"Bias = {metrics['bias']:.{precision}f}\n"
        f"MAE = {metrics['MAE']:.{precision}f}\n"
        f"RMSE = {metrics['RMSE']:.{precision}f}\n"
        f"CC = {metrics['CC']:.{precision}f}"
    )


def _transform_precipitation(
    values: np.ndarray,
    transform: str,
) -> np.ndarray:
    """
    Apply an explicitly selected precipitation-axis transformation.
    """

    transform = transform.lower()

    if transform == "linear":
        return values

    if transform == "log1p":
        return np.log1p(
            np.clip(
                values,
                a_min=0,
                a_max=None,
            )
        )

    raise ValueError(
        "transform must be either 'linear' or 'log1p'."
    )


def plot_mrms_product_density_panels(
    data: pd.DataFrame,
    *,
    products: Sequence[str] = (
        "GPROF_V7",
        "GPROF_V8",
        "ERA5",
    ),
    reference_column: str = "MRMS",
    product_labels: Mapping[str, str] | None = None,
    transform: str = "log1p",
    gridsize: int = 90,
    min_count: int = 1,
    axis_limits: tuple[float, float] | None = None,
    percentile_limit: float = 99.8,
    colorbar_label: str = "Footprint count",
    title: str | None = None,
    wet_reference_threshold: float | None = None,
    figure_size: tuple[float, float] = (16, 5.5),
    output_path: str | Path | None = None,
    preliminary_label: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot MRMS-versus-product hexbin density panels.

    MRMS is always placed on the x-axis.

    transform
        'linear' or 'log1p'.

    wet_reference_threshold
        When supplied, only MRMS values greater than or equal to this threshold
        are retained.
    """

    _validate_dataframe_columns(
        data,
        [reference_column, *products],
    )

    labels = _resolve_mapping(
        products,
        product_labels,
        DEFAULT_PRODUCT_LABELS,
    )

    working = data.copy()

    if wet_reference_threshold is not None:
        working = working.loc[
            working[reference_column]
            >= wet_reference_threshold
        ].copy()

    if working.empty:
        raise ValueError(
            "No samples remain for density plotting."
        )

    all_values = []

    for column in (
        reference_column,
        *products,
    ):
        values = pd.to_numeric(
            working[column],
            errors="coerce",
        ).to_numpy(dtype=float)

        values = values[
            np.isfinite(values)
            & (values >= 0)
        ]

        all_values.append(values)

    if axis_limits is None:
        pooled = np.concatenate(
            [
                values
                for values in all_values
                if values.size > 0
            ]
        )

        upper_limit = float(
            np.nanpercentile(
                pooled,
                percentile_limit,
            )
        )

        if not np.isfinite(upper_limit):
            upper_limit = 1.0

        if upper_limit <= 0:
            upper_limit = 1.0

        axis_limits = (
            0.0,
            upper_limit,
        )

    transformed_limits = (
        float(
            _transform_precipitation(
                np.array([axis_limits[0]]),
                transform,
            )[0]
        ),
        float(
            _transform_precipitation(
                np.array([axis_limits[1]]),
                transform,
            )[0]
        ),
    )

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(products),
        figsize=figure_size,
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    axes = axes.ravel()
    hexbin_artist = None

    for ax, product in zip(
        axes,
        products,
    ):
        reference, product_values = (
            _finite_pair(
                working,
                reference_column,
                product,
            )
        )

        nonnegative = (
            (reference >= 0)
            & (product_values >= 0)
        )

        reference = reference[nonnegative]
        product_values = product_values[
            nonnegative
        ]

        metrics = _calculate_plot_metrics(
            reference,
            product_values,
        )

        transformed_reference = (
            _transform_precipitation(
                reference,
                transform,
            )
        )

        transformed_product = (
            _transform_precipitation(
                product_values,
                transform,
            )
        )

        hexbin_artist = ax.hexbin(
            transformed_reference,
            transformed_product,
            gridsize=gridsize,
            mincnt=min_count,
            norm=LogNorm(),
            cmap="viridis",
            extent=(
                transformed_limits[0],
                transformed_limits[1],
                transformed_limits[0],
                transformed_limits[1],
            ),
            linewidths=0,
        )

        ax.plot(
            transformed_limits,
            transformed_limits,
            linestyle="--",
            linewidth=1.4,
            color="black",
            zorder=4,
            label="1:1",
        )

        ax.set_xlim(
            transformed_limits
        )

        ax.set_ylim(
            transformed_limits
        )

        ax.set_aspect(
            "equal",
            adjustable="box",
        )

        ax.set_title(
            labels[product],
            fontsize=13,
            fontweight="bold",
        )

        ax.text(
            0.04,
            0.96,
            _format_metrics_annotation(
                metrics
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
            bbox={
                "facecolor": "white",
                "edgecolor": "0.5",
                "alpha": 0.88,
                "boxstyle": "round,pad=0.35",
            },
        )

        apply_axis_style(
            ax,
            minor_ticks=False,
        )

    if transform == "log1p":
        axis_label = (
            "log1p precipitation"
        )
    else:
        axis_label = (
            "Precipitation"
        )

    for ax in axes:
        ax.set_xlabel(
            f"MRMS {axis_label}"
        )

    axes[0].set_ylabel(
        f"Product {axis_label}"
    )

    if hexbin_artist is not None:
        colorbar = fig.colorbar(
            hexbin_artist,
            ax=axes.tolist(),
            fraction=0.025,
            pad=0.02,
        )

        colorbar.set_label(
            colorbar_label,
            fontsize=11,
        )

    subtitle = ""

    if wet_reference_threshold is not None:
        subtitle = (
            f" — MRMS ≥ "
            f"{wet_reference_threshold:g}"
        )

    fig.suptitle(
        title + subtitle,
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    fig.subplots_adjust(
        left=0.06,
        right=0.91,
        bottom=0.14,
        top=0.86,
        wspace=0.18,
    )

    _add_preliminary_label(
        fig,
        preliminary_label,
    )

    save_figure(
        fig,
        output_path,
    )

    return fig, axes


# =============================================================================
# Performance / Roebber diagram
# =============================================================================

def draw_performance_background(
    ax: plt.Axes,
    *,
    contour_label_fontsize: float = 9,
    bias_label_fontsize: float = 9,
    axis_label_fontsize: float = 13,
    tick_fontsize: float = 11,
    csi_color: str = "brown",
    bias_color: str = "steelblue",
    show_csi_label: bool = True,
    show_bias_label: bool = True,
) -> None:
    """
    Draw CSI isolines and frequency-bias lines in POD versus SR space.

    x-axis:
        Success Ratio, SR = 1 - FAR

    y-axis:
        Probability of Detection, POD

    CSI:
        CSI = 1 / (1/SR + 1/POD - 1)

    Frequency bias:
        Bias = POD / SR
    """

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_autoscale_on(False)

    success_ratio = np.linspace(
        0.001,
        0.999,
        600,
    )

    pod = np.linspace(
        0.001,
        0.999,
        600,
    )

    SR, POD = np.meshgrid(
        success_ratio,
        pod,
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        CSI = 1.0 / (
            1.0 / SR
            + 1.0 / POD
            - 1.0
        )

    CSI = np.where(
        np.isfinite(CSI)
        & (CSI >= 0.0)
        & (CSI <= 1.0),
        CSI,
        np.nan,
    )

    csi_levels = np.arange(
        0.1,
        1.0,
        0.1,
    )

    contour_set = ax.contour(
        SR,
        POD,
        CSI,
        levels=csi_levels,
        colors=csi_color,
        linewidths=0.9,
        alpha=0.85,
        zorder=1,
    )

    ax.clabel(
        contour_set,
        fmt="%.1f",
        fontsize=contour_label_fontsize,
        colors=csi_color,
        inline=True,
    )

    sr_line = np.linspace(
        0.0,
        1.0,
        1000,
    )

    bias_levels = [
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        2.0,
        3.0,
        5.0,
    ]

    for frequency_bias in bias_levels:
        pod_line = (
            frequency_bias
            * sr_line
        )

        valid = (
            pod_line <= 1.0
        )

        ax.plot(
            sr_line[valid],
            pod_line[valid],
            linestyle="--",
            color=bias_color,
            linewidth=0.9,
            alpha=0.75,
            zorder=0,
        )

        if frequency_bias < 1:
            x_position = 0.96
            y_position = (
                frequency_bias
                * x_position
            )
        elif np.isclose(
            frequency_bias,
            1.0,
        ):
            x_position = 0.90
            y_position = 0.90
        else:
            y_position = 0.97
            x_position = (
                y_position
                / frequency_bias
            )

        ax.text(
            x_position,
            y_position,
            f"{frequency_bias:g}",
            fontsize=bias_label_fontsize,
            ha="center",
            va="center",
            color=bias_color,
            fontweight="bold",
            clip_on=False,
            zorder=2,
        )

    if show_bias_label:
        ax.text(
            0.40,
            0.56,
            "Frequency bias",
            transform=ax.transAxes,
            color=bias_color,
            fontsize=bias_label_fontsize + 1,
            fontweight="bold",
            ha="center",
            va="center",
            rotation=45,
            alpha=0.9,
        )

    if show_csi_label:
        ax.text(
            1.035,
            0.52,
            "CSI",
            transform=ax.transAxes,
            color=csi_color,
            fontsize=axis_label_fontsize,
            fontweight="bold",
            rotation=270,
            ha="center",
            va="center",
            clip_on=False,
        )

    major_ticks = np.arange(
        0.0,
        1.01,
        0.2,
    )

    ax.xaxis.set_major_locator(
        FixedLocator(major_ticks)
    )

    ax.yaxis.set_major_locator(
        FixedLocator(major_ticks)
    )

    ax.xaxis.set_major_formatter(
        FormatStrFormatter("%.1f")
    )

    ax.yaxis.set_major_formatter(
        FormatStrFormatter("%.1f")
    )

    ax.xaxis.set_minor_locator(
        AutoMinorLocator(4)
    )

    ax.yaxis.set_minor_locator(
        AutoMinorLocator(4)
    )

    ax.set_xlabel(
        "Success Ratio (1 − FAR)",
        fontsize=axis_label_fontsize,
        fontweight="bold",
    )

    ax.set_ylabel(
        "POD",
        fontsize=axis_label_fontsize,
        fontweight="bold",
    )

    ax.grid(
        linestyle="--",
        linewidth=0.6,
        alpha=0.3,
    )

    ax.tick_params(
        which="major",
        axis="both",
        direction="in",
        length=5,
        top=True,
        right=True,
        labelsize=tick_fontsize,
    )

    ax.tick_params(
        which="minor",
        axis="both",
        direction="in",
        length=2.5,
        top=True,
        right=True,
    )

    ax.text(
        0.03,
        0.97,
        "Higher detection",
        transform=ax.transAxes,
        fontsize=tick_fontsize - 1,
        ha="left",
        va="top",
        fontweight="bold",
        alpha=0.7,
    )

    ax.text(
        0.97,
        0.03,
        "Lower false alarm",
        transform=ax.transAxes,
        fontsize=tick_fontsize - 1,
        ha="right",
        va="bottom",
        fontweight="bold",
        alpha=0.7,
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_autoscale_on(False)


# Backward-compatible name used in your original code.
draw_perf_background = (
    draw_performance_background
)


def plot_performance_diagram(
    categorical_metrics: pd.DataFrame,
    *,
    product_column: str = "product",
    success_ratio_column: str = "success_ratio",
    pod_column: str = "POD",
    product_order: Sequence[str] = (
        "GPROF_V7",
        "GPROF_V8",
        "ERA5",
    ),
    product_labels: Mapping[str, str] | None = None,
    product_colors: Mapping[str, str] | None = None,
    product_markers: Mapping[str, str] | None = None,
    title: str = "Categorical precipitation performance",
    threshold_text: str | None = None,
    annotate_points: bool = True,
    figure_size: tuple[float, float] = (8.5, 8),
    output_path: str | Path | None = None,
    preliminary_label: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot product categorical metrics on a Roebber performance diagram.
    """

    _validate_dataframe_columns(
        categorical_metrics,
        [
            product_column,
            success_ratio_column,
            pod_column,
        ],
    )

    labels = _resolve_mapping(
        product_order,
        product_labels,
        DEFAULT_PRODUCT_LABELS,
    )

    colors = _resolve_mapping(
        product_order,
        product_colors,
        DEFAULT_PRODUCT_COLORS,
    )

    markers = _resolve_mapping(
        product_order,
        product_markers,
        DEFAULT_PRODUCT_MARKERS,
    )

    fig, ax = plt.subplots(
        figsize=figure_size
    )

    draw_performance_background(
        ax
    )

    plotted_products = []

    for product in product_order:
        subset = categorical_metrics.loc[
            categorical_metrics[
                product_column
            ]
            == product
        ]

        if subset.empty:
            continue

        row = subset.iloc[0]

        success_ratio = float(
            row[
                success_ratio_column
            ]
        )

        pod = float(
            row[pod_column]
        )

        if not (
            np.isfinite(success_ratio)
            and np.isfinite(pod)
        ):
            continue

        ax.scatter(
            success_ratio,
            pod,
            marker=markers[product],
            s=110,
            color=colors[product],
            edgecolor="black",
            linewidth=0.8,
            zorder=5,
            label=labels[product],
        )

        plotted_products.append(product)

        if annotate_points:
            ax.annotate(
                labels[product],
                xy=(
                    success_ratio,
                    pod,
                ),
                xytext=(7, 7),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
                color=colors[product],
            )

    full_title = title

    if threshold_text:
        full_title += (
            "\n" + threshold_text
        )

    ax.set_title(
        full_title,
        fontsize=15,
        fontweight="bold",
        pad=14,
    )

    if plotted_products:
        ax.legend(
            frameon=False,
            loc="lower left",
        )

    fig.tight_layout()
    _add_preliminary_label(
        fig,
        preliminary_label,
    )

    save_figure(
        fig,
        output_path,
    )

    return fig, ax


# =============================================================================
# Taylor diagram
# =============================================================================

def plot_taylor_diagram(
    metric_data: pd.DataFrame,
    *,
    product_column: str = "product",
    correlation_column: str = "correlation",
    normalized_std_column: str = "normalized_std",
    product_order: Sequence[str] = (
        "GPROF_V7",
        "GPROF_V8",
        "ERA5",
    ),
    product_labels: Mapping[str, str] | None = None,
    product_colors: Mapping[str, str] | None = None,
    product_markers: Mapping[str, str] | None = None,
    title: str = "Taylor diagram relative to MRMS",
    max_normalized_std: float | None = None,
    correlation_min: float = 0.0,
    centered_rmse_levels: Sequence[float] | None = None,
    annotate_points: bool = True,
    figure_size: tuple[float, float] = (8.5, 7.5),
    output_path: str | Path | None = None,
    preliminary_label: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a normalized Taylor diagram.

    Radius:
        Product standard deviation / MRMS standard deviation

    Angle:
        arccos(Pearson correlation)

    Reference MRMS point:
        correlation = 1
        normalized standard deviation = 1
    """

    _validate_dataframe_columns(
        metric_data,
        [
            product_column,
            correlation_column,
            normalized_std_column,
        ],
    )

    labels = _resolve_mapping(
        product_order,
        product_labels,
        DEFAULT_PRODUCT_LABELS,
    )

    colors = _resolve_mapping(
        product_order,
        product_colors,
        DEFAULT_PRODUCT_COLORS,
    )

    markers = _resolve_mapping(
        product_order,
        product_markers,
        DEFAULT_PRODUCT_MARKERS,
    )

    working = metric_data.copy()

    valid = (
        np.isfinite(
            working[
                correlation_column
            ]
        )
        & np.isfinite(
            working[
                normalized_std_column
            ]
        )
    )

    working = working.loc[
        valid
    ].copy()

    if working.empty:
        raise ValueError(
            "No finite Taylor-diagram metrics are available."
        )

    if max_normalized_std is None:
        observed_maximum = float(
            working[
                normalized_std_column
            ].max()
        )

        max_normalized_std = max(
            1.5,
            np.ceil(
                observed_maximum
                * 1.15
                * 2
            )
            / 2,
        )

    correlation_min = float(
        np.clip(
            correlation_min,
            -1.0,
            1.0,
        )
    )

    theta_max = float(
        np.arccos(
            correlation_min
        )
    )

    fig = plt.figure(
        figsize=figure_size
    )

    ax = fig.add_subplot(
        111,
        projection="polar",
    )

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    ax.set_thetamin(0)
    ax.set_thetamax(
        np.degrees(theta_max)
    )

    ax.set_ylim(
        0,
        max_normalized_std,
    )

    correlation_ticks = np.array(
        [
            1.0,
            0.99,
            0.95,
            0.9,
            0.8,
            0.7,
            0.6,
            0.5,
            0.4,
            0.3,
            0.2,
            0.1,
            0.0,
        ]
    )

    correlation_ticks = correlation_ticks[
        correlation_ticks
        >= correlation_min
    ]

    theta_ticks = np.arccos(
        correlation_ticks
    )

    ax.set_xticks(theta_ticks)

    ax.set_xticklabels(
        [
            f"{value:g}"
            for value in correlation_ticks
        ],
        fontsize=10,
    )

    ax.set_xlabel(
        "Correlation",
        fontsize=12,
        fontweight="bold",
        labelpad=18,
    )

    radial_ticks = np.arange(
        0.0,
        max_normalized_std + 0.001,
        0.5,
    )

    ax.set_yticks(
        radial_ticks
    )

    ax.set_yticklabels(
        [
            f"{value:.1f}"
            for value in radial_ticks
        ],
        fontsize=9,
    )

    ax.set_rlabel_position(135)

    ax.grid(
        linestyle="--",
        linewidth=0.6,
        alpha=0.4,
    )

    reference_theta = 0.0
    reference_radius = 1.0

    ax.scatter(
        reference_theta,
        reference_radius,
        marker="*",
        s=180,
        color="black",
        edgecolor="black",
        zorder=6,
        label="MRMS reference",
    )

    ax.annotate(
        "MRMS",
        xy=(
            reference_theta,
            reference_radius,
        ),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
    )

    if centered_rmse_levels is None:
        centered_rmse_levels = np.arange(
            0.25,
            max_normalized_std + 0.25,
            0.25,
        )

    theta_grid = np.linspace(
        0,
        theta_max,
        361,
    )

    radius_grid = np.linspace(
        0,
        max_normalized_std,
        361,
    )

    THETA, RADIUS = np.meshgrid(
        theta_grid,
        radius_grid,
    )

    centered_rmse = np.sqrt(
        1.0
        + RADIUS ** 2
        - 2.0
        * RADIUS
        * np.cos(THETA)
    )

    contour_set = ax.contour(
        THETA,
        RADIUS,
        centered_rmse,
        levels=centered_rmse_levels,
        colors="0.45",
        linewidths=0.7,
        linestyles=":",
        alpha=0.8,
    )

    ax.clabel(
        contour_set,
        inline=True,
        fontsize=8,
        fmt="%.2g",
    )

    plotted_products = []

    for product in product_order:
        subset = working.loc[
            working[product_column]
            == product
        ]

        if subset.empty:
            continue

        row = subset.iloc[0]

        correlation = float(
            row[
                correlation_column
            ]
        )

        normalized_std = float(
            row[
                normalized_std_column
            ]
        )

        correlation = float(
            np.clip(
                correlation,
                -1.0,
                1.0,
            )
        )

        if correlation < correlation_min:
            continue

        theta = float(
            np.arccos(correlation)
        )

        ax.scatter(
            theta,
            normalized_std,
            marker=markers[product],
            s=100,
            color=colors[product],
            edgecolor="black",
            linewidth=0.8,
            zorder=5,
            label=labels[product],
        )

        plotted_products.append(product)

        if annotate_points:
            ax.annotate(
                labels[product],
                xy=(
                    theta,
                    normalized_std,
                ),
                xytext=(7, 6),
                textcoords="offset points",
                fontsize=9.5,
                fontweight="bold",
                color=colors[product],
            )

    ax.text(
        np.radians(104),
        max_normalized_std * 0.76,
        "Normalized standard deviation",
        fontsize=11,
        fontweight="bold",
        rotation=75,
        ha="center",
        va="center",
    )

    ax.set_title(
        title,
        fontsize=15,
        fontweight="bold",
        pad=24,
    )

    handles, legend_labels = (
        ax.get_legend_handles_labels()
    )

    if handles:
        ax.legend(
            handles,
            legend_labels,
            frameon=False,
            loc="upper right",
            bbox_to_anchor=(1.27, 1.10),
        )

    fig.tight_layout()
    _add_preliminary_label(
        fig,
        preliminary_label,
    )

    save_figure(
        fig,
        output_path,
    )

    return fig, ax


def plot_mrms_product_scatter_density(
    data: pd.DataFrame,
    *,
    products: Sequence[str] = (
        "GPROF_V7",
        "GPROF_V8",
        "ERA5",
    ),
    reference_column: str = "MRMS",
    product_labels: Mapping[str, str] | None = None,
    axis_limit: tuple[float, float] = (
        0.0,
        12.0,
    ),
    tick_values: Sequence[float] = (
        0.0,
        1.0,
        2.0,
        4.0,
        8.0,
        12.0,
    ),
    density_vmin: float = 1.0,
    density_vmax: float = 5000.0,
    add_regression_line: bool = True,
    title: str | None = None,
    figure_size: tuple[float, float] = (
        16,
        5.5,
    ),
    output_path: str | Path | None = None,
    preliminary_label: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot MRMS-versus-product scatter-density panels on linear axes.

    Notes
    -----
    - Both axes remain fully linear.
    - Tick positions are explicitly controlled using tick_values.
    - Metrics are calculated from the complete paired finite sample.
    - Only the displayed density and regression line are restricted to
      axis_limit.
    """

    _validate_dataframe_columns(
        data,
        [
            reference_column,
            *products,
        ],
    )

    labels = _resolve_mapping(
        products,
        product_labels,
        DEFAULT_PRODUCT_LABELS,
    )

    axis_min = float(
        axis_limit[0]
    )

    axis_max = float(
        axis_limit[1]
    )

    if axis_min < 0.0:
        raise ValueError(
            "axis_limit minimum must be nonnegative."
        )

    if axis_max <= axis_min:
        raise ValueError(
            "axis_limit maximum must exceed its minimum."
        )

    tick_values = np.asarray(
        tick_values,
        dtype=float,
    )

    tick_values = tick_values[
        (tick_values >= axis_min)
        & (tick_values <= axis_max)
    ]

    if tick_values.size == 0:
        raise ValueError(
            "No tick values fall within axis_limit."
        )

    density_norm = ImageNormalize(
        vmin=density_vmin,
        vmax=density_vmax,
        stretch=LogStretch(),
    )

    fig = plt.figure(
        figsize=figure_size
    )

    axes = []

    for panel_index, product in enumerate(
        products,
        start=1,
    ):
        ax = fig.add_subplot(
            1,
            len(products),
            panel_index,
            projection="scatter_density",
        )

        reference_all, product_all = _finite_pair(
            data,
            reference_column,
            product,
        )

        nonnegative = (
            (reference_all >= 0.0)
            & (product_all >= 0.0)
        )

        reference_all = reference_all[
            nonnegative
        ]

        product_all = product_all[
            nonnegative
        ]

        if reference_all.size == 0:
            raise ValueError(
                f"No valid nonnegative samples are available for "
                f"{product}."
            )

        # --------------------------------------------------------------
        # Metrics from the complete paired finite sample
        # --------------------------------------------------------------
        metrics = _calculate_plot_metrics(
            reference_all,
            product_all,
        )

        reference_mean = float(
            np.mean(reference_all)
        )

        if (
            np.isfinite(reference_mean)
            and reference_mean > 0.0
        ):
            relative_bias = (
                100.0
                * np.mean(
                    product_all
                    - reference_all
                )
                / reference_mean
            )
        else:
            relative_bias = np.nan

        # --------------------------------------------------------------
        # Displayed samples only
        # --------------------------------------------------------------
        plot_mask = (
            (reference_all >= axis_min)
            & (reference_all <= axis_max)
            & (product_all >= axis_min)
            & (product_all <= axis_max)
        )

        reference = reference_all[
            plot_mask
        ]

        product_values = product_all[
            plot_mask
        ]

        if reference.size == 0:
            raise ValueError(
                f"No {product} samples fall within "
                f"{axis_limit} mm h-1."
            )

        density_artist = ScatterDensityArtist(
            ax,
            reference,
            product_values,
            norm=density_norm,
            cmap=WHITE_VIRIDIS,
        )

        ax.add_artist(
            density_artist
        )

        # --------------------------------------------------------------
        # One-to-one line
        # --------------------------------------------------------------
        ax.plot(
            [
                axis_min,
                axis_max,
            ],
            [
                axis_min,
                axis_max,
            ],
            linestyle="--",
            color="black",
            linewidth=1.4,
            alpha=0.85,
            zorder=10,
        )

        # --------------------------------------------------------------
        # Regression line calculated using displayed samples
        # --------------------------------------------------------------
        if (
            add_regression_line
            and reference.size >= 2
            and np.std(reference) > 0.0
        ):
            slope, intercept = np.polyfit(
                reference,
                product_values,
                1,
            )

            regression_x = np.linspace(
                axis_min,
                axis_max,
                500,
            )

            regression_y = (
                slope * regression_x
                + intercept
            )

            valid_regression = (
                np.isfinite(regression_y)
                & (regression_y >= axis_min)
                & (regression_y <= axis_max)
            )

            ax.plot(
                regression_x[
                    valid_regression
                ],
                regression_y[
                    valid_regression
                ],
                linestyle="-",
                color="red",
                linewidth=1.3,
                alpha=0.90,
                zorder=11,
            )

        # --------------------------------------------------------------
        # Metrics annotation
        # --------------------------------------------------------------
        sample_count = int(
            metrics["N"]
        )

        if sample_count >= 1_000_000:
            sample_text = (
                f"{sample_count / 1_000_000:.2f} M"
            )
        elif sample_count >= 1_000:
            sample_text = (
                f"{sample_count / 1_000:.1f} k"
            )
        else:
            sample_text = (
                f"{sample_count:,}"
            )

        metrics_text = (
            f"RB = {relative_bias:.1f}%\n"
            f"RMSE = {metrics['RMSE']:.3f} mm h$^{{-1}}$\n"
            f"CC = {metrics['CC']:.3f}\n"
            f"N = {sample_text}"
        )

        ax.text(
            0.035,
            0.965,
            metrics_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10.5,
            fontweight="bold",
            linespacing=1.25,
            bbox={
                "boxstyle": "round,pad=0.32",
                "facecolor": "white",
                "alpha": 0.82,
                "edgecolor": "0.55",
                "linewidth": 0.6,
            },
            zorder=20,
        )

        # --------------------------------------------------------------
        # Linear axes and selected real-number ticks
        # --------------------------------------------------------------
        ax.set_xlim(
            axis_min,
            axis_max,
        )

        ax.set_ylim(
            axis_min,
            axis_max,
        )

        ax.set_xticks(
            tick_values
        )

        ax.set_yticks(
            tick_values
        )

        tick_formatter = FuncFormatter(
            lambda value, position: f"{value:g}"
        )

        ax.xaxis.set_major_formatter(
            tick_formatter
        )

        ax.yaxis.set_major_formatter(
            tick_formatter
        )

        ax.xaxis.set_minor_locator(
            AutoMinorLocator(2)
        )

        ax.yaxis.set_minor_locator(
            AutoMinorLocator(2)
        )

        ax.set_aspect(
            "equal",
            adjustable="box",
        )

        ax.set_xlabel(
            "MRMS precipitation (mm h$^{-1}$)",
            fontsize=12,
            fontweight="bold",
        )

        if panel_index == 1:
            ax.set_ylabel(
                "Product precipitation (mm h$^{-1}$)",
                fontsize=12,
                fontweight="bold",
            )

        ax.set_title(
            labels[product],
            fontsize=14,
            fontweight="bold",
            pad=7,
        )

        ax.grid(
            True,
            which="major",
            color="gray",
            linestyle="--",
            linewidth=0.45,
            alpha=0.40,
        )

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=10,
            direction="in",
            top=True,
            right=True,
            length=5,
        )

        ax.tick_params(
            axis="both",
            which="minor",
            direction="in",
            top=True,
            right=True,
            length=2.5,
        )

        axes.append(
            ax
        )

    if title:
        fig.suptitle(
            title,
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        top_position = 0.87
    else:
        top_position = 0.91

    fig.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.15,
        top=top_position,
        wspace=0.18,
    )

    _add_preliminary_label(
        fig,
        preliminary_label,
    )

    save_figure(
        fig,
        output_path,
    )

    return fig, np.asarray(
        axes
    )


def plot_quantitative_metric_bars(
    metric_data: pd.DataFrame,
    *,
    product_order: Sequence[str] = (
        "GPROF_V7",
        "GPROF_V8",
        "ERA5",
    ),
    product_labels: Mapping[str, str] | None = None,
    product_colors: Mapping[str, str] | None = None,
    metrics: Sequence[str] = (
        "correlation",
        "relative_bias_percent",
        "RMSE",
    ),
    metric_labels: Mapping[str, str] | None = None,
    metric_units: Mapping[str, str] | None = None,
    title: str | None = None,
    figure_size: tuple[float, float] = (10, 8),
    output_path: str | Path | None = None,
    preliminary_label: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot one bar-chart panel for each quantitative metric.
    """

    required_columns = [
        "product",
        *metrics,
    ]

    _validate_dataframe_columns(
        metric_data,
        required_columns,
    )

    labels = _resolve_mapping(
        product_order,
        product_labels,
        DEFAULT_PRODUCT_LABELS,
    )

    colors = _resolve_mapping(
        product_order,
        product_colors,
        DEFAULT_PRODUCT_COLORS,
    )

    default_metric_labels = {
        "correlation": "CC",
        "relative_bias_percent": "Relative bias",
        "mean_bias": "Mean bias",
        "MAE": "MAE",
        "RMSE": "RMSE",
        "centered_RMSE": "Centered RMSE",
    }

    default_metric_units = {
        "correlation": "",
        "relative_bias_percent": "%",
        "mean_bias": "mm h$^{-1}$",
        "MAE": "mm h$^{-1}$",
        "RMSE": "mm h$^{-1}$",
        "centered_RMSE": "mm h$^{-1}$",
    }

    if metric_labels is not None:
        default_metric_labels.update(
            metric_labels
        )

    if metric_units is not None:
        default_metric_units.update(
            metric_units
        )

    ordered = (
        metric_data
        .set_index("product")
        .reindex(product_order)
    )

    number_of_metrics = len(metrics)

    fig, axes = plt.subplots(
        nrows=number_of_metrics,
        ncols=1,
        figsize=figure_size,
        sharex=True,
        squeeze=False,
    )

    axes = axes.ravel()

    x_positions = np.arange(
        len(product_order)
    )

    bar_colors = [
        colors[product]
        for product in product_order
    ]

    for panel_index, (
        ax,
        metric,
    ) in enumerate(
        zip(
            axes,
            metrics,
        ),
        start=1,
    ):
        values = ordered[
            metric
        ].to_numpy(dtype=float)

        bars = ax.bar(
            x_positions,
            values,
            color=bar_colors,
            edgecolor="black",
            linewidth=0.6,
            width=0.34,
            zorder=3,
        )

        if metric == "relative_bias_percent":
            ax.axhline(
                0,
                color="black",
                linestyle="--",
                linewidth=0.9,
            )

        ylabel = default_metric_labels[
            metric
        ]

        units = default_metric_units.get(
            metric,
            "",
        )

        if units:
            ylabel += f" [{units}]"

        ax.set_ylabel(
            ylabel,
            fontsize=12,
            fontweight="bold",
        )

        ax.text(
            0.015,
            0.93,
            f"({chr(96 + panel_index)})",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="top",
        )

        for bar, value in zip(
            bars,
            values,
        ):
            if not np.isfinite(value):
                continue

            vertical_alignment = (
                "bottom"
                if value >= 0
                else "top"
            )

            offset = (
                3
                if value >= 0
                else -3
            )

            ax.annotate(
                f"{value:.3f}",
                xy=(
                    bar.get_x()
                    + bar.get_width() / 2,
                    value,
                ),
                xytext=(
                    0,
                    offset,
                ),
                textcoords="offset points",
                ha="center",
                va=vertical_alignment,
                fontsize=9,
            )

        ax.grid(
            axis="y",
            linestyle="--",
            linewidth=0.6,
            alpha=0.35,
        )

        ax.tick_params(
            direction="in",
            top=True,
            right=True,
            labelsize=10,
        )

    axes[-1].set_xticks(
        x_positions
    )

    axes[-1].set_xticklabels(
        [
            labels[product]
            for product in product_order
        ],
        rotation=20,
        ha="right",
        fontsize=11,
        fontweight="bold",
    )

    # legend_handles = [
    #     plt.Rectangle(
    #         (0, 0),
    #         1,
    #         1,
    #         facecolor=colors[product],
    #         edgecolor="black",
    #         label=labels[product],
    #     )
    #     for product in product_order
    # ]

    # fig.legend(
    #     handles=legend_handles,
    #     ncol=len(product_order),
    #     frameon=False,
    #     loc="lower center",
    #     bbox_to_anchor=(0.5, 0.005),
    # )

    if title:
        fig.suptitle(
            title,
            fontsize=16,
            fontweight="bold",
            y=0.99,
        )

    fig.tight_layout()

    _add_preliminary_label(
        fig,
        preliminary_label,
    )

    save_figure(
        fig,
        output_path,
    )

    return fig, axes


# =============================================================================
# Spatial map plotting
# =============================================================================

def _resolve_metric_plot_configuration(
    metric_configuration=None,
):
    """
    Return default discrete plotting settings for spatial metrics.

    The returned dictionary contains, for each metric:
        - label
        - discrete contour/color boundaries
        - colormap
        - colorbar extension behavior

    Values supplied through metric_configuration override the defaults.
    """

    defaults = {
        "CC": {
            "label": "Correlation coefficient",
            "levels": np.arange(
                0.0,
                1.01,
                0.1,
            ),
            "cmap": "viridis",
            "extend": "neither",
        },

        "relative_bias": {
            "label": "Relative bias (%)",
            "levels": np.array(
                [
                    -100,
                    -60,
                    -40,
                    -25,
                    -15,
                    -5,
                    0,
                    5,
                    15,
                    25,
                    40,
                    60,
                    100,
                ],
                dtype=float,
            ),
            "cmap": "RdBu_r",
            "extend": "both",
        },

        "RMSE": {
            "label": "RMSE (mm h$^{-1}$)",
            "levels": np.array(
                [
                    0.00,
                    0.05,
                    0.10,
                    0.15,
                    0.20,
                    0.30,
                    0.40,
                    0.50,
                    0.75,
                    1.00,
                ],
                dtype=float,
            ),
            "cmap": "magma",
            "extend": "max",
        },
    }

    if metric_configuration is not None:
        for metric, updates in (
            metric_configuration.items()
        ):
            if metric not in defaults:
                raise KeyError(
                    f"Unsupported spatial metric configuration: "
                    f"{metric}"
                )

            defaults[metric].update(
                updates
            )

    return defaults

def _format_conus_map_axis(
    ax,
    *,
    extent: tuple[float, float, float, float],
    longitude_limits: tuple[float, float] | None = None,
    latitude_limits: tuple[float, float] | None = None,
    coast_resolution: str = "50m",
    coast_linewidth: float = 0.8,
    border_linewidth: float = 0.7,    
    show_left_labels: bool = False,
    show_bottom_labels: bool = True,
    show_gridlines: bool = True,
) -> None:
    """
    Apply consistent CONUS map formatting.
    """

    projection = ccrs.PlateCarree()

    west, east, south, north = extent

    if longitude_limits is not None:
        west = max(
            west,
            longitude_limits[0],
        )

        east = min(
            east,
            longitude_limits[1],
        )

    if latitude_limits is not None:
        south = max(
            south,
            latitude_limits[0],
        )

        north = min(
            north,
            latitude_limits[1],
        )

    map_extent = [
        west,
        east,
        south,
        north,
    ]

    ax.set_extent(
        map_extent,
        crs=projection,
    )

    ax.set_xlim(
        west,
        east,
    )

    ax.set_ylim(
        south,
        north,
    )

    ax.coastlines(
        resolution=coast_resolution,
        linewidth=coast_linewidth,
        color="black",
        zorder=5,
    )

    ax.add_feature(
        cfeature.BORDERS.with_scale(
            coast_resolution
        ),
        linewidth=border_linewidth,
        edgecolor="black",
        zorder=5,
    )    

    if show_gridlines:
        gridlines = ax.gridlines(
            crs=projection,
            draw_labels=True,
            linewidth=0.5,
            linestyle="--",
            color="gray",
            alpha=0.45,
        )

        gridlines.top_labels = False
        gridlines.right_labels = False
        gridlines.left_labels = (
            show_left_labels
        )
        gridlines.bottom_labels = (
            show_bottom_labels
        )

        gridlines.xlabel_style = {
            "size": 9,
            "weight": "bold",
        }

        gridlines.ylabel_style = {
            "size": 9,
            "weight": "bold",
        }


def plot_spatial_mean_maps(
    spatial_dataset,
    *,
    product_order: Sequence[str] = (
        "GPROF_V7",
        "GPROF_V8",
        "MRMS",
        "ERA5",
    ),
    product_labels: Mapping[str, str] | None = None,
    extent: tuple[float, float, float, float] = (
        -125.0,
        -66.0,
        24.0,
        50.0,
    ),
    vmin: float = 0.0,
    vmax: float = 0.525,
    levels: Sequence[float] | None = None,
    cmap_name: str = "nipy_spectral",
    number_of_colors: int = 21,
    customize_low_end: bool = True,
    number_of_ticks: int = 7,
    title: str | None = None,
    colorbar_label: str = "(mm h$^{-1}$)",
    figure_size: tuple[float, float] = (
        14,
        8.8,
    ),
    dpi: int = 150,
    output_path: str | Path | None = None,
    preliminary_label: str | None = None,
):
    """
    Plot four spatial mean-precipitation maps in a 2 x 2 layout.
    """

    labels = _resolve_mapping(
        product_order,
        product_labels,
        DEFAULT_PRODUCT_LABELS,
    )

    if len(product_order) != 4:
        raise ValueError(
            "plot_spatial_mean_maps expects exactly four products."
        )

    # ------------------------------------------------------------------
    # Coordinates and sorting
    # ------------------------------------------------------------------
    longitude_original = np.asarray(
        spatial_dataset["longitude"].values,
        dtype=float,
    )

    latitude_original = np.asarray(
        spatial_dataset["latitude"].values,
        dtype=float,
    )

    if (
        longitude_original.ndim != 1
        or latitude_original.ndim != 1
    ):
        raise ValueError(
            "Longitude and latitude coordinates must be one-dimensional."
        )

    longitude_sort = np.argsort(
        longitude_original
    )

    latitude_sort = np.argsort(
        latitude_original
    )

    longitude = longitude_original[
        longitude_sort
    ]

    latitude = latitude_original[
        latitude_sort
    ]

    longitude_min = float(
        np.nanmin(longitude)
    )

    longitude_max = float(
        np.nanmax(longitude)
    )

    latitude_min = float(
        np.nanmin(latitude)
    )

    latitude_max = float(
        np.nanmax(latitude)
    )

    map_extent = [
        max(float(extent[0]), longitude_min),
        min(float(extent[1]), longitude_max),
        max(float(extent[2]), latitude_min),
        min(float(extent[3]), latitude_max),
    ]

    # ------------------------------------------------------------------
    # Discrete colormap
    # ------------------------------------------------------------------
    if levels is None:
        source_cmap = plt.get_cmap(
            cmap_name,
            number_of_colors,
        )

        color_array = source_cmap(
            np.linspace(
                0,
                1,
                number_of_colors,
            )
        )[:-1]

        color_levels = np.linspace(
            vmin,
            vmax,
            len(color_array) + 1,
        )

    else:
        color_levels = np.asarray(
            levels,
            dtype=float,
        )

        if color_levels.ndim != 1:
            raise ValueError(
                "levels must be one-dimensional."
            )

        if not np.all(
            np.diff(color_levels) > 0
        ):
            raise ValueError(
                "levels must be strictly increasing."
            )

        source_cmap = plt.get_cmap(
            cmap_name,
            len(color_levels),
        )

        color_array = source_cmap(
            np.linspace(
                0,
                1,
                len(color_levels),
            )
        )[:-1]

    if (
        customize_low_end
        and len(color_array) >= 3
    ):
        color_array[0] = [
            128 / 256,
            100 / 256,
            128 / 256,
            1,
        ]

        color_array[2] = (
            color_array[1].copy()
        )

        color_array[1] = [
            0.7,
            0.1,
            0.7,
            1,
        ]

    discrete_cmap = ListedColormap(
        color_array
    )

    norm = BoundaryNorm(
        color_levels,
        discrete_cmap.N,
    )

    # Missing cells.
    discrete_cmap.set_bad(
        color="white"
    )

    # Values outside plotting limits.
    discrete_cmap.set_under(
        discrete_cmap.colors[0]
    )

    discrete_cmap.set_over(
        discrete_cmap.colors[-1]
    )

    # ------------------------------------------------------------------
    # Figure and map panels
    # ------------------------------------------------------------------
    projection = ccrs.PlateCarree()

    fig, axes = plt.subplots(
        2,
        2,
        figsize=figure_size,
        dpi=dpi,
        subplot_kw={
            "projection": projection,
        },
    )

    axes_flat = np.asarray(
        axes
    ).ravel()

    artists = []

    for panel_index, (
        ax,
        product,
    ) in enumerate(
        zip(
            axes_flat,
            product_order,
        ),
        start=1,
    ):
        field = np.asarray(
            spatial_dataset[
                f"{product}_mean"
            ].values,
            dtype=float,
        )

        field = np.squeeze(
            field
        )

        expected_shape = (
            len(latitude_original),
            len(longitude_original),
        )

        if field.shape != expected_shape:
            raise ValueError(
                f"{product}: field shape {field.shape} "
                f"does not match expected shape "
                f"{expected_shape}."
            )

        field = field[
            latitude_sort,
            :,
        ]

        field = field[
            :,
            longitude_sort,
        ]

        ax.set_extent(
            map_extent,
            crs=projection,
        )

        ax.set_xlim(
            map_extent[0],
            map_extent[1],
        )

        ax.set_ylim(
            map_extent[2],
            map_extent[3],
        )

        artist = ax.contourf(
            longitude,
            latitude,
            field,
            levels=color_levels,
            cmap=discrete_cmap,
            norm=norm,
            transform=projection,
            extend="max",
        )

        ax.coastlines(
            resolution="50m",
            linewidth=0.9,
            color="black",
            zorder=5,
        )

        ax.add_feature(
            cfeature.BORDERS.with_scale(
                "50m"
            ),
            linewidth=0.8,
            edgecolor="black",
            zorder=5,
        )

        row_index = (
            panel_index - 1
        ) // 2

        column_index = (
            panel_index - 1
        ) % 2

        gridlines = ax.gridlines(
            crs=projection,
            draw_labels=True,
            linewidth=0.5,
            linestyle="--",
            color="gray",
            alpha=0.45,
        )

        gridlines.top_labels = False
        gridlines.right_labels = False

        gridlines.left_labels = (
            column_index == 0
        )

        gridlines.bottom_labels = (
            row_index == 1
        )

        gridlines.xlabel_style = {
            "size": 12,
            "weight": "bold",
        }

        gridlines.ylabel_style = {
            "size": 12,
            "weight": "bold",
        }

        ax.set_title(
            f"({chr(96 + panel_index)}) "
            f"{labels[product]}",
            fontsize=14,
            fontweight="bold",
            pad=6,
        )

        artists.append(
            artist
        )

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    if title:
        top_position = 0.92
    else:
        top_position = 0.965

    fig.subplots_adjust(
        left=0.055,
        right=0.985,
        top=top_position,
        bottom=0.16,
        wspace=0.055,
        hspace=0.055,
    )

    # ------------------------------------------------------------------
    # Dedicated colorbar axis
    # ------------------------------------------------------------------
    colorbar_axis = fig.add_axes(
        [
            0.20,   # left
            0.065,  # bottom
            0.60,   # width
            0.025,  # height
        ]
    )

    colorbar = fig.colorbar(
        artists[0],
        cax=colorbar_axis,
        orientation="horizontal",
        extend="max",
    )

    colorbar_ticks = np.linspace(
        float(color_levels[0]),
        float(color_levels[-1]),
        number_of_ticks,
    )

    colorbar.set_ticks(
        colorbar_ticks
    )

    colorbar.set_ticklabels(
        [
            f"{tick:.3f}"
            if vmax < 1
            else f"{tick:g}"
            for tick in colorbar_ticks
        ]
    )

    colorbar.ax.tick_params(
        labelsize=12
    )

    colorbar.set_label(
        colorbar_label,
        fontsize=12,
        fontweight="bold",
        labelpad=5,
    )

    if title:
        fig.suptitle(
            title,
            fontsize=16,
            fontweight="bold",
            y=0.985,
        )

    _add_preliminary_label(
        fig,
        preliminary_label,
    )

    save_figure(
        fig,
        output_path,
        dpi=dpi,
    )

    return fig, axes


def plot_spatial_means_and_differences(
    spatial_mean_dataset,
    spatial_difference_dataset=None,
    *,
    extent: tuple[float, float, float, float] = (
        -125.0,
        -66.0,
        24.0,
        50.0,
    ),
    mean_vmin: float = 0.0,
    mean_vmax: float = 0.525,
    mean_levels: Sequence[float] | None = None,
    mean_cmap_name: str = "nipy_spectral",
    mean_number_of_colors: int = 21,
    customize_mean_low_end: bool = True,
    difference_vmin: float = -0.30,
    difference_vmax: float = 0.30,
    difference_step: float = 0.05,
    difference_cmap_name: str = "RdBu_r",
    mean_colorbar_ticks: Sequence[float] | None = None,
    difference_colorbar_ticks: Sequence[float] | None = None,
    mean_colorbar_label: str = (
        "Mean precipitation (mm h$^{-1}$)"
    ),
    difference_colorbar_label: str = (
        "Mean precipitation difference (mm h$^{-1}$)"
    ),
    figure_size: tuple[float, float] = (
        16,
        15,
    ),
    dpi: int = 150,
    title: str | None = None,
    output_path: str | Path | None = None,
    preliminary_label: str | None = None,
):
    """
    Plot spatial mean precipitation and associated differences.

    Layout
    ------
    MRMS mean       | MRMS - GPROF V8
    GPROF V8 mean   | MRMS - GPROF V7
    GPROF V7 mean   | MRMS - ERA5
    ERA5 mean       | GPROF V7 - GPROF V8

    Positive difference values indicate the first product in the
    difference label has the higher mean precipitation.
    """

    required_mean_variables = (
        "MRMS_mean",
        "GPROF_V8_mean",
        "GPROF_V7_mean",
        "ERA5_mean",
    )

    missing_mean_variables = [
        variable
        for variable in required_mean_variables
        if variable not in spatial_mean_dataset
    ]

    if missing_mean_variables:
        raise KeyError(
            "Missing required spatial mean variables: "
            + ", ".join(missing_mean_variables)
        )

    # --------------------------------------------------------------
    # Calculate differences internally when not supplied
    # --------------------------------------------------------------
    if spatial_difference_dataset is None:
        import xarray as xr

        spatial_difference_dataset = xr.Dataset(
            {
                "MRMS_minus_GPROF_V8": (
                    spatial_mean_dataset["MRMS_mean"]
                    - spatial_mean_dataset["GPROF_V8_mean"]
                ),
                "MRMS_minus_GPROF_V7": (
                    spatial_mean_dataset["MRMS_mean"]
                    - spatial_mean_dataset["GPROF_V7_mean"]
                ),
                "MRMS_minus_ERA5": (
                    spatial_mean_dataset["MRMS_mean"]
                    - spatial_mean_dataset["ERA5_mean"]
                ),
                "GPROF_V7_minus_GPROF_V8": (
                    spatial_mean_dataset["GPROF_V7_mean"]
                    - spatial_mean_dataset["GPROF_V8_mean"]
                ),
            }
        )

    required_difference_variables = (
            "GPROF_V8_minus_MRMS",
            "GPROF_V7_minus_MRMS",
            "ERA5_minus_MRMS",
            "GPROF_V7_minus_GPROF_V8",
        )

    missing_difference_variables = [
        variable
        for variable in required_difference_variables
        if variable not in spatial_difference_dataset
    ]

    if missing_difference_variables:
        raise KeyError(
            "Missing required difference variables: "
            + ", ".join(missing_difference_variables)
        )

    if difference_vmax <= difference_vmin:
        raise ValueError(
            "difference_vmax must exceed difference_vmin."
        )

    if difference_step <= 0:
        raise ValueError(
            "difference_step must be positive."
        )

    # --------------------------------------------------------------
    # Coordinates
    # --------------------------------------------------------------
    longitude_original = np.asarray(
        spatial_mean_dataset["longitude"].values,
        dtype=float,
    )

    latitude_original = np.asarray(
        spatial_mean_dataset["latitude"].values,
        dtype=float,
    )

    if (
        longitude_original.ndim != 1
        or latitude_original.ndim != 1
    ):
        raise ValueError(
            "Longitude and latitude must be one-dimensional."
        )

    longitude_sort = np.argsort(
        longitude_original
    )

    latitude_sort = np.argsort(
        latitude_original
    )

    longitude = longitude_original[
        longitude_sort
    ]

    latitude = latitude_original[
        latitude_sort
    ]

    map_extent = [
        max(
            float(extent[0]),
            float(np.nanmin(longitude)),
        ),
        min(
            float(extent[1]),
            float(np.nanmax(longitude)),
        ),
        max(
            float(extent[2]),
            float(np.nanmin(latitude)),
        ),
        min(
            float(extent[3]),
            float(np.nanmax(latitude)),
        ),
    ]

    # --------------------------------------------------------------
    # Mean-field discrete colormap
    # --------------------------------------------------------------
    if mean_levels is None:
        mean_source_cmap = plt.get_cmap(
            mean_cmap_name,
            mean_number_of_colors,
        )

        mean_color_array = mean_source_cmap(
            np.linspace(
                0,
                1,
                mean_number_of_colors,
            )
        )[:-1]

        mean_boundaries = np.linspace(
            mean_vmin,
            mean_vmax,
            len(mean_color_array) + 1,
        )

    else:
        mean_boundaries = np.asarray(
            mean_levels,
            dtype=float,
        )

        if (
            mean_boundaries.ndim != 1
            or not np.all(
                np.diff(mean_boundaries) > 0
            )
        ):
            raise ValueError(
                "mean_levels must be a strictly increasing "
                "one-dimensional sequence."
            )

        mean_source_cmap = plt.get_cmap(
            mean_cmap_name,
            len(mean_boundaries),
        )

        mean_color_array = mean_source_cmap(
            np.linspace(
                0,
                1,
                len(mean_boundaries),
            )
        )[:-1]

    if (
        customize_mean_low_end
        and len(mean_color_array) >= 3
    ):
        mean_color_array[0] = [
            128 / 256,
            100 / 256,
            128 / 256,
            1,
        ]

        mean_color_array[2] = (
            mean_color_array[1].copy()
        )

        mean_color_array[1] = [
            0.7,
            0.1,
            0.7,
            1,
        ]

    mean_cmap = ListedColormap(
        mean_color_array
    )

    mean_norm = BoundaryNorm(
        mean_boundaries,
        mean_cmap.N,
    )

    mean_cmap.set_bad(
        color="white"
    )

    mean_cmap.set_under(
        mean_cmap.colors[0]
    )

    mean_cmap.set_over(
        mean_cmap.colors[-1]
    )

    # --------------------------------------------------------------
    # Difference discrete colormap
    # --------------------------------------------------------------
    difference_epsilon = (
        1.0e-10
        * max(
            1.0,
            abs(
                difference_vmax
                - difference_vmin
            ),
        )
    )

    difference_boundaries = np.arange(
        difference_vmin,
        difference_vmax
        + difference_epsilon,
        difference_step,
        dtype=float,
    )

    if (
        difference_boundaries[-1]
        < difference_vmax
        - difference_epsilon
    ):
        difference_boundaries = np.append(
            difference_boundaries,
            difference_vmax,
        )

    # Ensure the final boundary is exactly difference_vmax.
    difference_boundaries[-1] = (
        difference_vmax
    )

    number_of_difference_intervals = (
        len(difference_boundaries) - 1
    )

    difference_source_cmap = plt.get_cmap(
        difference_cmap_name,
        number_of_difference_intervals,
    )

    difference_color_array = (
        difference_source_cmap(
            np.linspace(
                0,
                1,
                number_of_difference_intervals,
            )
        )
    )

    difference_cmap = ListedColormap(
        difference_color_array
    )

    difference_norm = BoundaryNorm(
        difference_boundaries,
        difference_cmap.N,
    )

    difference_cmap.set_bad(
        color="white"
    )

    difference_cmap.set_under(
        difference_cmap.colors[0]
    )

    difference_cmap.set_over(
        difference_cmap.colors[-1]
    )

    # --------------------------------------------------------------
    # Row definitions
    # --------------------------------------------------------------
    mean_rows = (
        (
            "MRMS_mean",
            "MRMS",
        ),
        (
            "GPROF_V8_mean",
            "GPROF V8",
        ),
        (
            "GPROF_V7_mean",
            "GPROF V7",
        ),
        (
            "ERA5_mean",
            "ERA5",
        ),
    )

    difference_rows = (
            (
                "GPROF_V8_minus_MRMS",
                "GPROF V8 − MRMS",
            ),
            (
                "GPROF_V7_minus_MRMS",
                "GPROF V7 − MRMS",
            ),
            (
                "ERA5_minus_MRMS",
                "ERA5 − MRMS",
            ),
            (
                "GPROF_V7_minus_GPROF_V8",
                "GPROF V7 − GPROF V8",
            ),
        )

    # --------------------------------------------------------------
    # Figure layout
    # --------------------------------------------------------------
    projection = ccrs.PlateCarree()

    fig = plt.figure(
        figsize=figure_size,
        dpi=dpi,
    )

    grid_spec = fig.add_gridspec(
        nrows=4,
        ncols=2,
        left=0.055,
        right=0.985,
        top=0.955 if title is None else 0.925,
        bottom=0.105,
        wspace=0.045,
        hspace=0.25,
    )

    axes = np.empty(
        (
            4,
            2,
        ),
        dtype=object,
    )

    mean_artist = None
    difference_artist = None
    panel_index = 0

    # --------------------------------------------------------------
    # Plot all rows
    # --------------------------------------------------------------
    for row_index in range(4):
        mean_variable, mean_label = (
            mean_rows[row_index]
        )

        difference_variable, difference_label = (
            difference_rows[row_index]
        )

        for column_index in range(2):
            ax = fig.add_subplot(
                grid_spec[
                    row_index,
                    column_index,
                ],
                projection=projection,
            )

            axes[
                row_index,
                column_index,
            ] = ax

            if column_index == 0:
                source_dataset = (
                    spatial_mean_dataset
                )

                variable_name = (
                    mean_variable
                )

                panel_title = (
                    mean_label
                )

                panel_cmap = (
                    mean_cmap
                )

                panel_norm = (
                    mean_norm
                )

                panel_levels = (
                    mean_boundaries
                )

                panel_extend = "max"

            else:
                source_dataset = (
                    spatial_difference_dataset
                )

                variable_name = (
                    difference_variable
                )

                panel_title = (
                    difference_label
                )

                panel_cmap = (
                    difference_cmap
                )

                panel_norm = (
                    difference_norm
                )

                panel_levels = (
                    difference_boundaries
                )

                panel_extend = "both"

            field = np.asarray(
                source_dataset[
                    variable_name
                ].values,
                dtype=float,
            )

            field = np.squeeze(
                field
            )

            expected_shape = (
                len(latitude_original),
                len(longitude_original),
            )

            if field.shape != expected_shape:
                raise ValueError(
                    f"{variable_name}: field shape "
                    f"{field.shape} does not match "
                    f"{expected_shape}."
                )

            field = field[
                latitude_sort,
                :,
            ]

            field = field[
                :,
                longitude_sort,
            ]

            ax.set_extent(
                map_extent,
                crs=projection,
            )

            ax.set_xlim(
                map_extent[0],
                map_extent[1],
            )

            ax.set_ylim(
                map_extent[2],
                map_extent[3],
            )

            artist = ax.contourf(
                longitude,
                latitude,
                field,
                levels=panel_levels,
                cmap=panel_cmap,
                norm=panel_norm,
                transform=projection,
                extend=panel_extend,
            )

            if (
                column_index == 0
                and mean_artist is None
            ):
                mean_artist = artist

            if (
                column_index == 1
                and difference_artist is None
            ):
                difference_artist = artist

            ax.coastlines(
                resolution="50m",
                linewidth=0.9,
                color="black",
                zorder=5,
            )

            ax.add_feature(
                cfeature.BORDERS.with_scale(
                    "50m"
                ),
                linewidth=0.8,
                edgecolor="black",
                zorder=5,
            )

            gridlines = ax.gridlines(
                crs=projection,
                draw_labels=True,
                linewidth=0.5,
                linestyle="--",
                color="gray",
                alpha=0.45,
            )

            gridlines.top_labels = False
            gridlines.right_labels = False

            gridlines.left_labels = (
                column_index == 0
            )

            gridlines.bottom_labels = (
                row_index == 3
            )

            gridlines.xlabel_style = {
                "size": 10,
                "weight": "bold",
            }

            gridlines.ylabel_style = {
                "size": 10,
                "weight": "bold",
            }

            panel_index += 1

            ax.set_title(
                f"({chr(96 + panel_index)}) "
                f"{panel_title}",
                fontsize=14,
                fontweight="bold",
                pad=6,
            )

    # Finalize panel positions before adding inset colorbars.
    fig.canvas.draw()

    # --------------------------------------------------------------
    # Mean colorbar beneath bottom-left panel
    # --------------------------------------------------------------
    mean_colorbar_axis = inset_axes(
        axes[3, 0],
        width="94%",
        height="9%",
        loc="lower center",
        bbox_to_anchor=(
            0.0,
            -0.34,
            1.0,
            1.0,
        ),
        bbox_transform=axes[3, 0].transAxes,
        borderpad=0,
    )

    mean_colorbar = fig.colorbar(
        ScalarMappable(
            norm=mean_norm,
            cmap=mean_cmap,
        ),
        cax=mean_colorbar_axis,
        orientation="horizontal",
        extend="max",
    )

    if mean_colorbar_ticks is None:
        mean_colorbar_ticks = np.linspace(
            mean_boundaries[0],
            mean_boundaries[-1],
            7,
        )

    mean_colorbar_ticks = np.asarray(
        mean_colorbar_ticks,
        dtype=float,
    )

    mean_colorbar.set_ticks(
        mean_colorbar_ticks
    )

    mean_colorbar.set_ticklabels(
        [
            f"{tick:.3f}"
            for tick in mean_colorbar_ticks
        ]
    )

    mean_colorbar.ax.tick_params(
        labelsize=11,
        length=5,
    )

    mean_colorbar.set_label(
        mean_colorbar_label,
        fontsize=12,
        fontweight="bold",
        labelpad=5,
    )

    # --------------------------------------------------------------
    # Difference colorbar beneath bottom-right panel
    # --------------------------------------------------------------
    difference_colorbar_axis = inset_axes(
        axes[3, 1],
        width="94%",
        height="9%",
        loc="lower center",
        bbox_to_anchor=(
            0.0,
            -0.34,
            1.0,
            1.0,
        ),
        bbox_transform=axes[3, 1].transAxes,
        borderpad=0,
    )

    difference_colorbar = fig.colorbar(
        ScalarMappable(
            norm=difference_norm,
            cmap=difference_cmap,
        ),
        cax=difference_colorbar_axis,
        orientation="horizontal",
        boundaries=difference_boundaries,
        extend="both",
    )

    if difference_colorbar_ticks is None:
        difference_colorbar_ticks = (
            np.arange(
                difference_vmin,
                difference_vmax
                + 0.5 * difference_step,
                difference_step,
                dtype=float,
            )
        )

    difference_colorbar_ticks = np.asarray(
        difference_colorbar_ticks,
        dtype=float,
    )

    difference_colorbar_ticks = (
        difference_colorbar_ticks[
            (
                difference_colorbar_ticks
                >= difference_vmin
            )
            & (
                difference_colorbar_ticks
                <= difference_vmax
            )
        ]
    )

    difference_colorbar.set_ticks(
        difference_colorbar_ticks
    )

    difference_tick_labels = []

    for tick in difference_colorbar_ticks:
        if np.isclose(
            tick,
            round(tick),
        ):
            difference_tick_labels.append(
                f"{int(round(tick))}"
            )
        elif abs(tick) < 0.1:
            difference_tick_labels.append(
                f"{tick:.2f}"
            )
        else:
            difference_tick_labels.append(
                f"{tick:.2f}"
            )

    difference_colorbar.set_ticklabels(
        difference_tick_labels
    )

    difference_colorbar.ax.tick_params(
        labelsize=11,
        length=5,
    )

    difference_colorbar.set_label(
        difference_colorbar_label,
        fontsize=12,
        fontweight="bold",
        labelpad=5,
    )

    if title:
        fig.suptitle(
            title,
            fontsize=17,
            fontweight="bold",
            y=0.985,
        )

    _add_preliminary_label(
        fig,
        preliminary_label,
    )

    save_figure(
        fig,
        output_path,
        dpi=dpi,
    )

    return fig, axes

def plot_spatial_continuous_metric_maps(
    spatial_dataset,
    *,
    product_order: Sequence[str] = (
        "GPROF_V7",
        "GPROF_V8",
        "ERA5",
    ),
    product_labels: Mapping[str, str] | None = None,
    extent: tuple[float, float, float, float] = (
        -125.0,
        -66.0,
        24.0,
        50.0,
    ),
    metric_configuration=None,
    title: str | None = None,
    figure_size: tuple[float, float] = (
        18,
        13,
    ),
    dpi: int = 150,
    colorbar_width_fraction: float = 0.48,
    colorbar_height: float = 0.014,
    colorbar_gap: float = 0.043,
    output_path: str | Path | None = None,
    preliminary_label: str | None = None,
):
    """
    Plot spatial CC, relative bias, and RMSE for three products.

    Layout
    ------
    Row 1:
        GPROF V7, GPROF V8, and ERA5 CC maps

    Row 2:
        GPROF V7, GPROF V8, and ERA5 relative-bias maps

    Row 3:
        GPROF V7, GPROF V8, and ERA5 RMSE maps

    Each row has one short centered horizontal colorbar.
    """

    if len(product_order) != 3:
        raise ValueError(
            "plot_spatial_continuous_metric_maps "
            "expects exactly three products."
        )

    labels = _resolve_mapping(
        product_order,
        product_labels,
        DEFAULT_PRODUCT_LABELS,
    )

    configuration = (
        _resolve_metric_plot_configuration(
            metric_configuration
        )
    )

    metrics = (
        "CC",
        "relative_bias",
        "RMSE",
    )

    short_metric_labels = {
        "CC": "CC",
        "relative_bias": "RB (%)",
        "RMSE": "RMSE (mm h$^{-1}$)",
    }

    # ------------------------------------------------------------------
    # Coordinates
    # ------------------------------------------------------------------
    longitude_original = np.asarray(
        spatial_dataset["longitude"].values,
        dtype=float,
    )

    latitude_original = np.asarray(
        spatial_dataset["latitude"].values,
        dtype=float,
    )

    if (
        longitude_original.ndim != 1
        or latitude_original.ndim != 1
    ):
        raise ValueError(
            "Longitude and latitude must be one-dimensional."
        )

    longitude_sort = np.argsort(
        longitude_original
    )

    latitude_sort = np.argsort(
        latitude_original
    )

    longitude = longitude_original[
        longitude_sort
    ]

    latitude = latitude_original[
        latitude_sort
    ]

    map_extent = [
        max(
            float(extent[0]),
            float(np.nanmin(longitude)),
        ),
        min(
            float(extent[1]),
            float(np.nanmax(longitude)),
        ),
        max(
            float(extent[2]),
            float(np.nanmin(latitude)),
        ),
        min(
            float(extent[3]),
            float(np.nanmax(latitude)),
        ),
    ]

    projection = ccrs.PlateCarree()

    # ------------------------------------------------------------------
    # Figure and main map grid
    # ------------------------------------------------------------------
    fig = plt.figure(
        figsize=figure_size,
        dpi=dpi,
    )

    grid_spec = fig.add_gridspec(
        nrows=3,
        ncols=3,
        left=0.055,
        right=0.985,
        top=0.94 if title else 0.975,
        bottom=0.075,
        wspace=0.045,
        hspace=0.48,
    )

    axes = np.empty(
        (
            len(metrics),
            len(product_order),
        ),
        dtype=object,
    )

    row_mappables = [
        None
    ] * len(metrics)

    row_levels = [
        None
    ] * len(metrics)

    row_extend = [
        None
    ] * len(metrics)

    panel_index = 0

    # ------------------------------------------------------------------
    # Metric rows
    # ------------------------------------------------------------------
    for row_index, metric in enumerate(
        metrics
    ):
        config = configuration[
            metric
        ]

        levels = np.asarray(
            config["levels"],
            dtype=float,
        )

        if levels.ndim != 1:
            raise ValueError(
                f"{metric} levels must be one-dimensional."
            )

        if not np.all(
            np.diff(levels) > 0
        ):
            raise ValueError(
                f"{metric} levels must be strictly increasing."
            )

        discrete_cmap, norm, levels = (
            create_discrete_colormap(
                vmin=float(levels[0]),
                vmax=float(levels[-1]),
                cmap_name=config["cmap"],
                levels=levels,
            )
        )

        # Undefined metric cells.
        discrete_cmap.set_bad(
            color="white"
        )

        # Use actual terminal colors for out-of-range values.
        discrete_cmap.set_under(
            discrete_cmap.colors[0]
        )

        discrete_cmap.set_over(
            discrete_cmap.colors[-1]
        )

        row_levels[
            row_index
        ] = levels

        row_extend[
            row_index
        ] = config["extend"]

        for column_index, product in enumerate(
            product_order
        ):
            ax = fig.add_subplot(
                grid_spec[
                    row_index,
                    column_index,
                ],
                projection=projection,
            )

            axes[
                row_index,
                column_index,
            ] = ax

            variable_name = (
                f"{product}_{metric}"
            )

            if variable_name not in spatial_dataset:
                raise KeyError(
                    f"Missing spatial variable: "
                    f"{variable_name}"
                )

            field = np.asarray(
                spatial_dataset[
                    variable_name
                ].values,
                dtype=float,
            )

            field = np.squeeze(
                field
            )

            expected_shape = (
                len(latitude_original),
                len(longitude_original),
            )

            if field.shape != expected_shape:
                raise ValueError(
                    f"{variable_name}: field shape "
                    f"{field.shape} does not match "
                    f"{expected_shape}."
                )

            field = field[
                latitude_sort,
                :,
            ]

            field = field[
                :,
                longitude_sort,
            ]

            ax.set_extent(
                map_extent,
                crs=projection,
            )

            ax.set_xlim(
                map_extent[0],
                map_extent[1],
            )

            ax.set_ylim(
                map_extent[2],
                map_extent[3],
            )

            artist = ax.contourf(
                longitude,
                latitude,
                field,
                levels=levels,
                cmap=discrete_cmap,
                norm=norm,
                transform=projection,
                extend=config["extend"],
            )

            if row_mappables[
                row_index
            ] is None:
                row_mappables[
                    row_index
                ] = artist

            ax.coastlines(
                resolution="50m",
                linewidth=0.8,
                color="black",
                zorder=5,
            )

            ax.add_feature(
                cfeature.BORDERS.with_scale(
                    "50m"
                ),
                linewidth=0.7,
                edgecolor="black",
                zorder=5,
            )

            gridlines = ax.gridlines(
                crs=projection,
                draw_labels=True,
                linewidth=0.5,
                linestyle="--",
                color="gray",
                alpha=0.40,
            )

            gridlines.top_labels = False
            gridlines.right_labels = False

            gridlines.left_labels = (
                column_index == 0
            )

            gridlines.bottom_labels = (
                row_index
                == len(metrics) - 1
            )

            gridlines.xlabel_style = {
                "size": 11,
                "weight": "bold",
            }

            gridlines.ylabel_style = {
                "size": 11,
                "weight": "bold",
            }

            if row_index == 0:
                ax.set_title(
                    labels[product],
                    fontsize=15,
                    fontweight="bold",
                    pad=7,
                )

            panel_index += 1

            ax.text(
                0.018,
                0.965,
                f"({chr(96 + panel_index)})",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=11,
                fontweight="bold",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.72,
                    "pad": 1.3,
                },
                zorder=10,
            )

    # ------------------------------------------------------------------
    # Finalize map positions before adding independent colorbars
    # ------------------------------------------------------------------
    fig.canvas.draw()

    # ------------------------------------------------------------------
    # Short centered colorbar beneath each metric row
    # ------------------------------------------------------------------
    for row_index, metric in enumerate(
        metrics
    ):
        left_map_position = (
            axes[
                row_index,
                0,
            ].get_position()
        )

        right_map_position = (
            axes[
                row_index,
                -1,
            ].get_position()
        )

        row_left = (
            left_map_position.x0
        )

        row_right = (
            right_map_position.x1
        )

        row_width = (
            row_right
            - row_left
        )

        colorbar_width = (
            row_width
            * colorbar_width_fraction
        )

        colorbar_left = (
            row_left
            + 0.5
            * (
                row_width
                - colorbar_width
            )
        )

        colorbar_bottom = (
            left_map_position.y0
            - colorbar_gap
        )

        colorbar_axis = fig.add_axes(
            [
                colorbar_left,
                colorbar_bottom,
                colorbar_width,
                colorbar_height,
            ]
        )

        colorbar = fig.colorbar(
            row_mappables[
                row_index
            ],
            cax=colorbar_axis,
            orientation="horizontal",
            extend=row_extend[
                row_index
            ],
        )

        levels = row_levels[
            row_index
        ]

        if metric == "CC":
            colorbar_ticks = np.array(
                [
                    0.0,
                    0.2,
                    0.4,
                    0.6,
                    0.8,
                    1.0,
                ],
                dtype=float,
            )

            colorbar_labels = [
                f"{tick:.1f}"
                for tick in colorbar_ticks
            ]

        elif metric == "relative_bias":
            colorbar_ticks = np.array(
                [
                    -100,
                    -60,
                    -25,
                    0,
                    25,
                    60,
                    100,
                ],
                dtype=float,
            )

            colorbar_labels = [
                f"{int(tick)}"
                for tick in colorbar_ticks
            ]

        else:
            colorbar_ticks = np.array(
                [
                    0.00,
                    0.10,
                    0.20,
                    0.40,
                    0.60,
                    0.80,
                    1.00,
                ],
                dtype=float,
            )

            # Retain only values supported by the configured RMSE levels.
            colorbar_ticks = colorbar_ticks[
                (
                    colorbar_ticks
                    >= levels[0]
                )
                & (
                    colorbar_ticks
                    <= levels[-1]
                )
            ]

            colorbar_labels = [
                f"{tick:.2f}"
                for tick in colorbar_ticks
            ]

        colorbar.set_ticks(
            colorbar_ticks
        )

        colorbar.set_ticklabels(
            colorbar_labels
        )

        colorbar.ax.tick_params(
            labelsize=12,
            length=5,
            width=1.0,
        )

        colorbar_axis.set_title(
            short_metric_labels[
                metric
            ],
            fontsize=13,
            fontweight="bold",
            pad=7,
        )

    if title:
        fig.suptitle(
            title,
            fontsize=17,
            fontweight="bold",
            y=0.985,
        )

    _add_preliminary_label(
        fig,
        preliminary_label,
    )

    save_figure(
        fig,
        output_path,
        dpi=dpi,
    )

    return fig, axes

def plot_spatial_categorical_metric_maps(
    spatial_dataset,
    *,
    product_order: Sequence[str] = (
        "GPROF_V7",
        "GPROF_V8",
        "ERA5",
    ),
    product_labels: Mapping[str, str] | None = None,
    extent: tuple[float, float, float, float] = (
        -125.0,
        -66.0,
        24.0,
        50.0,
    ),
    levels: Sequence[float] = (
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ),
    cmap_name: str = "viridis",
    title: str | None = None,
    figure_size: tuple[float, float] = (
        18,
        13,
    ),
    dpi: int = 150,
    colorbar_width_fraction: float = 0.50,
    colorbar_height: float = 0.014,
    colorbar_gap: float = 0.042,
    output_path: str | Path | None = None,
    preliminary_label: str | None = None,
):
    """
    Plot POD, FAR, and CSI maps for GPROF V7, GPROF V8, and ERA5.

    Each metric row has one short centered horizontal colorbar placed
    beneath the corresponding three maps.
    """

    if len(product_order) != 3:
        raise ValueError(
            "plot_spatial_categorical_metric_maps expects "
            "exactly three products."
        )

    labels = _resolve_mapping(
        product_order,
        product_labels,
        DEFAULT_PRODUCT_LABELS,
    )

    metrics = (
        "POD",
        "FAR",
        "CSI",
    )

    longitude_original = np.asarray(
        spatial_dataset["longitude"].values,
        dtype=float,
    )

    latitude_original = np.asarray(
        spatial_dataset["latitude"].values,
        dtype=float,
    )

    if (
        longitude_original.ndim != 1
        or latitude_original.ndim != 1
    ):
        raise ValueError(
            "Longitude and latitude must be one-dimensional."
        )

    longitude_sort = np.argsort(
        longitude_original
    )

    latitude_sort = np.argsort(
        latitude_original
    )

    longitude = longitude_original[
        longitude_sort
    ]

    latitude = latitude_original[
        latitude_sort
    ]

    map_extent = [
        max(
            float(extent[0]),
            float(np.nanmin(longitude)),
        ),
        min(
            float(extent[1]),
            float(np.nanmax(longitude)),
        ),
        max(
            float(extent[2]),
            float(np.nanmin(latitude)),
        ),
        min(
            float(extent[3]),
            float(np.nanmax(latitude)),
        ),
    ]

    boundaries = np.asarray(
        levels,
        dtype=float,
    )

    if boundaries.ndim != 1:
        raise ValueError(
            "levels must be one-dimensional."
        )

    if not np.all(
        np.diff(boundaries) > 0
    ):
        raise ValueError(
            "levels must be strictly increasing."
        )

    cmap, norm, boundaries = (
        create_discrete_colormap(
            vmin=float(boundaries[0]),
            vmax=float(boundaries[-1]),
            cmap_name=cmap_name,
            levels=boundaries,
        )
    )

    cmap.set_bad(
        color="white"
    )

    cmap.set_under(
        cmap.colors[0]
    )

    cmap.set_over(
        cmap.colors[-1]
    )

    projection = ccrs.PlateCarree()

    # --------------------------------------------------------------
    # Main map layout only
    # --------------------------------------------------------------
    fig = plt.figure(
        figsize=figure_size,
        dpi=dpi,
    )

    grid_spec = fig.add_gridspec(
        nrows=3,
        ncols=3,
        left=0.055,
        right=0.985,
        top=0.94 if title else 0.975,
        bottom=0.075,
        wspace=0.045,
        hspace=0.48,
    )

    axes = np.empty(
        (
            len(metrics),
            len(product_order),
        ),
        dtype=object,
    )

    row_mappables = [
        None
    ] * len(metrics)

    panel_index = 0

    # --------------------------------------------------------------
    # Metric map rows
    # --------------------------------------------------------------
    for row_index, metric in enumerate(
        metrics
    ):
        for column_index, product in enumerate(
            product_order
        ):
            ax = fig.add_subplot(
                grid_spec[
                    row_index,
                    column_index,
                ],
                projection=projection,
            )

            axes[
                row_index,
                column_index,
            ] = ax

            variable_name = (
                f"{product}_{metric}"
            )

            if variable_name not in spatial_dataset:
                raise KeyError(
                    f"Missing spatial variable: "
                    f"{variable_name}"
                )

            field = np.asarray(
                spatial_dataset[
                    variable_name
                ].values,
                dtype=float,
            )

            field = np.squeeze(
                field
            )

            expected_shape = (
                len(latitude_original),
                len(longitude_original),
            )

            if field.shape != expected_shape:
                raise ValueError(
                    f"{variable_name}: field shape "
                    f"{field.shape} does not match "
                    f"{expected_shape}."
                )

            field = field[
                latitude_sort,
                :,
            ]

            field = field[
                :,
                longitude_sort,
            ]

            ax.set_extent(
                map_extent,
                crs=projection,
            )

            ax.set_xlim(
                map_extent[0],
                map_extent[1],
            )

            ax.set_ylim(
                map_extent[2],
                map_extent[3],
            )

            artist = ax.contourf(
                longitude,
                latitude,
                field,
                levels=boundaries,
                cmap=cmap,
                norm=norm,
                transform=projection,
                extend="neither",
            )

            if row_mappables[row_index] is None:
                row_mappables[row_index] = artist

            ax.coastlines(
                resolution="50m",
                linewidth=0.8,
                color="black",
                zorder=5,
            )

            ax.add_feature(
                cfeature.BORDERS.with_scale(
                    "50m"
                ),
                linewidth=0.7,
                edgecolor="black",
                zorder=5,
            )

            gridlines = ax.gridlines(
                crs=projection,
                draw_labels=True,
                linewidth=0.5,
                linestyle="--",
                color="gray",
                alpha=0.40,
            )

            gridlines.top_labels = False
            gridlines.right_labels = False

            gridlines.left_labels = (
                column_index == 0
            )

            gridlines.bottom_labels = (
                row_index
                == len(metrics) - 1
            )

            gridlines.xlabel_style = {
                "size": 11,
                "weight": "bold",
            }

            gridlines.ylabel_style = {
                "size": 11,
                "weight": "bold",
            }

            if row_index == 0:
                ax.set_title(
                    labels[product],
                    fontsize=15,
                    fontweight="bold",
                    pad=7,
                )

            panel_index += 1

            ax.text(
                0.018,
                0.965,
                f"({chr(96 + panel_index)})",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=11,
                fontweight="bold",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.72,
                    "pad": 1.3,
                },
                zorder=10,
            )

    # --------------------------------------------------------------
    # Finalize map positions before placing colorbars
    # --------------------------------------------------------------
    fig.canvas.draw()

    # --------------------------------------------------------------
    # Short centered colorbar beneath each metric row
    # --------------------------------------------------------------
    colorbar_ticks = np.array(
        [
            0.0,
            0.2,
            0.4,
            0.6,
            0.8,
            1.0,
        ],
        dtype=float,
    )

    for row_index, metric in enumerate(
        metrics
    ):
        left_map_position = (
            axes[
                row_index,
                0,
            ].get_position()
        )

        right_map_position = (
            axes[
                row_index,
                -1,
            ].get_position()
        )

        row_left = (
            left_map_position.x0
        )

        row_right = (
            right_map_position.x1
        )

        row_width = (
            row_right
            - row_left
        )

        colorbar_width = (
            row_width
            * colorbar_width_fraction
        )

        colorbar_left = (
            row_left
            + 0.5
            * (
                row_width
                - colorbar_width
            )
        )

        colorbar_bottom = (
            left_map_position.y0
            - colorbar_gap
        )

        colorbar_axis = fig.add_axes(
            [
                colorbar_left,
                colorbar_bottom,
                colorbar_width,
                colorbar_height,
            ]
        )

        colorbar = fig.colorbar(
            row_mappables[
                row_index
            ],
            cax=colorbar_axis,
            orientation="horizontal",
            extend="neither",
        )

        colorbar.set_ticks(
            colorbar_ticks
        )

        colorbar.set_ticklabels(
            [
                f"{tick:.1f}"
                for tick in colorbar_ticks
            ]
        )

        colorbar.ax.tick_params(
            labelsize=12,
            length=5,
            width=1.0,
        )

        # Place POD, FAR, or CSI above its colorbar.
        colorbar_axis.set_title(
            metric,
            fontsize=13,
            fontweight="bold",
            pad=7,
        )

    if title:
        fig.suptitle(
            title,
            fontsize=17,
            fontweight="bold",
            y=0.985,
        )

    _add_preliminary_label(
        fig,
        preliminary_label,
    )

    save_figure(
        fig,
        output_path,
        dpi=dpi,
    )

    return fig, axes


def plot_spatial_sample_count(
    spatial_dataset,
    *,
    variable: str = "sample_count",
    extent: tuple[float, float, float, float] = (
        -125.0,
        -66.0,
        24.0,
        50.0,
    ),
    title: str = "Common-valid sample count",
    figure_size: tuple[float, float] = (
        10,
        5.5,
    ),
    output_path: str | Path | None = None,
    preliminary_label: str | None = None,
):
    """
    Plot common-valid footprint counts on the CONUS grid.
    """

    longitude = spatial_dataset[
        "longitude"
    ].values

    latitude = spatial_dataset[
        "latitude"
    ].values

    field = spatial_dataset[
        variable
    ].values.astype(float)

    field[
        field <= 0
    ] = np.nan

    finite = field[
        np.isfinite(field)
    ]

    if finite.size == 0:
        raise ValueError(
            "No positive spatial sample counts are available."
        )

    fig, ax = plt.subplots(
        figsize=figure_size,
        subplot_kw={
            "projection": (
                ccrs.PlateCarree()
            )
        },
        constrained_layout=True,
    )

    plotted = ax.pcolormesh(
        longitude,
        latitude,
        field,
        shading="auto",
        cmap="viridis",
        norm=LogNorm(
            vmin=max(
                1,
                float(
                    np.nanpercentile(
                        finite,
                        2,
                    )
                ),
            ),
            vmax=float(
                np.nanpercentile(
                    finite,
                    98,
                )
            ),
        ),
        transform=ccrs.PlateCarree(),
    )

    _format_conus_map_axis(
        ax,
        extent=extent,
    )

    colorbar = fig.colorbar(
        plotted,
        ax=ax,
        orientation="vertical",
        pad=0.02,
    )

    colorbar.set_label(
        "Footprint count",
        fontsize=11,
        fontweight="bold",
    )

    ax.set_title(
        title,
        fontsize=15,
        fontweight="bold",
    )

    _add_preliminary_label(
        fig,
        preliminary_label,
    )

    save_figure(
        fig,
        output_path,
    )

    return fig, ax

def create_discrete_colormap(
    *,
    vmin: float,
    vmax: float,
    cmap_name: str = "nipy_spectral",
    number_of_colors: int = 20,
    levels: Sequence[float] | None = None,
    customize_low_end: bool = False,
):
    """
    Create a discrete Matplotlib colormap and BoundaryNorm.

    Parameters
    ----------
    vmin, vmax
        Color-scale limits.

    cmap_name
        Name of the source Matplotlib colormap.

    number_of_colors
        Number of discrete color intervals when levels are not supplied.

    levels
        Optional explicit color boundaries.

    customize_low_end
        Apply the low-end color modifications used in earlier precipitation
        figures.
    """

    if levels is None:
        levels_array = np.linspace(
            vmin,
            vmax,
            number_of_colors + 1,
        )
    else:
        levels_array = np.asarray(
            levels,
            dtype=float,
        )

        if levels_array.ndim != 1:
            raise ValueError(
                "levels must be one-dimensional."
            )

        if not np.all(
            np.diff(levels_array) > 0
        ):
            raise ValueError(
                "levels must be strictly increasing."
            )

        number_of_colors = (
            len(levels_array) - 1
        )

    source_cmap = plt.get_cmap(
        cmap_name,
        number_of_colors,
    )

    color_array = source_cmap(
        np.linspace(
            0,
            1,
            number_of_colors,
        )
    )

    if customize_low_end and number_of_colors >= 3:
        color_array[0] = [
            128 / 256,
            100 / 256,
            128 / 256,
            1,
        ]

        color_array[1] = [
            0.7,
            0.1,
            0.7,
            1,
        ]

        color_array[2] = (
            color_array[1].copy()
        )

    discrete_cmap = ListedColormap(
        color_array
    )

    norm = BoundaryNorm(
        levels_array,
        discrete_cmap.N,
    )

    return (
        discrete_cmap,
        norm,
        levels_array,
    )

# =============================================================================
# Time-series and additional spatial plots
# =============================================================================

def plot_monthly_mean_timeseries(
    monthly_data: pd.DataFrame,
    *,
    product_columns: Sequence[str] = ("GPROF_V7", "GPROF_V8", "MRMS", "ERA5"),
    date_column: str = "year_month",
    product_labels: Mapping[str, str] | None = None,
    product_colors: Mapping[str, str] | None = None,
    product_markers: Mapping[str, str] | None = None,
    product_linestyles: Mapping[str, str] | None = None,
    ylabel: str = "Monthly mean precipitation (mm h$^{-1}$)",
    figure_size: tuple[float, float] = (12, 5.5),
    output_path: str | Path | None = None,
):
    """Plot common-sample monthly mean precipitation time series."""
    _validate_dataframe_columns(monthly_data, [date_column, *product_columns])
    labels = _resolve_mapping(product_columns, product_labels, DEFAULT_PRODUCT_LABELS)
    colors = _resolve_mapping(product_columns, product_colors, DEFAULT_PRODUCT_COLORS)
    markers = _resolve_mapping(product_columns, product_markers, DEFAULT_PRODUCT_MARKERS)
    linestyles = _resolve_mapping(product_columns, product_linestyles, DEFAULT_PRODUCT_LINESTYLES)
    fig, ax = plt.subplots(figsize=figure_size)
    for product in product_columns:
        ax.plot(monthly_data[date_column], monthly_data[product], label=labels[product], color=colors[product], marker=markers[product], linestyle=linestyles[product], linewidth=1.6, markersize=3.5)
    ax.set_xlabel("Month", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.tick_params(direction="in", top=True, right=True, labelsize=10)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()
    save_figure(fig, output_path, dpi=150)
    return fig, ax


def plot_seasonal_spatial_mean_maps(
    spatial_dataset,
    *,
    product_order: Sequence[str] = (
        "GPROF_V7",
        "GPROF_V8",
        "MRMS",
        "ERA5",
    ),
    season_order: Sequence[str] = (
        "DJF",
        "MAM",
        "JJA",
        "SON",
    ),
    product_labels: Mapping[str, str] | None = None,
    extent: tuple[float, float, float, float] = (
        -125.0,
        -66.0,
        24.0,
        50.0,
    ),
    levels: Sequence[float] = (
            0.000,
            0.010,
            0.020,
            0.035,
            0.050,
            0.075,
            0.100,
            0.150,
            0.225,
            0.300,
            0.450,
            0.600,

        ),
    cmap_name: str = "nipy_spectral",
    customize_low_end: bool = True,
    colorbar_ticks: Sequence[float] = (
        0.000,
        0.020,
        0.050,
        0.100,
        0.200,
        0.300,
        0.450,
        0.600,
        ),
    colorbar_label: str = (
        "Seasonal mean precipitation "
        "(mm h$^{-1}$)"
    ),
    figure_size: tuple[float, float] = (
        18,
        13,
    ),
    dpi: int = 150,
    title: str | None = None,
    colorbar_width_fraction: float = 0.54,
    colorbar_height: float = 0.018,
    colorbar_bottom: float = 0.045,
    output_path: str | Path | None = None,
    preliminary_label: str | None = None,
):
    """
    Plot seasonal spatial mean precipitation.

    Rows:
        DJF, MAM, JJA, SON

    Columns:
        GPROF V7, GPROF V8, MRMS, ERA5
    """

    if len(product_order) != 4:
        raise ValueError(
            "plot_seasonal_spatial_mean_maps expects "
            "exactly four products."
        )

    if len(season_order) != 4:
        raise ValueError(
            "plot_seasonal_spatial_mean_maps expects "
            "exactly four seasons."
        )

    labels = _resolve_mapping(
        product_order,
        product_labels,
        DEFAULT_PRODUCT_LABELS,
    )

    # ------------------------------------------------------------------
    # Coordinates and sorting
    # ------------------------------------------------------------------
    longitude_original = np.asarray(
        spatial_dataset[
            "longitude"
        ].values,
        dtype=float,
    )

    latitude_original = np.asarray(
        spatial_dataset[
            "latitude"
        ].values,
        dtype=float,
    )

    if (
        longitude_original.ndim != 1
        or latitude_original.ndim != 1
    ):
        raise ValueError(
            "Longitude and latitude must be one-dimensional."
        )

    longitude_sort = np.argsort(
        longitude_original
    )

    latitude_sort = np.argsort(
        latitude_original
    )

    longitude = longitude_original[
        longitude_sort
    ]

    latitude = latitude_original[
        latitude_sort
    ]

    map_extent = [
        max(
            float(extent[0]),
            float(np.nanmin(longitude)),
        ),
        min(
            float(extent[1]),
            float(np.nanmax(longitude)),
        ),
        max(
            float(extent[2]),
            float(np.nanmin(latitude)),
        ),
        min(
            float(extent[3]),
            float(np.nanmax(latitude)),
        ),
    ]

    # ------------------------------------------------------------------
    # Discrete colormap
    # ------------------------------------------------------------------
    boundaries = np.asarray(
        levels,
        dtype=float,
    )

    if boundaries.ndim != 1:
        raise ValueError(
            "levels must be one-dimensional."
        )

    if not np.all(
        np.diff(boundaries) > 0
    ):
        raise ValueError(
            "levels must be strictly increasing."
        )

    source_cmap = plt.get_cmap(
        cmap_name,
        len(boundaries),
    )

    color_array = source_cmap(
        np.linspace(
            0,
            1,
            len(boundaries),
        )
    )[:-1]

    if (
        customize_low_end
        and len(color_array) >= 3
    ):
        color_array[0] = [
            128 / 256,
            100 / 256,
            128 / 256,
            1,
        ]

        color_array[2] = (
            color_array[1].copy()
        )

        color_array[1] = [
            0.7,
            0.1,
            0.7,
            1,
        ]

    cmap = ListedColormap(
        color_array
    )

    norm = BoundaryNorm(
        boundaries,
        cmap.N,
    )

    cmap.set_bad(
        color="white"
    )

    cmap.set_under(
        cmap.colors[0]
    )

    cmap.set_over(
        cmap.colors[-1]
    )

    projection = ccrs.PlateCarree()

    # ------------------------------------------------------------------
    # Figure and map grid
    # ------------------------------------------------------------------
    fig = plt.figure(
        figsize=figure_size,
        dpi=dpi,
    )

    grid_spec = fig.add_gridspec(
        nrows=len(season_order),
        ncols=len(product_order),
        left=0.055,
        right=0.985,
        top=0.945 if title else 0.975,
        bottom=0.115,
        wspace=0.035,
        hspace=0.16,
    )

    axes = np.empty(
        (
            len(season_order),
            len(product_order),
        ),
        dtype=object,
    )

    first_artist = None
    panel_index = 0

    # ------------------------------------------------------------------
    # Seasonal panels
    # ------------------------------------------------------------------
    for row_index, season in enumerate(
        season_order
    ):
        if season not in spatial_dataset[
            "season"
        ].values:
            raise KeyError(
                f"Season {season!r} is not available "
                "in spatial_dataset."
            )

        seasonal_dataset = (
            spatial_dataset.sel(
                season=season
            )
        )

        for column_index, product in enumerate(
            product_order
        ):
            ax = fig.add_subplot(
                grid_spec[
                    row_index,
                    column_index,
                ],
                projection=projection,
            )

            axes[
                row_index,
                column_index,
            ] = ax

            variable_name = (
                f"{product}_mean"
            )

            if variable_name not in seasonal_dataset:
                raise KeyError(
                    f"Missing seasonal variable: "
                    f"{variable_name}"
                )

            field = np.asarray(
                seasonal_dataset[
                    variable_name
                ].values,
                dtype=float,
            )

            field = np.squeeze(
                field
            )

            expected_shape = (
                len(latitude_original),
                len(longitude_original),
            )

            if field.shape != expected_shape:
                raise ValueError(
                    f"{season} {product}: field shape "
                    f"{field.shape} does not match "
                    f"{expected_shape}."
                )

            field = field[
                latitude_sort,
                :,
            ]

            field = field[
                :,
                longitude_sort,
            ]

            ax.set_extent(
                map_extent,
                crs=projection,
            )

            ax.set_xlim(
                map_extent[0],
                map_extent[1],
            )

            ax.set_ylim(
                map_extent[2],
                map_extent[3],
            )

            artist = ax.contourf(
                longitude,
                latitude,
                field,
                levels=boundaries,
                cmap=cmap,
                norm=norm,
                transform=projection,
                extend="max",
            )

            if first_artist is None:
                first_artist = artist

            ax.coastlines(
                resolution="50m",
                linewidth=0.8,
                color="black",
                zorder=5,
            )

            ax.add_feature(
                cfeature.BORDERS.with_scale(
                    "50m"
                ),
                linewidth=0.7,
                edgecolor="black",
                zorder=5,
            )

            gridlines = ax.gridlines(
                crs=projection,
                draw_labels=True,
                linewidth=0.45,
                linestyle="--",
                color="gray",
                alpha=0.40,
            )

            gridlines.top_labels = False
            gridlines.right_labels = False

            gridlines.left_labels = (
                column_index == 0
            )

            gridlines.bottom_labels = (
                row_index
                == len(season_order) - 1
            )

            gridlines.xlabel_style = {
                "size": 10,
                "weight": "bold",
            }

            gridlines.ylabel_style = {
                "size": 10,
                "weight": "bold",
            }

            if row_index == 0:
                ax.set_title(
                    labels[
                        product
                    ],
                    fontsize=14,
                    fontweight="bold",
                    pad=7,
                )

            panel_index += 1

            ax.text(
            0.0,
            1.025,
            f"({chr(96 + panel_index)})",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            clip_on=False,
            zorder=10,
        )

        # Season label to the left of each row.
        axes[
            row_index,
            0,
        ].text(
            -0.13,
            0.50,
            season,
            transform=axes[
                row_index,
                0,
            ].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            clip_on=False,
        )

    # ------------------------------------------------------------------
    # Finalize layout before adding the colorbar
    # ------------------------------------------------------------------
    fig.canvas.draw()

    # ------------------------------------------------------------------
    # Short centered horizontal colorbar
    # ------------------------------------------------------------------
    bottom_left_position = (
        axes[
            -1,
            0,
        ].get_position()
    )

    bottom_right_position = (
        axes[
            -1,
            -1,
        ].get_position()
    )

    full_row_left = (
        bottom_left_position.x0
    )

    full_row_right = (
        bottom_right_position.x1
    )

    full_row_width = (
        full_row_right
        - full_row_left
    )

    cbar_width = (
        full_row_width
        * colorbar_width_fraction
    )

    cbar_left = (
        full_row_left
        + 0.5
        * (
            full_row_width
            - cbar_width
        )
    )

    colorbar_axis = fig.add_axes(
        [
            cbar_left,
            colorbar_bottom,
            cbar_width,
            colorbar_height,
        ]
    )

    colorbar = fig.colorbar(
        first_artist,
        cax=colorbar_axis,
        orientation="horizontal",
        extend="max",
    )

    colorbar_ticks = np.asarray(
        colorbar_ticks,
        dtype=float,
    )

    colorbar_ticks = colorbar_ticks[
        (
            colorbar_ticks
            >= boundaries[0]
        )
        & (
            colorbar_ticks
            <= boundaries[-1]
        )
    ]

    colorbar.set_ticks(
        colorbar_ticks
    )

    colorbar.set_ticklabels(
        [
            f"{tick:.3f}"
            if tick < 1
            else f"{tick:g}"
            for tick in colorbar_ticks
        ]
    )

    colorbar.ax.tick_params(
        labelsize=12,
        length=5,
        width=1.0,
    )

    colorbar.set_label(
        colorbar_label,
        fontsize=13,
        fontweight="bold",
        labelpad=5,
    )

    if title:
        fig.suptitle(
            title,
            fontsize=17,
            fontweight="bold",
            y=0.985,
        )

    _add_preliminary_label(
        fig,
        preliminary_label,
    )

    save_figure(
        fig,
        output_path,
        dpi=dpi,
    )

    return fig, axes
def plot_spatial_precipitation_fraction_maps(
    spatial_dataset,
    *,
    product_order: Sequence[str] = (
        "GPROF_V7",
        "GPROF_V8",
        "MRMS",
        "ERA5",
    ),
    product_labels: Mapping[str, str] | None = None,
    extent: tuple[float, float, float, float] = (
        -125.0,
        -66.0,
        24.0,
        50.0,
    ),
    levels: Sequence[float] = (
        0,
        2,
        5,
        10,
        15,
        20,
        30,
        40,
        50,
        60,
        80,
        100,
    ),
    cmap_name: str = "nipy_spectral",
    colorbar_label: str = "Precipitation fraction (%)",
    figure_size: tuple[float, float] = (
        14,
        8.8,
    ),
    dpi: int = 150,
    title: str | None = None,
    output_path: str | Path | None = None,
    preliminary_label: str | None = None,
):
    """
    Plot wet-footprint precipitation fraction for four products
    in a 2 x 2 map layout.
    """

    if len(product_order) != 4:
        raise ValueError(
            "plot_spatial_precipitation_fraction_maps "
            "expects exactly four products."
        )

    labels = _resolve_mapping(
        product_order,
        product_labels,
        DEFAULT_PRODUCT_LABELS,
    )

    longitude_original = np.asarray(
        spatial_dataset["longitude"].values,
        dtype=float,
    )

    latitude_original = np.asarray(
        spatial_dataset["latitude"].values,
        dtype=float,
    )

    if (
        longitude_original.ndim != 1
        or latitude_original.ndim != 1
    ):
        raise ValueError(
            "Longitude and latitude must be one-dimensional."
        )

    longitude_sort = np.argsort(
        longitude_original
    )

    latitude_sort = np.argsort(
        latitude_original
    )

    longitude = longitude_original[
        longitude_sort
    ]

    latitude = latitude_original[
        latitude_sort
    ]

    map_extent = [
        max(
            float(extent[0]),
            float(np.nanmin(longitude)),
        ),
        min(
            float(extent[1]),
            float(np.nanmax(longitude)),
        ),
        max(
            float(extent[2]),
            float(np.nanmin(latitude)),
        ),
        min(
            float(extent[3]),
            float(np.nanmax(latitude)),
        ),
    ]

    boundaries = np.asarray(
        levels,
        dtype=float,
    )

    if boundaries.ndim != 1:
        raise ValueError(
            "levels must be one-dimensional."
        )

    if not np.all(
        np.diff(boundaries) > 0
    ):
        raise ValueError(
            "levels must be strictly increasing."
        )

    cmap, norm, boundaries = (
        create_discrete_colormap(
            vmin=float(boundaries[0]),
            vmax=float(boundaries[-1]),
            cmap_name=cmap_name,
            levels=boundaries,
        )
    )

    # Missing values are white.
    cmap.set_bad(
        "white"
    )

    # Values below and above the plotted range use the end colors.
    cmap.set_under(
        cmap.colors[0]
    )

    # cmap.set_over(
    #     cmap.colors[-1]
    # )

    projection = ccrs.PlateCarree()

    fig, axes = plt.subplots(
        2,
        2,
        figsize=figure_size,
        dpi=dpi,
        subplot_kw={
            "projection": projection,
        },
    )

    axes_flat = np.asarray(
        axes
    ).ravel()

    artists = []

    for panel_index, (
        ax,
        product,
    ) in enumerate(
        zip(
            axes_flat,
            product_order,
        ),
        start=1,
    ):
        variable_name = (
            f"{product}_precipitation_fraction"
        )

        if variable_name not in spatial_dataset:
            raise KeyError(
                f"Missing spatial variable: {variable_name}"
            )

        field = np.asarray(
            spatial_dataset[
                variable_name
            ].values,
            dtype=float,
        )

        field = np.squeeze(
            field
        )

        field = np.clip(
            field,
            0.0,
            100.0,
        )

        expected_shape = (
            len(latitude_original),
            len(longitude_original),
        )

        if field.shape != expected_shape:
            raise ValueError(
                f"{product}: field shape {field.shape} "
                f"does not match expected shape "
                f"{expected_shape}."
            )

        field = field[
            latitude_sort,
            :,
        ]

        field = field[
            :,
            longitude_sort,
        ]

        ax.set_extent(
            map_extent,
            crs=projection,
        )

        ax.set_xlim(
            map_extent[0],
            map_extent[1],
        )

        ax.set_ylim(
            map_extent[2],
            map_extent[3],
        )

        artist = ax.contourf(
        longitude,
        latitude,
        field,
        levels=boundaries,
        cmap=cmap,
        norm=norm,
        transform=projection,
        extend="neither",
    )

        # Force the over-range extension to use the final real color.
        # artist.cmap.set_over(
        #     cmap.colors[-1]
        # )

        # artist.changed()

        ax.coastlines(
            resolution="50m",
            linewidth=0.9,
            color="black",
            zorder=5,
        )

        ax.add_feature(
            cfeature.BORDERS.with_scale(
                "50m"
            ),
            linewidth=0.8,
            edgecolor="black",
            zorder=5,
        )

        row_index = (
            panel_index - 1
        ) // 2

        column_index = (
            panel_index - 1
        ) % 2

        gridlines = ax.gridlines(
            crs=projection,
            draw_labels=True,
            linewidth=0.5,
            linestyle="--",
            color="gray",
            alpha=0.45,
        )

        gridlines.top_labels = False
        gridlines.right_labels = False

        gridlines.left_labels = (
            column_index == 0
        )

        gridlines.bottom_labels = (
            row_index == 1
        )

        gridlines.xlabel_style = {
            "size": 12,
            "weight": "bold",
        }

        gridlines.ylabel_style = {
            "size": 12,
            "weight": "bold",
        }

        ax.set_title(
            f"({chr(96 + panel_index)}) "
            f"{labels[product]}",
            fontsize=14,
            fontweight="bold",
            pad=6,
        )

        artists.append(
            artist
        )

    if title:
        top_position = 0.92
    else:
        top_position = 0.965

    fig.subplots_adjust(
        left=0.055,
        right=0.985,
        top=top_position,
        bottom=0.16,
        wspace=0.055,
        hspace=0.055,
    )

    colorbar_axis = fig.add_axes(
            [
                0.27,   # left
                0.065,  # bottom
                0.46,   # width
                0.025,  # height
            ]
        )

    colorbar = fig.colorbar(
        artists[0],
        cax=colorbar_axis,
        orientation="horizontal",
        extend="neither",
    )

    colorbar_ticks = np.array(
        [
            0,
            2, 
            5,
            10,
            15,
            20,
            40,
            60,
            80,
            100,
        ],
        dtype=float,
    )

    colorbar.set_ticks(
        colorbar_ticks
    )

    colorbar.set_ticklabels(
        [
            f"{int(tick)}"
            for tick in colorbar_ticks
        ]
    )

    colorbar.ax.tick_params(
            labelsize=14,
            length=6,
            width=1.1,
        )

    colorbar.set_label(
            colorbar_label,
            fontsize=14,
            fontweight="bold",
            labelpad=6,
        )

    if title:
        fig.suptitle(
            title,
            fontsize=16,
            fontweight="bold",
            y=0.985,
        )

    _add_preliminary_label(
        fig,
        preliminary_label,
    )

    save_figure(
        fig,
        output_path,
        dpi=dpi,
    )

    return fig, axes


def plot_spatial_precipitation_fractions_and_differences(
    spatial_fraction_dataset,
    spatial_difference_dataset=None,
    *,
    extent: tuple[float, float, float, float] = (
        -125.0,
        -66.0,
        24.0,
        50.0,
    ),
    fraction_levels: Sequence[float] = (
        0,
        2,
        5,
        8,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
    ),
    fraction_cmap_name: str = "nipy_spectral",
    customize_fraction_low_end: bool = True,
    difference_vmin: float = -20.0,
    difference_vmax: float = 20.0,
    difference_step: float = 2.5,
    difference_cmap_name: str = "RdBu_r",
    fraction_colorbar_ticks: Sequence[float] = (
        0,
        2,
        5,
        10,
        15,
        20,
        30,
        40,
    ),
    difference_colorbar_ticks: Sequence[float] | None = (
        -20,
        -15,
        -10,
        -5,
        0,
        5,
        10,
        15,
        20,
    ),
    fraction_colorbar_label: str = (
        "Precipitation fraction (%)"
    ),
    difference_colorbar_label: str = (
        "Precipitation-fraction difference "
        "(percentage points)"
    ),
    figure_size: tuple[float, float] = (
        16,
        15,
    ),
    dpi: int = 150,
    title: str | None = None,
    output_path: str | Path | None = None,
    preliminary_label: str | None = None,
):
    """
    Plot precipitation fractions and corresponding differences.

    Layout
    ------
    MRMS fraction       | MRMS - GPROF V8
    GPROF V8 fraction   | MRMS - GPROF V7
    GPROF V7 fraction   | MRMS - ERA5
    ERA5 fraction       | GPROF V7 - GPROF V8

    The left-column scale ends at 40%. Values above 40% use the
    final fraction-colormap color.

    Difference units are percentage points.
    """

    required_fraction_variables = (
        "MRMS_precipitation_fraction",
        "GPROF_V8_precipitation_fraction",
        "GPROF_V7_precipitation_fraction",
        "ERA5_precipitation_fraction",
    )

    missing_fraction_variables = [
        variable
        for variable in required_fraction_variables
        if variable not in spatial_fraction_dataset
    ]

    if missing_fraction_variables:
        raise KeyError(
            "Missing required precipitation-fraction variables: "
            + ", ".join(missing_fraction_variables)
        )

    # ------------------------------------------------------------------
    # Calculate differences internally when not supplied
    # ------------------------------------------------------------------
    if spatial_difference_dataset is None:
        import xarray as xr

        spatial_difference_dataset = xr.Dataset(
            {
                "MRMS_minus_GPROF_V8": (
                    spatial_fraction_dataset[
                        "MRMS_precipitation_fraction"
                    ]
                    - spatial_fraction_dataset[
                        "GPROF_V8_precipitation_fraction"
                    ]
                ),
                "MRMS_minus_GPROF_V7": (
                    spatial_fraction_dataset[
                        "MRMS_precipitation_fraction"
                    ]
                    - spatial_fraction_dataset[
                        "GPROF_V7_precipitation_fraction"
                    ]
                ),
                "MRMS_minus_ERA5": (
                    spatial_fraction_dataset[
                        "MRMS_precipitation_fraction"
                    ]
                    - spatial_fraction_dataset[
                        "ERA5_precipitation_fraction"
                    ]
                ),
                "GPROF_V7_minus_GPROF_V8": (
                    spatial_fraction_dataset[
                        "GPROF_V7_precipitation_fraction"
                    ]
                    - spatial_fraction_dataset[
                        "GPROF_V8_precipitation_fraction"
                    ]
                ),
            }
        )

    required_difference_variables = (
            "GPROF_V8_minus_MRMS",
            "GPROF_V7_minus_MRMS",
            "ERA5_minus_MRMS",
            "GPROF_V7_minus_GPROF_V8",
        )

    missing_difference_variables = [
        variable
        for variable in required_difference_variables
        if variable not in spatial_difference_dataset
    ]

    if missing_difference_variables:
        raise KeyError(
            "Missing required fraction-difference variables: "
            + ", ".join(missing_difference_variables)
        )

    if difference_vmax <= difference_vmin:
        raise ValueError(
            "difference_vmax must exceed difference_vmin."
        )

    if difference_step <= 0:
        raise ValueError(
            "difference_step must be positive."
        )

    # ------------------------------------------------------------------
    # Coordinates
    # ------------------------------------------------------------------
    longitude_original = np.asarray(
        spatial_fraction_dataset[
            "longitude"
        ].values,
        dtype=float,
    )

    latitude_original = np.asarray(
        spatial_fraction_dataset[
            "latitude"
        ].values,
        dtype=float,
    )

    if (
        longitude_original.ndim != 1
        or latitude_original.ndim != 1
    ):
        raise ValueError(
            "Longitude and latitude must be one-dimensional."
        )

    longitude_sort = np.argsort(
        longitude_original
    )

    latitude_sort = np.argsort(
        latitude_original
    )

    longitude = longitude_original[
        longitude_sort
    ]

    latitude = latitude_original[
        latitude_sort
    ]

    map_extent = [
        max(
            float(extent[0]),
            float(np.nanmin(longitude)),
        ),
        min(
            float(extent[1]),
            float(np.nanmax(longitude)),
        ),
        max(
            float(extent[2]),
            float(np.nanmin(latitude)),
        ),
        min(
            float(extent[3]),
            float(np.nanmax(latitude)),
        ),
    ]

    # ------------------------------------------------------------------
    # Fraction colormap
    # ------------------------------------------------------------------
    fraction_boundaries = np.asarray(
        fraction_levels,
        dtype=float,
    )

    if fraction_boundaries.ndim != 1:
        raise ValueError(
            "fraction_levels must be one-dimensional."
        )

    if not np.all(
        np.diff(fraction_boundaries) > 0
    ):
        raise ValueError(
            "fraction_levels must be strictly increasing."
        )

    number_of_fraction_intervals = (
        len(fraction_boundaries) - 1
    )

    fraction_source_cmap = plt.get_cmap(
        fraction_cmap_name,
        number_of_fraction_intervals + 1,
    )

    fraction_colors = fraction_source_cmap(
        np.linspace(
            0,
            1,
            number_of_fraction_intervals + 1,
        )
    )[:-1]

    if (
        customize_fraction_low_end
        and len(fraction_colors) >= 3
    ):
        fraction_colors[0] = [
            128 / 256,
            100 / 256,
            128 / 256,
            1,
        ]

        fraction_colors[2] = (
            fraction_colors[1].copy()
        )

        fraction_colors[1] = [
            0.7,
            0.1,
            0.7,
            1,
        ]

    fraction_cmap = ListedColormap(
        fraction_colors
    )

    fraction_norm = BoundaryNorm(
        fraction_boundaries,
        fraction_cmap.N,
    )

    fraction_cmap.set_bad(
        color="white"
    )

    fraction_cmap.set_under(
        fraction_cmap.colors[0]
    )

    fraction_cmap.set_over(
        fraction_cmap.colors[-1]
    )

    # ------------------------------------------------------------------
    # Difference colormap
    # ------------------------------------------------------------------
    epsilon = (
        1.0e-10
        * max(
            1.0,
            abs(
                difference_vmax
                - difference_vmin
            ),
        )
    )

    difference_boundaries = np.arange(
        difference_vmin,
        difference_vmax + epsilon,
        difference_step,
        dtype=float,
    )

    if (
        difference_boundaries[-1]
        < difference_vmax - epsilon
    ):
        difference_boundaries = np.append(
            difference_boundaries,
            difference_vmax,
        )

    difference_boundaries[-1] = (
        difference_vmax
    )

    number_of_difference_intervals = (
        len(difference_boundaries) - 1
    )

    difference_source_cmap = plt.get_cmap(
        difference_cmap_name,
        number_of_difference_intervals,
    )

    difference_colors = difference_source_cmap(
        np.linspace(
            0,
            1,
            number_of_difference_intervals,
        )
    )

    difference_cmap = ListedColormap(
        difference_colors
    )

    difference_norm = BoundaryNorm(
        difference_boundaries,
        difference_cmap.N,
    )

    difference_cmap.set_bad(
        color="white"
    )

    difference_cmap.set_under(
        difference_cmap.colors[0]
    )

    difference_cmap.set_over(
        difference_cmap.colors[-1]
    )

    # ------------------------------------------------------------------
    # Panel definitions
    # ------------------------------------------------------------------
    fraction_rows = (
        (
            "MRMS_precipitation_fraction",
            "MRMS",
        ),
        (
            "GPROF_V8_precipitation_fraction",
            "GPROF V8",
        ),
        (
            "GPROF_V7_precipitation_fraction",
            "GPROF V7",
        ),
        (
            "ERA5_precipitation_fraction",
            "ERA5",
        ),
    )

    difference_rows = (
            (
                "GPROF_V8_minus_MRMS",
                "GPROF V8 − MRMS",
            ),
            (
                "GPROF_V7_minus_MRMS",
                "GPROF V7 − MRMS",
            ),
            (
                "ERA5_minus_MRMS",
                "ERA5 − MRMS",
            ),
            (
                "GPROF_V7_minus_GPROF_V8",
                "GPROF V7 − GPROF V8",
            ),
        )

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    projection = ccrs.PlateCarree()

    fig = plt.figure(
        figsize=figure_size,
        dpi=dpi,
    )

    grid_spec = fig.add_gridspec(
        nrows=4,
        ncols=2,
        left=0.055,
        right=0.985,
        top=0.955 if title is None else 0.925,
        bottom=0.105,
        wspace=0.045,
        hspace=0.25,
    )

    axes = np.empty(
        (
            4,
            2,
        ),
        dtype=object,
    )

    fraction_artist = None
    difference_artist = None
    panel_index = 0

    # ------------------------------------------------------------------
    # Panels
    # ------------------------------------------------------------------
    for row_index in range(4):
        fraction_variable, fraction_label = (
            fraction_rows[row_index]
        )

        difference_variable, difference_label = (
            difference_rows[row_index]
        )

        for column_index in range(2):
            ax = fig.add_subplot(
                grid_spec[
                    row_index,
                    column_index,
                ],
                projection=projection,
            )

            axes[
                row_index,
                column_index,
            ] = ax

            if column_index == 0:
                source_dataset = (
                    spatial_fraction_dataset
                )

                variable_name = (
                    fraction_variable
                )

                panel_label = (
                    fraction_label
                )

                panel_cmap = (
                    fraction_cmap
                )

                panel_norm = (
                    fraction_norm
                )

                panel_levels = (
                    fraction_boundaries
                )

                panel_extend = "max"

            else:
                source_dataset = (
                    spatial_difference_dataset
                )

                variable_name = (
                    difference_variable
                )

                panel_label = (
                    difference_label
                )

                panel_cmap = (
                    difference_cmap
                )

                panel_norm = (
                    difference_norm
                )

                panel_levels = (
                    difference_boundaries
                )

                panel_extend = "both"

            field = np.asarray(
                source_dataset[
                    variable_name
                ].values,
                dtype=float,
            )

            field = np.squeeze(
                field
            )

            expected_shape = (
                len(latitude_original),
                len(longitude_original),
            )

            if field.shape != expected_shape:
                raise ValueError(
                    f"{variable_name}: field shape "
                    f"{field.shape} does not match "
                    f"{expected_shape}."
                )

            if column_index == 0:
                # Preserve values above the displayed upper boundary.
                # contourf and extend='max' will show them with the
                # final fraction-colormap color.
                field = np.where(
                    np.isfinite(field),
                    np.maximum(
                        field,
                        0.0,
                    ),
                    np.nan,
                )

            field = field[
                latitude_sort,
                :,
            ]

            field = field[
                :,
                longitude_sort,
            ]

            ax.set_extent(
                map_extent,
                crs=projection,
            )

            ax.set_xlim(
                map_extent[0],
                map_extent[1],
            )

            ax.set_ylim(
                map_extent[2],
                map_extent[3],
            )

            artist = ax.contourf(
                longitude,
                latitude,
                field,
                levels=panel_levels,
                cmap=panel_cmap,
                norm=panel_norm,
                transform=projection,
                extend=panel_extend,
            )

            # Explicitly refresh over/under colors used by ContourSet.
            artist.cmap.set_under(
                panel_cmap.colors[0]
            )

            artist.cmap.set_over(
                panel_cmap.colors[-1]
            )

            artist.changed()

            if (
                column_index == 0
                and fraction_artist is None
            ):
                fraction_artist = artist

            if (
                column_index == 1
                and difference_artist is None
            ):
                difference_artist = artist

            ax.coastlines(
                resolution="50m",
                linewidth=0.9,
                color="black",
                zorder=5,
            )

            ax.add_feature(
                cfeature.BORDERS.with_scale(
                    "50m"
                ),
                linewidth=0.8,
                edgecolor="black",
                zorder=5,
            )

            gridlines = ax.gridlines(
                crs=projection,
                draw_labels=True,
                linewidth=0.5,
                linestyle="--",
                color="gray",
                alpha=0.45,
            )

            gridlines.top_labels = False
            gridlines.right_labels = False

            gridlines.left_labels = (
                column_index == 0
            )

            gridlines.bottom_labels = (
                row_index == 3
            )

            gridlines.xlabel_style = {
                "size": 10,
                "weight": "bold",
            }

            gridlines.ylabel_style = {
                "size": 10,
                "weight": "bold",
            }

            panel_index += 1

            ax.set_title(
                f"({chr(96 + panel_index)}) "
                f"{panel_label}",
                fontsize=14,
                fontweight="bold",
                pad=6,
            )

    fig.canvas.draw()

    # ------------------------------------------------------------------
    # Fraction colorbar
    # ------------------------------------------------------------------
    fraction_colorbar_axis = inset_axes(
        axes[3, 0],
        width="94%",
        height="9%",
        loc="lower center",
        bbox_to_anchor=(
            0.0,
            -0.34,
            1.0,
            1.0,
        ),
        bbox_transform=axes[3, 0].transAxes,
        borderpad=0,
    )

    # Use the actual contour artist. This preserves the customized
    # over-range color and extension behavior.
    fraction_colorbar = fig.colorbar(
        fraction_artist,
        cax=fraction_colorbar_axis,
        orientation="horizontal",
        extend="max",
    )

    fraction_ticks = np.asarray(
        fraction_colorbar_ticks,
        dtype=float,
    )

    fraction_ticks = fraction_ticks[
        (
            fraction_ticks
            >= fraction_boundaries[0]
        )
        & (
            fraction_ticks
            <= fraction_boundaries[-1]
        )
    ]

    fraction_colorbar.set_ticks(
        fraction_ticks
    )

    fraction_colorbar.set_ticklabels(
        [
            f"{int(round(tick))}"
            if np.isclose(
                tick,
                round(tick),
            )
            else f"{tick:g}"
            for tick in fraction_ticks
        ]
    )

    fraction_colorbar.ax.tick_params(
        labelsize=11,
        length=5,
    )

    fraction_colorbar.set_label(
        fraction_colorbar_label,
        fontsize=12,
        fontweight="bold",
        labelpad=5,
    )

    # ------------------------------------------------------------------
    # Difference colorbar
    # ------------------------------------------------------------------
    difference_colorbar_axis = inset_axes(
        axes[3, 1],
        width="94%",
        height="9%",
        loc="lower center",
        bbox_to_anchor=(
            0.0,
            -0.34,
            1.0,
            1.0,
        ),
        bbox_transform=axes[3, 1].transAxes,
        borderpad=0,
    )

    difference_colorbar = fig.colorbar(
        difference_artist,
        cax=difference_colorbar_axis,
        orientation="horizontal",
        extend="both",
    )

    if difference_colorbar_ticks is None:
        difference_ticks = np.arange(
            difference_vmin,
            difference_vmax
            + 0.5 * difference_step,
            difference_step,
            dtype=float,
        )
    else:
        difference_ticks = np.asarray(
            difference_colorbar_ticks,
            dtype=float,
        )

    difference_ticks = difference_ticks[
        (
            difference_ticks
            >= difference_vmin
        )
        & (
            difference_ticks
            <= difference_vmax
        )
    ]

    difference_colorbar.set_ticks(
        difference_ticks
    )

    difference_colorbar.set_ticklabels(
        [
            f"{int(round(tick))}"
            if np.isclose(
                tick,
                round(tick),
            )
            else f"{tick:.1f}"
            for tick in difference_ticks
        ]
    )

    difference_colorbar.ax.tick_params(
        labelsize=11,
        length=5,
    )

    difference_colorbar.set_label(
        difference_colorbar_label,
        fontsize=12,
        fontweight="bold",
        labelpad=5,
    )

    if title:
        fig.suptitle(
            title,
            fontsize=17,
            fontweight="bold",
            y=0.985,
        )

    _add_preliminary_label(
        fig,
        preliminary_label,
    )

    save_figure(
        fig,
        output_path,
        dpi=dpi,
    )

    return fig, axes
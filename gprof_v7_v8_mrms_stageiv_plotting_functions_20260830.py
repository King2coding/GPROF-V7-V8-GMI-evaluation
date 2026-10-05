"""Publication-oriented plots for the GPROF V7/V8 manuscript assessment."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
from matplotlib.ticker import (
    AutoMinorLocator,
    FixedLocator,
    FormatStrFormatter,
    FuncFormatter,
)
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# =============================================================================
# VISUAL CONSTANTS AND SHARED HELPERS
# =============================================================================

PRODUCT_LABELS = {
    "GPROF_V7": "GPROF V7",
    "GPROF_V8": "GPROF V8",
    "MRMS": "MRMS",
    "StageIV": "Stage IV",
    "ERA5": "ERA5",
}
PRODUCT_COLORS = {
    "GPROF_V7": "#1f5a94",
    "GPROF_V8": "#e28e2c",
    "MRMS": "#383838",
    "StageIV": "#8a4f8f",
    "ERA5": "#71843f",
}
PRODUCT_MARKERS = {
    "GPROF_V7": "o",
    "GPROF_V8": "s",
    "MRMS": "^",
    "StageIV": "D",
    "ERA5": "v",
}
PRODUCT_LINESTYLES = {
    "GPROF_V7": "--",
    "GPROF_V8": "-",
    "MRMS": "-.",
    "StageIV": ":",
    "ERA5": (0, (5, 2, 1, 2)),
}
CONUS_EXTENT = (-125.0, -66.0, 24.0, 50.0)


def save_figure(fig, output_path: str | Path | None, *, dpi: int = 150) -> None:
    """Save a figure with the established tight, white-background convention."""

    if output_path is None:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")


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



def _require_columns(data: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in data]
    if missing:
        raise KeyError("Missing plotting columns: " + ", ".join(missing))


def _finish_figure(fig, output_path: str | Path | None, dpi: int) -> None:
    if output_path is not None:
        save_figure(fig, output_path, dpi=dpi)


def _format_map_axis(ax, *, extent, left_labels: bool, bottom_labels: bool) -> None:
    projection = ccrs.PlateCarree()
    ax.set_extent(extent, crs=projection)
    ax.coastlines(resolution="50m", linewidth=0.65, color="#303030", zorder=5)
    ax.add_feature(
        cfeature.BORDERS.with_scale("50m"), linewidth=0.55, edgecolor="#303030", zorder=5
    )
    gridlines = ax.gridlines(
        crs=projection,
        draw_labels=True,
        linewidth=0.35,
        linestyle="--",
        color="#777777",
        alpha=0.4,
    )
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.left_labels = left_labels
    gridlines.bottom_labels = bottom_labels
    gridlines.xlabel_style = {"size": 10}
    gridlines.ylabel_style = {"size": 10}


# =============================================================================
# RAIN/SNOW PERFORMANCE
# =============================================================================

def plot_multireference_performance_diagram(
    categorical_metrics: pd.DataFrame,
    *,
    phase: str,
    reference_order: Sequence[str],
    threshold: float,
    product_order: Sequence[str] = (
        "GPROF_V7",
        "GPROF_V8",
        "ERA5",
    ),
    dpi: int = 150,
    output_path: str | Path | None = None,
):
    """
    Plot separate Roebber performance-diagram panels for each reference.

    All evaluated products are shown using the common-valid footprint
    sample already established in the analysis functions.

    Rain:
        GPROF V7, GPROF V8, and ERA5 can be shown against both
        MRMS and Stage IV.

    Snow:
        GPROF V7, GPROF V8, and ERA5 can be shown against Stage IV.
    """

    _require_columns(
        categorical_metrics,
        [
            "phase",
            "reference_label",
            "product",
            "success_ratio",
            "POD",
            "N",
        ],
    )

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    figure, axes = plt.subplots(
        1,
        len(reference_order),
        figsize=(
            8.5 * len(reference_order),
            7.5,
        ),
        dpi=dpi,
        squeeze=False,
    )

    axes = axes.ravel()

    # ------------------------------------------------------------------
    # Reference panels
    # ------------------------------------------------------------------
    for axis, reference in zip(
        axes,
        reference_order,
    ):

        # Use the established Roebber background.
        draw_performance_background(
            axis,
            contour_label_fontsize=8,
            bias_label_fontsize=8,
        )

        subset = categorical_metrics.loc[
            categorical_metrics["phase"].eq(
                phase
            )
            & categorical_metrics[
                "reference_label"
            ].eq(reference)
        ].copy()

        # --------------------------------------------------------------
        # Plot products
        # --------------------------------------------------------------
        plotted_products = []

        for product in product_order:

            row = subset.loc[
                subset["product"].eq(product)
            ]

            if row.empty:
                continue

            row = row.iloc[0]

            success_ratio = float(
                row["success_ratio"]
            )

            pod = float(
                row["POD"]
            )

            if not (
                np.isfinite(success_ratio)
                and np.isfinite(pod)
            ):
                continue

            axis.scatter(
                success_ratio,
                pod,
                s=120,
                marker=PRODUCT_MARKERS[
                    product
                ],
                facecolor=PRODUCT_COLORS[
                    product
                ],
                edgecolor="black",
                linewidth=0.9,
                label=PRODUCT_LABELS[
                    product
                ],
                zorder=8,
            )

            plotted_products.append(
                product
            )

        # --------------------------------------------------------------
        # Common sample count
        # --------------------------------------------------------------
        if not subset.empty:

            counts = (
                subset.loc[
                    subset["product"].isin(
                        plotted_products
                    ),
                    "N",
                ]
                .dropna()
                .astype(int)
                .unique()
            )

            if len(counts) == 1:
                sample_text = (
                    f"N = {counts[0]:,}"
                )
            else:
                sample_text = (
                    "N varies"
                )

            axis.text(
                0.03,
                0.04,
                sample_text,
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        # --------------------------------------------------------------
        # Panel title and legend
        # --------------------------------------------------------------
        axis.set_title(
            (
                f"{PRODUCT_LABELS.get(reference, reference)} "
                "reference"
            ),
            fontsize=14,
            fontweight="bold",
            pad=10,
        )

        if plotted_products:
            axis.legend(
                frameon=False,
                loc="lower right",
                fontsize=10,
            )

    # ------------------------------------------------------------------
    # Figure title
    # ------------------------------------------------------------------
    figure.suptitle(
        (
            f"{phase.capitalize()} detection performance | "
            f"wet threshold ≥ {threshold:g} mm h$^{{-1}}$"
        ),
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )

    figure.tight_layout(
        rect=(
            0,
            0,
            1,
            0.95,
        )
    )

    _finish_figure(
        figure,
        output_path,
        dpi,
    )

    return (
        figure,
        axes,
    )


def plot_quantitative_metric_bars(
    continuous_metrics: pd.DataFrame,
    *,
    phase: str = "rain",
    reference_order: Sequence[str] = (
        "MRMS",
        "StageIV",
    ),
    product_order: Sequence[str] = (
        "GPROF_V7",
        "GPROF_V8",
        "ERA5",
    ),
    metric_order: Sequence[str] = (
        "correlation",
        "RMSE",
        "relative_bias_percent",
    ),
    dpi: int = 150,
    output_path: str | Path | None = None,
):
    """
    Plot rain quantitative performance as a 3-row x 2-column figure.

    Rows
    ----
    1. Correlation coefficient
    2. RMSE
    3. Relative bias

    Columns
    -------
    1. MRMS reference
    2. Stage IV reference

    Products
    --------
    GPROF V7
    GPROF V8
    ERA5
    """

    _require_columns(
        continuous_metrics,
        [
            "phase",
            "reference_label",
            "product",
            "N",
            "correlation",
            "RMSE",
            "relative_bias_percent",
        ],
    )

    # ------------------------------------------------------------------
    # Metric labels and axis labels
    # ------------------------------------------------------------------
    metric_labels = {
        "correlation": "CC",
        "RMSE": "RMSE (mm h$^{-1}$)",
        "relative_bias_percent": "Relative bias (%)",
    }

    # ------------------------------------------------------------------
    # Restrict to requested phase
    # ------------------------------------------------------------------
    phase_metrics = continuous_metrics.loc[
        continuous_metrics["phase"].eq(phase)
    ].copy()

    if phase_metrics.empty:
        raise ValueError(
            f"No continuous metrics available for phase={phase!r}."
        )

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        nrows=len(metric_order),
        ncols=len(reference_order),
        figsize=(11.5, 10.0),
        dpi=dpi,
        squeeze=False,
    )

    bar_width = 0.56

    x = np.arange(
        len(product_order),
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Panels
    # ------------------------------------------------------------------
    for row_index, metric in enumerate(
        metric_order
    ):

        for column_index, reference in enumerate(
            reference_order
        ):

            ax = axes[
                row_index,
                column_index,
            ]

            subset = phase_metrics.loc[
                phase_metrics[
                    "reference_label"
                ].eq(reference)
            ].copy()

            # ----------------------------------------------------------
            # Preserve requested product order
            # ----------------------------------------------------------
            subset = (
                subset
                .set_index("product")
                .reindex(product_order)
            )

            values = subset[
                metric
            ].to_numpy(dtype=float)

            bars = ax.bar(
                x,
                values,
                width=bar_width,
                color=[
                    PRODUCT_COLORS[
                        product
                    ]
                    for product in product_order
                ],
                edgecolor="black",
                linewidth=0.7,
            )

            # ----------------------------------------------------------
            # Zero reference line for bias
            # ----------------------------------------------------------
            if metric == "relative_bias_percent":
                ax.axhline(
                    0.0,
                    color="black",
                    linewidth=0.9,
                    zorder=1,
                )

            # ----------------------------------------------------------
            # Bar annotations
            # ----------------------------------------------------------
            for bar, value in zip(
                bars,
                values,
            ):

                if not np.isfinite(value):
                    continue

                if metric == "correlation":
                    label = f"{value:.2f}"

                elif metric == "RMSE":
                    label = f"{value:.3f}"

                else:
                    label = f"{value:.1f}"

                # Position annotation correctly for
                # positive and negative bars.
                if value >= 0:
                    y = (
                        bar.get_height()
                        + 0.02
                        * max(
                            1.0,
                            abs(value),
                        )
                    )
                    va = "bottom"
                else:
                    y = (
                        bar.get_height()
                        - 0.02
                        * max(
                            1.0,
                            abs(value),
                        )
                    )
                    va = "top"

                ax.text(
                    bar.get_x()
                    + 0.5 * bar.get_width(),
                    y,
                    label,
                    ha="center",
                    va=va,
                    fontsize=10,
                    fontweight="bold",
                )

            # ----------------------------------------------------------
            # Axis labels
            # ----------------------------------------------------------
            ax.set_xticks(
                x
            )

            # Only show product names on bottom row.
            if row_index == (
                len(metric_order) - 1
            ):
                ax.set_xticklabels(
                    [
                        PRODUCT_LABELS[
                            product
                        ]
                        for product in product_order
                    ],
                    fontsize=10,
                )
            else:
                ax.set_xticklabels(
                    []
                )

            # Y labels only on left column.
            if column_index == 0:
                ax.set_ylabel(
                    metric_labels[
                        metric
                    ],
                    fontsize=12,
                    fontweight="bold",
                )

            # ----------------------------------------------------------
            # Reference titles on top row
            # ----------------------------------------------------------
            if row_index == 0:

                reference_label = (
                    PRODUCT_LABELS.get(
                        reference,
                        reference,
                    )
                )

                # Common N for this reference.
                counts = (
                    subset["N"]
                    .dropna()
                    .astype(int)
                    .unique()
                )

                if len(counts) == 1:
                    panel_title = (
                        f"{reference_label} reference "
                        f"(N={counts[0]:,})"
                    )
                else:
                    panel_title = (
                        f"{reference_label} reference"
                    )

                ax.set_title(
                    panel_title,
                    fontsize=13,
                    fontweight="bold",
                    pad=8,
                )

            # ----------------------------------------------------------
            # Metric-specific limits
            # ----------------------------------------------------------
            if metric == "correlation":
                ax.set_ylim(
                    0.0,
                    max(
                        0.6,
                        np.nanmax(values) * 1.25,
                    ),
                )

            elif metric == "RMSE":
                ax.set_ylim(
                    0.0,
                    np.nanmax(values) * 1.25,
                )

            elif metric == "relative_bias_percent":

                finite_values = values[
                    np.isfinite(values)
                ]

                if finite_values.size > 0:

                    lower = min(
                        0.0,
                        np.nanmin(finite_values),
                    )

                    upper = max(
                        0.0,
                        np.nanmax(finite_values),
                    )

                    padding = (
                        0.15
                        * max(
                            1.0,
                            upper - lower,
                        )
                    )

                    ax.set_ylim(
                        lower - padding,
                        upper + padding,
                    )

            # ----------------------------------------------------------
            # Styling
            # ----------------------------------------------------------
            ax.grid(
                axis="y",
                linestyle="--",
                linewidth=0.5,
                alpha=0.35,
            )

            ax.tick_params(
                axis="both",
                labelsize=10,
            )

    # ------------------------------------------------------------------
    # Shared legend
    # ------------------------------------------------------------------
    legend_handles = [
        Patch(
            facecolor=PRODUCT_COLORS[
                product
            ],
            edgecolor="black",
            label=PRODUCT_LABELS[
                product
            ],
        )
        for product in product_order
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=len(product_order),
        frameon=False,
        fontsize=11,
    )

    # ------------------------------------------------------------------
    # Figure title
    # ------------------------------------------------------------------
    fig.suptitle(
        f"{phase.capitalize()} quantitative performance",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    fig.tight_layout(
        rect=(
            0.03,
            0.03,
            0.98,
            0.94,
        )
    )

    _finish_figure(
        fig,
        output_path,
        dpi,
    )

    return (
        fig,
        axes,
    )


# =============================================================================
# DISTRIBUTIONS AND TEMPERATURE DEPENDENCE
# =============================================================================

def format_bin_tick_labels(bin_left_edges):
    """
    Format lower-edge bin labels using the legacy plotting style.

    Examples
    --------
    0.1
    0.2
    0.5
    1
    2
    4
    8
    16
    32
    64
    """

    labels = []

    for value in bin_left_edges:

        if value >= 1:
            labels.append(f"{int(value)}")
        else:
            labels.append(f"{value:g}")

    return labels

def plot_occurrence_and_volume_distributions(
    rain_distribution_data: pd.DataFrame,
    snow_distribution_data: pd.DataFrame,
    *,
    rain_product_order: Sequence[str] = (
        "GPROF_V7",
        "GPROF_V8",
        "MRMS",
        "StageIV",
        "ERA5",
    ),
    snow_product_order: Sequence[str] = (
        "GPROF_V7",
        "GPROF_V8",
        "StageIV",
        "ERA5",
    ),
    dpi: int = 150,
    output_path: str | Path | None = None,
):
    """
    Plot wet-occurrence and precipitation-volume intensity distributions.

    Layout
    ------
    Row 1: high-confidence rain
    Row 2: high-confidence snow

    Column 1: wet-occurrence contribution
    Column 2: precipitation-volume contribution

    Each panel has its own x-axis labels.
    """

    required_columns = [
        "phase",
        "product",
        "bin_index",
        "bin_label",
        "occurrence_percent",
        "volume_percent",
    ]

    _require_columns(
        rain_distribution_data,
        required_columns,
    )

    _require_columns(
        snow_distribution_data,
        required_columns,
    )

    rain = rain_distribution_data.loc[
        rain_distribution_data[
            "phase"
        ].eq("rain")
    ].copy()

    snow = snow_distribution_data.loc[
        snow_distribution_data[
            "phase"
        ].eq("snow")
    ].copy()

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    figure, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(14, 10),
        dpi=dpi,
        sharex=False,
        sharey=False,
    )

    panel_configuration = (
        (
            axes[0, 0],
            rain,
            rain_product_order,
            "occurrence_percent",
            "(a) Rain — occurrence",
            "Occurrence PDF [%]",
        ),
        (
            axes[0, 1],
            rain,
            rain_product_order,
            "volume_percent",
            "(b) Rain — precipitation volume",
            "Volume PDF [%]",
        ),
        (
            axes[1, 0],
            snow,
            snow_product_order,
            "occurrence_percent",
            "(c) Snow — occurrence",
            "Occurrence PDF [%]",
        ),
        (
            axes[1, 1],
            snow,
            snow_product_order,
            "volume_percent",
            "(d) Snow — precipitation volume",
            "Volume PDF [%]",
        ),
    )

    # ------------------------------------------------------------------
    # Panels
    # ------------------------------------------------------------------
    for (
        axis,
        working,
        product_order,
        metric,
        panel_title,
        ylabel,
    ) in panel_configuration:

        bins = (
            working[
                [
                    "bin_index",
                    "bin_left",
                    "bin_label",
                ]
            ]
            .drop_duplicates()
            .sort_values("bin_index")
        )

        positions = np.arange(
            len(bins)
        )

        for product in product_order:

            subset = working.loc[
                working[
                    "product"
                ].eq(product)
            ].sort_values(
                "bin_index"
            )

            if subset.empty:
                continue

            axis.plot(
                subset["bin_index"],
                subset[metric],
                color=PRODUCT_COLORS[
                    product
                ],
                marker=PRODUCT_MARKERS[
                    product
                ],
                linestyle=PRODUCT_LINESTYLES[
                    product
                ],
                linewidth=3,
                markersize=6.5,
                label=PRODUCT_LABELS[
                    product
                ],
                zorder=3,
            )

        # --------------------------------------------------------------
        # Labels
        # --------------------------------------------------------------
        axis.set_title(
            panel_title,
            loc="left",
            fontsize=15,
            fontweight="bold",
            pad=8,
        )

        axis.set_ylabel(
            ylabel,
            fontsize=14,
            fontweight="bold",
        )

        axis.set_xlabel(
            "Precipitation intensity (mm h$^{-1}$)",
            fontsize=14,
            fontweight="bold",
        )

        # lower-edge labels (legacy style)

        tick_labels = []

        for value in bins["bin_left"]:

            if value >= 1:
                tick_labels.append(f"{int(value)}")
            else:
                tick_labels.append(f"{value:g}")

        axis.set_xticks(positions)

        axis.set_xticklabels(
            format_bin_tick_labels(bins["bin_left"])
        )

        axis.set_ylim(
            bottom=0
        )

        axis.grid(
            linestyle="--",
            linewidth=0.6,
            alpha=0.35,
        )

        axis.tick_params(
            axis="both",
            which="major",
            direction="in",
            top=True,
            right=True,
            labelsize=12,
            length=5,
        )

    # ------------------------------------------------------------------
    # Legends
    #
    # Rain and snow use the same product order.
    # Keep separate row legends so reference availability is obvious.
    # ------------------------------------------------------------------
    axes[0, 0].legend(
        frameon=False,
        ncol=3,
        loc="upper right",
        fontsize=12,
    )

    axes[1, 0].legend(
        frameon=False,
        ncol=2,
        loc="upper right",
        fontsize=12,
    )

    # ------------------------------------------------------------------
    # Overall title
    # ------------------------------------------------------------------
    figure.suptitle(
        "Precipitation intensity distributions",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )

    figure.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.08,
        top=0.93,
        wspace=0.22,
        hspace=0.30,
    )

    _finish_figure(
        figure,
        output_path,
        dpi,
    )

    return (
        figure,
        axes,
    )


def plot_temperature_binned_means(
    temperature_data: pd.DataFrame,
    *,
    product_order: Sequence[str],
    dpi: int = 150,
    output_path: str | Path | None = None,
):
    """Plot footprint-level mean precipitation as a function of MERRA-2 T2M."""

    _require_columns(
        temperature_data,
        ["temperature_midpoint", *[f"{product}_mean" for product in product_order]],
    )
    figure, axis = plt.subplots(figsize=(10, 6.2), dpi=dpi)
    for product in product_order:
        axis.plot(
            temperature_data["temperature_midpoint"],
            temperature_data[f"{product}_mean"],
            color=PRODUCT_COLORS[product],
            marker=PRODUCT_MARKERS[product],
            linestyle=PRODUCT_LINESTYLES[product],
            linewidth=2.5,
            markersize=4.5,
            label=PRODUCT_LABELS[product],
        )
    axis.set_xlabel("MERRA-2 2-m air temperature [K]", fontweight="bold", fontsize=14)
    axis.set_ylabel("Mean precipitation [mm h$^{-1}$]", fontweight="bold", fontsize=14)
    axis.tick_params(
        labelsize=14,
        length=5,
        width=1.0,
    )
    axis.set_ylim(bottom=0)
    axis.grid(linestyle="--", linewidth=0.5, alpha=0.35)
    axis.legend(frameon=False, ncol=3, fontsize=13)
    figure.tight_layout()
    _finish_figure(figure, output_path, dpi)
    return figure, axis

# =============================================================================
# METRIC AS A FUNCTION OF TEMPERATURE
def plot_twet_binned_v8_v7_metric_improvements(
    metric_data: pd.DataFrame,
    *,
    reference_order: Sequence[str] = (
        "MRMS",
        "StageIV",
    ),
    dpi: int = 150,
    output_path: str | Path | None = None,
):
    """
    Plot V8-versus-V7 metric improvements by wet-bulb temperature.

    Positive values mean that V8 performs better than V7.
    Negative values mean that V7 performs better than V8.
    """

    # ------------------------------------------------------------------
    # These names must match the columns created by
    # calculate_twet_binned_v8_v7_metrics().
    # ------------------------------------------------------------------
    panel_configuration = (
        (
            "relative_bias_error_improvement_pp",
            "Relative-bias error improvement\n(percentage points)",
        ),
        (
            "frequency_bias_error_improvement",
            "Frequency-bias error improvement",
        ),
        (
            "CSI_improvement",
            "CSI improvement",
        ),
    )

    surface_styles = {
        "snow_free_land": {
            "color": "#d95f02",
            "label": "Snow-free land",
        },
        "snow_covered_land": {
            "color": "#1b9e77",
            "label": "Snow-covered land",
        },
    }

    # ------------------------------------------------------------------
    # Check that the required columns exist
    # ------------------------------------------------------------------
    required_columns = [
        "reference_label",
        "surface_regime",
        "twet_midpoint",
        *[
            column
            for column, _ in panel_configuration
        ],
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in metric_data.columns
    ]

    if missing_columns:
        raise KeyError(
            "Missing required Twet metric columns: "
            + ", ".join(missing_columns)
        )

    # ------------------------------------------------------------------
    # Create the 3-row x 2-column figure
    # ------------------------------------------------------------------
    figure, axes = plt.subplots(
        nrows=len(panel_configuration),
        ncols=len(reference_order),
        figsize=(
            5.7 * len(reference_order),
            3.2 * len(panel_configuration),
        ),
        sharex=True,
        squeeze=False,
        dpi=dpi,
    )

    # ------------------------------------------------------------------
    # Plot one column for each reference
    # ------------------------------------------------------------------
    for column_index, reference in enumerate(
        reference_order
    ):
        reference_data = metric_data.loc[
            metric_data["reference_label"].eq(reference)
        ].copy()

        # --------------------------------------------------------------
        # Plot one row for each metric
        # --------------------------------------------------------------
        for row_index, (
            metric_column,
            y_axis_label,
        ) in enumerate(panel_configuration):

            axis = axes[
                row_index,
                column_index,
            ]

            # ----------------------------------------------------------
            # Plot snow/ice-free and snow/ice-covered curves
            # ----------------------------------------------------------
            for surface_regime, style in surface_styles.items():

                subset = (
                    reference_data.loc[
                        reference_data[
                            "surface_regime"
                        ].eq(surface_regime)
                    ]
                    .sort_values(
                        "twet_midpoint"
                    )
                    .copy()
                )

                axis.plot(
                    subset["twet_midpoint"],
                    subset[metric_column],
                    color=style["color"],
                    label=style["label"],
                    marker="o",
                    markersize=3.5,
                    linewidth=1.5,
                )

            # ----------------------------------------------------------
            # Zero separates V8 improvement from V7 improvement
            # ----------------------------------------------------------
            axis.axhline(
                0.0,
                color="0.25",
                linewidth=0.9,
                linestyle="--",
            )

            axis.grid(
                linestyle="--",
                linewidth=0.5,
                alpha=0.30,
            )

            panel_number = (
                row_index * len(reference_order)
                + column_index
            )

            panel_label = (
                f"({chr(ord('a') + panel_number)})"
            )

            axis.text(
                0.965,
                0.045,
                panel_label,
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=12,
                fontweight="bold",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.75,
                    "pad": 1.5,
                },
                zorder=10,
            )

            axis.set_ylabel(
                y_axis_label,
                fontweight="bold",
            )

            # ----------------------------------------------------------
            # Reference titles
            # ----------------------------------------------------------
            if row_index == 0:
                axis.set_title(
                    f"{reference} reference",
                    fontweight="bold",
                )

            # ----------------------------------------------------------
            # X-axis labels on bottom row only
            # ----------------------------------------------------------
            if row_index == (
                len(panel_configuration) - 1
            ):
                axis.set_xlabel(
                    "MERRA-2 wet-bulb temperature (K)",
                    fontweight="bold",
                )

    # ------------------------------------------------------------------
    # One shared legend
    # ------------------------------------------------------------------
    axes[0, 0].legend(
        frameon=False,
        fontsize=9,
    )

    figure.suptitle(
        "GPROF V8 performance improvement relative to V7 "
        "by wet-bulb temperature",
        fontweight="bold",
    )

    figure.tight_layout()

    _finish_figure(
        figure,
        output_path,
        dpi,
    )

    return figure, axes


def plot_twet_binned_raw_metrics(
    metric_data: pd.DataFrame,
    *,
    reference_order: Sequence[str] = (
        "MRMS",
        "StageIV",
    ),
    dpi: int = 150,
    output_path: str | Path | None = None,
):
    """
    Plot raw GPROF V7, GPROF V8, and ERA5 performance metrics
    by wet-bulb temperature and snow-cover state.

    Layout
    ------
    Rows:
        1. Common-valid footprint count
        2. Relative bias
        3. Frequency bias
        4. Critical success index
        5. Correlation coefficient

    Columns:
        1. Snow-covered land, MRMS reference
        2. Snow-covered land, Stage IV reference
        3. Snow-free land, MRMS reference
        4. Snow-free land, Stage IV reference
    """

    # ------------------------------------------------------------------
    # Metric-row configuration
    # ------------------------------------------------------------------
    metric_configuration = (
            (
                "correlation",
                "CC",
                None,
            ),
            (
                "relative_bias_percent",
                "Relative bias (%)",
                0.0,
            ),
            (
                "frequency_bias",
                "Frequency bias",
                1.0,
            ),
            (
                "CSI",
                "Critical success index",
                None,
            ),
        )

    # ------------------------------------------------------------------
    # Product appearance
    # ------------------------------------------------------------------
    product_configuration = {
        "GPROF_V7": {
            "color": PRODUCT_COLORS["GPROF_V7"],
            "label": PRODUCT_LABELS["GPROF_V7"],
            "linewidth": 2.7,
            "linestyle": "--",
            "zorder": 3,
        },
        "GPROF_V8": {
            "color": PRODUCT_COLORS["GPROF_V8"],
            "label": PRODUCT_LABELS["GPROF_V8"],
            "linewidth": 2.9,
            "linestyle": "-",
            "zorder": 5,
        },
        "ERA5": {
            "color": PRODUCT_COLORS["ERA5"],
            "label": PRODUCT_LABELS["ERA5"],
            "linewidth": 2.4,
            "linestyle": "-",
            "zorder": 2,
        },
    }

    # V8 is plotted last so it remains visible where curves overlap.
    product_plot_order = (
        "ERA5",
        "GPROF_V7",
        "GPROF_V8",
    )

    # ------------------------------------------------------------------
    # First two columns are snow-covered; last two are snow-free.
    # ------------------------------------------------------------------
    column_configuration = (
        (
            "snow_covered_land",
            reference_order[0],
        ),
        (
            "snow_covered_land",
            reference_order[1],
        ),
        (
            "snow_free_land",
            reference_order[0],
        ),
        (
            "snow_free_land",
            reference_order[1],
        ),
    )

    # ------------------------------------------------------------------
    # Validate required columns
    # ------------------------------------------------------------------
    required_columns = [
        "reference_label",
        "surface_regime",
        "twet_midpoint",
        "N",
    ]

    for product in product_configuration:
        for metric, _, _ in metric_configuration:
            required_columns.append(
                f"{product}_{metric}"
            )

    missing_columns = [
        column
        for column in required_columns
        if column not in metric_data.columns
    ]

    if missing_columns:
        raise KeyError(
            "Missing raw Twet metric columns: "
            + ", ".join(missing_columns)
        )

    # ------------------------------------------------------------------
    # Count row + four metric rows
    # ------------------------------------------------------------------
    number_of_rows = (
        1 + len(metric_configuration)
    )

    number_of_columns = len(
        column_configuration
    )

    figure, axes = plt.subplots(
        nrows=number_of_rows,
        ncols=number_of_columns,
        figsize=(20.5, 14.2),
        sharex=True,
        sharey="row",
        squeeze=False,
        dpi=dpi,
    )

    # ------------------------------------------------------------------
    # Column loop
    # ------------------------------------------------------------------
    for column_index, (
        surface_regime,
        reference,
    ) in enumerate(column_configuration):

        panel_data = (
            metric_data.loc[
                metric_data[
                    "reference_label"
                ].eq(reference)
                & metric_data[
                    "surface_regime"
                ].eq(surface_regime)
            ]
            .sort_values("twet_midpoint")
            .copy()
        )

        # ==============================================================
        # ROW 1: BIN COUNT
        # ==============================================================
        count_axis = axes[
            0,
            column_index,
        ]

        positive_count = (
            panel_data["N"] > 0
        )

        count_axis.plot(
            panel_data.loc[
                positive_count,
                "twet_midpoint",
            ],
            panel_data.loc[
                positive_count,
                "N",
            ],
            color="0.25",
            linestyle="-",
            linewidth=1.8,
            marker="o",
            markersize=3.5,
            zorder=3,
        )

        count_axis.set_yscale(
            "log"
        )

        count_axis.grid(
            linestyle="--",
            linewidth=0.5,
            alpha=0.28,
            zorder=0,
        )

        if column_index == 0:
            count_axis.set_ylabel(
                "Common-valid\nfootprint count",
                fontsize=12,
                fontweight="bold",
            )

        reference_label = (
            "Stage IV"
            if reference == "StageIV"
            else reference
        )

        count_axis.set_title(
            f"{reference_label} reference",
            fontsize=15,
            fontweight="bold",
            pad=9,
        )

        count_axis.tick_params(
            axis="both",
            which="major",
            labelsize=11.5,
            direction="out",
        )

        # ==============================================================
        # ROWS 2–5: RAW PERFORMANCE METRICS
        # ==============================================================
        for metric_index, (
            metric,
            y_axis_label,
            ideal_value,
        ) in enumerate(metric_configuration):

            row_index = (
                metric_index + 1
            )

            axis = axes[
                row_index,
                column_index,
            ]

            for product in product_plot_order:
                product_style = (
                    product_configuration[
                        product
                    ]
                )

                metric_column = (
                    f"{product}_{metric}"
                )

                axis.plot(
                    panel_data["twet_midpoint"],
                    panel_data[metric_column],
                    color=product_style["color"],
                    linestyle=product_style["linestyle"],
                    linewidth=product_style["linewidth"],
                    label=product_style["label"],
                    solid_capstyle="round",
                    dash_capstyle="butt",
                    zorder=product_style["zorder"],
                )

            # ----------------------------------------------------------
            # Ideal/reference values
            # ----------------------------------------------------------
            if ideal_value is not None:
                axis.axhline(
                    ideal_value,
                    color="0.25",
                    linestyle=":",
                    linewidth=1.1,
                    zorder=1,
                )

            axis.grid(
                linestyle="--",
                linewidth=0.5,
                alpha=0.28,
                zorder=0,
            )

            if column_index == 0:
                axis.set_ylabel(
                    y_axis_label,
                    fontsize=12,
                    fontweight="bold",
                )

            axis.tick_params(
                axis="both",
                which="major",
                labelsize=11.5,
                length=4,
                width=1.0,
                direction="out",
            )

            if row_index == (
                number_of_rows - 1
            ):
                axis.set_xlabel(
                    "MERRA-2 wet-bulb temperature (K)",
                    fontsize=11.5,
                    fontweight="bold",
                )

        # ==============================================================
        # PANEL LABELS
        # ==============================================================
        for row_index in range(
            number_of_rows
        ):
            panel_number = (
                row_index
                * number_of_columns
                + column_index
            )

            panel_label = (
                f"({chr(ord('a') + panel_number)})"
            )

            axes[
                row_index,
                column_index,
            ].text(
                # 0.965,
                # 0.045,
                0.09,
                0.9,
                panel_label,
                transform=axes[
                    row_index,
                    column_index,
                ].transAxes,
                ha="right",
                va="bottom",
                fontsize=12.5,
                fontweight="bold",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.72,
                    "pad": 1.2,
                },
                zorder=10,
            )

    # ------------------------------------------------------------------
    # Surface-group headings
    #
    # x=1 on the first axis is approximately centered across columns 1–2.
    # The same applies to columns 3–4.
    # ------------------------------------------------------------------
    axes[0, 0].text(
        1.0,
        1.34,
        "Snow-covered land",
        transform=axes[0, 0].transAxes,
        ha="center",
        va="bottom",
        fontsize=15,
        fontweight="bold",
    )

    axes[0, 2].text(
        1.0,
        1.34,
        "Snow-free land",
        transform=axes[0, 2].transAxes,
        ha="center",
        va="bottom",
        fontsize=15,
        fontweight="bold",
    )

    # ------------------------------------------------------------------
    # Product legend
    # ------------------------------------------------------------------
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=(
                product_configuration[
                    product
                ]["color"]
            ),
            linestyle=(
                product_configuration[
                    product
                ]["linestyle"]
            ),
            linewidth=(
                product_configuration[
                    product
                ]["linewidth"]
            ),
            label=(
                product_configuration[
                    product
                ]["label"]
            ),
        )
        for product in (
            "GPROF_V7",
            "GPROF_V8",
            "ERA5",
        )
    ]

    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(
            0.5,
            0.012,
        ),
        ncol=3,
        frameon=False,
        fontsize=12.5,
        handlelength=3.5,
        columnspacing=2.2,
    )

    figure.suptitle(
        "GPROF V7, GPROF V8, and ERA5 performance "
        "by wet-bulb temperature and surface state",
        fontsize=16,
        fontweight="bold",
        y=0.992,
    )

    figure.tight_layout(
        rect=(
            0.0,
            0.055,
            1.0,
            0.945,
        ),
        w_pad=0.6,
        h_pad=0.8,
    )

    figure.subplots_adjust(
        wspace=0.10,
    )

    _finish_figure(
        figure,
        output_path,
        dpi,
    )

    return figure, axes
# =============================================================================
# REFERENCE AGREEMENT AND ROBUSTNESS
# =============================================================================
# =============================================================================
# MRMS–STAGE IV REFERENCE DENSITY PLOT
# =============================================================================

def format_sample_count(
    value: int | float,
) -> str:
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

def _finite_pair(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract paired finite values from two DataFrame columns.
    """

    missing = [
        column
        for column in (
            x_column,
            y_column,
        )
        if column not in data.columns
    ]

    if missing:
        raise KeyError(
            "Missing required plotting columns: "
            + ", ".join(missing)
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

    return (
        x[valid],
        y[valid],
    )


def _calculate_plot_metrics(
    reference: np.ndarray,
    comparison: np.ndarray,
) -> dict[str, float | int]:
    """
    Calculate scatterplot statistics in original precipitation space.

    Bias is defined as comparison - reference.
    """

    if reference.size == 0:
        return {
            "N": 0,
            "bias": np.nan,
            "MAE": np.nan,
            "RMSE": np.nan,
            "CC": np.nan,
        }

    residual = (
        comparison
        - reference
    )

    if (
        reference.size >= 2
        and np.std(reference) > 0
        and np.std(comparison) > 0
    ):
        correlation = float(
            np.corrcoef(
                reference,
                comparison,
            )[0, 1]
        )

    else:
        correlation = np.nan

    return {
        "N": int(reference.size),

        "bias": float(
            np.mean(
                residual
            )
        ),

        "MAE": float(
            np.mean(
                np.abs(
                    residual
                )
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
    Format density-plot metric annotation.
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
    Transform precipitation values for plotting only.

    Supported transforms:
        linear
        log1p
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
        "transform must be either "
        "'linear' or 'log1p'."
    )


def plot_reference_density(
    data: pd.DataFrame,
    *,
    reference_column: str = "MRMS",
    comparison_column: str = "StageIV",
    reference_label: str = "MRMS",
    comparison_label: str = "Stage IV",
    maximum_precipitation: float = 12.0,
    precipitation_ticks: Sequence[float] = (
        0.0,
        0.1,
        0.2,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        12.0,
    ),
    gridsize: int = 100,
    min_count: int = 1,
    colorbar_label: str = "Footprint count",
    title: str | None = None,
    figure_size: tuple[float, float] = (
        7.5,
        6.5,
    ),
    dpi: int = 150,
    output_path: str | Path | None = None,
):
    """
    Plot footprint-level MRMS–Stage IV precipitation density.

    The precipitation coordinates are transformed internally using log1p
    so that light and heavy precipitation can be displayed simultaneously.

    Axis tick labels remain in physical precipitation units (mm h^-1),
    making the figure directly interpretable.

    All statistics are calculated using the ORIGINAL precipitation values,
    not the transformed values.
    """

    # ------------------------------------------------------------------
    # Check required columns
    # ------------------------------------------------------------------

    required_columns = [
        reference_column,
        comparison_column,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise KeyError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    # ------------------------------------------------------------------
    # Extract finite paired observations
    # ------------------------------------------------------------------

    reference = pd.to_numeric(
        data[reference_column],
        errors="coerce",
    ).to_numpy(dtype=float)

    comparison = pd.to_numeric(
        data[comparison_column],
        errors="coerce",
    ).to_numpy(dtype=float)

    valid = (
        np.isfinite(reference)
        & np.isfinite(comparison)
        & (reference >= 0)
        & (comparison >= 0)
    )

    reference = reference[valid]
    comparison = comparison[valid]

    if reference.size == 0:
        raise ValueError(
            "No finite paired MRMS/Stage IV observations remain."
        )

    # ------------------------------------------------------------------
    # Statistics on ORIGINAL precipitation values
    # ------------------------------------------------------------------

    residual = (
        comparison
        - reference
    )

    bias = float(
        np.mean(residual)
    )

    mae = float(
        np.mean(
            np.abs(residual)
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                residual ** 2
            )
        )
    )

    if (
        reference.size >= 2
        and np.std(reference) > 0
        and np.std(comparison) > 0
    ):
        correlation = float(
            np.corrcoef(
                reference,
                comparison,
            )[0, 1]
        )
    else:
        correlation = np.nan

    # ------------------------------------------------------------------
    # Internal log1p transformation
    # ------------------------------------------------------------------

    x = np.log1p(
        reference
    )

    y = np.log1p(
        comparison
    )

    transformed_maximum = float(
        np.log1p(
            maximum_precipitation
        )
    )

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------

    figure, axis = plt.subplots(
        figsize=figure_size,
        dpi=dpi,
    )

    density = axis.hexbin(
        x,
        y,
        gridsize=gridsize,
        mincnt=min_count,
        norm=LogNorm(),
        cmap="viridis",
        extent=(
            0.0,
            transformed_maximum,
            0.0,
            transformed_maximum,
        ),
        linewidths=0,
    )

    # ------------------------------------------------------------------
    # 1:1 line
    # ------------------------------------------------------------------

    axis.plot(
        [
            0.0,
            transformed_maximum,
        ],
        [
            0.0,
            transformed_maximum,
        ],
        linestyle="--",
        linewidth=1.4,
        color="black",
        zorder=5,
    )

    axis.set_xlim(
        0.0,
        transformed_maximum,
    )

    axis.set_ylim(
        0.0,
        transformed_maximum,
    )

    axis.set_aspect(
        "equal",
        adjustable="box",
    )

    # ------------------------------------------------------------------
    # Physical precipitation tick labels
    # ------------------------------------------------------------------

    precipitation_ticks = np.asarray(
        precipitation_ticks,
        dtype=float,
    )

    precipitation_ticks = precipitation_ticks[
        (
            precipitation_ticks >= 0
        )
        & (
            precipitation_ticks
            <= maximum_precipitation
        )
    ]

    transformed_ticks = np.log1p(
        precipitation_ticks
    )

    tick_labels = []

    for value in precipitation_ticks:

        if np.isclose(
            value,
            round(value),
        ):
            tick_labels.append(
                f"{int(round(value))}"
            )

        else:
            tick_labels.append(
                f"{value:g}"
            )

    axis.set_xticks(
        transformed_ticks
    )

    axis.set_xticklabels(
        tick_labels
    )

    axis.set_yticks(
        transformed_ticks
    )

    axis.set_yticklabels(
        tick_labels
    )

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------

    axis.set_xlabel(
        f"{reference_label} precipitation "
        "(mm h$^{{-1}}$)",
        fontsize=13,
        fontweight="bold",
    )

    axis.set_ylabel(
        f"{comparison_label} precipitation "
        "(mm h$^{{-1}}$)",
        fontsize=13,
        fontweight="bold",
    )

    # ------------------------------------------------------------------
    # Statistics box
    # ------------------------------------------------------------------

    if reference.size >= 1_000_000:
        sample_label = (
            f"N = {reference.size / 1_000_000:.2f}M"
        )

    elif reference.size >= 1_000:
        sample_label = (
            f"N = {reference.size / 1_000:.1f}k"
        )

    else:
        sample_label = (
            f"N = {reference.size:,}"
        )

    statistics_text = (
        f"{sample_label}\n"
        f"Bias = {bias:.3f}\n"
        f"MAE = {mae:.3f}\n"
        f"RMSE = {rmse:.3f}\n"
        f"CC = {correlation:.3f}"
    )

    axis.text(
        0.035,
        0.965,
        statistics_text,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={
            "facecolor": "white",
            "edgecolor": "0.5",
            "alpha": 0.90,
            "boxstyle": "round,pad=0.35",
        },
        zorder=10,
    )

    # ------------------------------------------------------------------
    # Axis formatting
    # ------------------------------------------------------------------

    axis.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        labelsize=12,
        length=5,
    )

    axis.grid(
        linestyle="--",
        linewidth=0.5,
        alpha=0.30,
    )

    # ------------------------------------------------------------------
    # Colorbar
    # ------------------------------------------------------------------

    colorbar = figure.colorbar(
        density,
        ax=axis,
        fraction=0.045,
        pad=0.025,
    )

    colorbar.set_label(
        colorbar_label,
        fontsize=12,
        fontweight="bold",
    )

    colorbar.ax.tick_params(
        labelsize=10,
    )

    # ------------------------------------------------------------------
    # Optional title
    # ------------------------------------------------------------------

    if title is not None:

        axis.set_title(
            title,
            fontsize=14,
            fontweight="bold",
            pad=8,
        )

    figure.tight_layout()

    if output_path is not None:

        figure.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
        )

    return figure, axis


def plot_reference_density_by_phase(
    samples: Mapping[str, pd.DataFrame],
    *,
    maximum_precipitation: float = 12.0,
    gridsize: int = 90,
    dpi: int = 150,
    output_path: str | Path | None = None,
):
    """
    Plot MRMS–Stage IV footprint density for rain and snow.

    Both panels use the same density normalization and one dedicated
    colorbar axis.
    """

    # ------------------------------------------------------------------
    # Figure layout:
    # column 1 = rain
    # column 2 = snow
    # column 3 = colorbar
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Compact nested layout
    #
    # Outer grid:
    #     main two-panel region | colorbar
    #
    # Inner grid:
    #     rain panel | snow panel
    #
    # This allows panel spacing and colorbar spacing to be adjusted
    # independently.
    # ------------------------------------------------------------------
    figure = plt.figure(
        figsize=(11.8, 6.2),
        dpi=dpi,
    )

    outer_grid = figure.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=(
            1.0,
            0.026,
        ),
        left=0.075,
        right=0.945,
        bottom=0.16,
        top=0.84,

        # Controls distance between panel (b) and the colorbar.
        wspace=0.04,
    )

    panel_grid = outer_grid[
        0,
        0,
    ].subgridspec(
        nrows=1,
        ncols=2,

        # Controls distance between panels (a) and (b).
        wspace=0.16,
    )

    axes = np.array(
        [
            figure.add_subplot(
                panel_grid[0, 0]
            ),
            figure.add_subplot(
                panel_grid[0, 1]
            ),
        ]
    )

    colorbar_axis = figure.add_subplot(
        outer_grid[0, 1]
    )

    transformed_maximum = float(
        np.log1p(maximum_precipitation)
    )

    # Remove 0.1 because 0, 0.1 and 0.2 are crowded.
    precipitation_ticks = np.array(
        [
            0.0,
            0.2,
            0.5,
            1.0,
            2.0,
            5.0,
            10.0,
            12.0,
        ]
    )

    precipitation_ticks = precipitation_ticks[
        precipitation_ticks <= maximum_precipitation
    ]

    transformed_ticks = np.log1p(
        precipitation_ticks
    )

    tick_labels = [
        f"{value:g}"
        for value in precipitation_ticks
    ]

    density_artists = []

    for panel_index, (
        axis,
        phase,
    ) in enumerate(
        zip(
            axes,
            ("rain", "snow"),
        )
    ):
        data = samples[phase]

        mrms = pd.to_numeric(
            data["MRMS"],
            errors="coerce",
        ).to_numpy(dtype=float)

        stageiv = pd.to_numeric(
            data["StageIV"],
            errors="coerce",
        ).to_numpy(dtype=float)

        valid = (
            np.isfinite(mrms)
            & np.isfinite(stageiv)
            & (mrms >= 0.0)
            & (stageiv >= 0.0)
        )

        mrms = mrms[valid]
        stageiv = stageiv[valid]

        if len(mrms) == 0:
            raise ValueError(
                f"No valid MRMS–Stage IV {phase} pairs."
            )

        x = np.log1p(mrms)
        y = np.log1p(stageiv)

        density_artist = axis.hexbin(
            x,
            y,
            gridsize=gridsize,
            mincnt=1,
            norm=LogNorm(),
            cmap="viridis",
            extent=(
                0.0,
                transformed_maximum,
                0.0,
                transformed_maximum,
            ),
            linewidths=0,
        )

        density_artists.append(
            density_artist
        )

        # --------------------------------------------------------------
        # 1:1 agreement line
        # --------------------------------------------------------------
        axis.plot(
            [0.0, transformed_maximum],
            [0.0, transformed_maximum],
            color="black",
            linestyle="--",
            linewidth=1.2,
        )

        # --------------------------------------------------------------
        # Statistics calculated in original precipitation units
        # --------------------------------------------------------------
        residual = stageiv - mrms

        bias = float(
            np.mean(residual)
        )

        reference_mean = float(
            np.mean(mrms)
        )

        relative_bias_percent = (
            100.0 * bias / reference_mean
            if not np.isclose(reference_mean, 0.0)
            else np.nan
        )

        relative_bias_label = (
            f"{relative_bias_percent:+.2f}%"
            if np.isfinite(relative_bias_percent)
            else "undefined"
        )

        mae = float(
            np.mean(np.abs(residual))
        )

        rmse = float(
            np.sqrt(np.mean(residual**2))
        )

        correlation = (
            float(np.corrcoef(mrms, stageiv)[0, 1])
            if (
                len(mrms) > 1
                and np.std(mrms) > 0
                and np.std(stageiv) > 0
            )
            else np.nan
        )

        statistics_text = (
            f"N = {len(mrms):,}\n"
            f"RB = {relative_bias_label}\n"
            f"MAE = {mae:.3f}\n"
            f"RMSE = {rmse:.3f}\n"
            f"CC = {correlation:.3f}"
        )

        axis.text(
            0.035,
            0.965,
            statistics_text,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
            bbox={
                "facecolor": "white",
                "edgecolor": "0.5",
                "alpha": 0.90,
                "boxstyle": "square,pad=0.35",
            },
            zorder=10,
        )

        # --------------------------------------------------------------
        # Physical precipitation tick labels
        # --------------------------------------------------------------
        axis.set_xticks(
            transformed_ticks
        )

        axis.set_xticklabels(
            tick_labels,
            rotation=40,
            ha="right",
            rotation_mode="anchor",
        )

        axis.set_yticks(
            transformed_ticks
        )

        axis.set_yticklabels(
            tick_labels
        )

        axis.set_xlim(
            0.0,
            transformed_maximum,
        )

        axis.set_ylim(
            0.0,
            transformed_maximum,
        )

        axis.set_aspect(
            "equal",
            adjustable="box",
        )

        axis.set_title(
            f"{phase.capitalize()} footprints",
            fontsize=15,
            fontweight="bold",
            pad=8,
        )

        axis.set_xlabel(
            "MRMS precipitation (mm h$^{-1}$)",
            fontsize=12,
            fontweight="bold",
            labelpad=7,
        )

        axis.grid(
            linestyle="--",
            linewidth=0.5,
            alpha=0.25,
        )

        # --------------------------------------------------------------
        # Panel label in lower-right corner
        # --------------------------------------------------------------
        panel_label = (
            f"({chr(ord('a') + panel_index)})"
        )

        axis.text(
            0.965,
            0.035,
            panel_label,
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75,
                "pad": 1.5,
            },
            zorder=12,
        )

    axes[0].set_ylabel(
        "Stage IV precipitation (mm h$^{-1}$)",
        fontsize=12,
        fontweight="bold",
    )

    # ------------------------------------------------------------------
    # Force the two panels to use exactly the same color normalization
    # ------------------------------------------------------------------
    maximum_bin_count = max(
        float(np.nanmax(artist.get_array()))
        for artist in density_artists
    )

    shared_density_norm = LogNorm(
        vmin=1.0,
        vmax=maximum_bin_count,
    )

    for artist in density_artists:
        artist.set_norm(
            shared_density_norm
        )

    # ------------------------------------------------------------------
    # Dedicated colorbar axis
    # ------------------------------------------------------------------
    colorbar = figure.colorbar(
        density_artists[0],
        cax=colorbar_axis,
        orientation="vertical",
    )

    colorbar.set_label(
        "Footprint count",
        fontsize=11,
        fontweight="bold",
        labelpad=8,
    )

    colorbar.ax.tick_params(
        labelsize=10,
    )

    figure.suptitle(
        "MRMS–Stage IV reference consistency by precipitation phase",
        fontsize=15,
        fontweight="normal",
        y=0.945,
    )

    _finish_figure(
        figure,
        output_path,
        dpi,
    )

    return figure, axes

def plot_reference_robustness_summary(
    robustness_summary: pd.DataFrame,
    *,
    dpi: int = 150,
    output_path: str | Path | None = None,
):
    """Plot positive-is-V8-improvement values for both references by metric."""

    _require_columns(
        robustness_summary,
        ["metric", "MRMS_v8_improvement", "StageIV_v8_improvement", "classification"],
    )
    values = robustness_summary[["MRMS_v8_improvement", "StageIV_v8_improvement"]].to_numpy(float)
    tolerances = (
        robustness_summary["negligible_tolerance"].to_numpy(float)
        if "negligible_tolerance" in robustness_summary
        else np.zeros(len(robustness_summary), dtype=float)
    )
    directions = np.sign(values)
    directions[np.abs(values) <= tolerances[:, None]] = 0
    direction_cmap = ListedColormap(["#d88a2d", "#eeeeee", "#356fa3"])
    direction_norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], direction_cmap.N)
    figure, axis = plt.subplots(figsize=(8.5, 6), dpi=dpi)
    artist = axis.imshow(
        directions,
        aspect="auto",
        cmap=direction_cmap,
        norm=direction_norm,
    )
    axis.set_xticks([0, 1], ["MRMS reference", "Stage IV reference"])
    metric_labels = robustness_summary["metric"].replace(
        {"absolute_relative_bias": "Absolute relative bias"}
    )
    axis.set_yticks(np.arange(len(robustness_summary)), metric_labels)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(column, row, f"{values[row, column]:+.3g}", ha="center", va="center", fontweight="bold")
    axis.set_title(
        "Reference dependence of V8–V7 performance\nColor shows direction; labels retain metric-specific magnitudes",
        fontweight="bold",
    )
    colorbar = figure.colorbar(artist, ax=axis, pad=0.03)
    colorbar.set_ticks([-1, 0, 1])
    colorbar.set_ticklabels(["V7 better", "Negligible", "V8 improved"])
    colorbar.set_label("Direction of performance difference", fontweight="bold")
    figure.tight_layout()
    _finish_figure(figure, output_path, dpi)
    return figure, axis


# =============================================================================
# SPATIAL MAPS
# =============================================================================

def _plot_map_matrix(
    dataset,
    variable_matrix: Sequence[Sequence[str]],
    *,
    row_labels: Sequence[str] | None,
    column_labels: Sequence[str],
    levels: Sequence[float],
    cmap: str,
    colorbar_label: str,
    extent=CONUS_EXTENT,
    extend: str = "max",
    dpi: int = 150,
    output_path: str | Path | None = None,
):
    nrows = len(variable_matrix)
    ncols = len(column_labels)
    boundaries = np.asarray(levels, dtype=float)
    norm = BoundaryNorm(boundaries, plt.get_cmap(cmap).N)
    figure = plt.figure(figsize=(3.5 * ncols, 2.55 * nrows + 1.0), dpi=dpi)
    grid = figure.add_gridspec(nrows, ncols, left=0.055, right=0.985, top=0.95, bottom=0.12, wspace=0.035, hspace=0.12)
    axes = np.empty((nrows, ncols), dtype=object)
    artist = None
    for row in range(nrows):
        for column in range(ncols):
            axis = figure.add_subplot(grid[row, column], projection=ccrs.PlateCarree())
            axes[row, column] = axis
            variable = variable_matrix[row][column]
            if variable not in dataset:
                raise KeyError(f"Missing map variable: {variable}")
            field = np.asarray(dataset[variable].values, dtype=float).squeeze()
            artist = axis.pcolormesh(
                dataset["longitude"], dataset["latitude"], field,
                transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, shading="auto"
            )
            _format_map_axis(axis, extent=extent, left_labels=column == 0, bottom_labels=row == nrows - 1)
            if row == 0:
                axis.set_title(column_labels[column], fontsize=11, fontweight="bold")
            if row_labels is not None and column == 0:
                axis.text(-0.16, 0.5, row_labels[row], transform=axis.transAxes, rotation=90, ha="center", va="center", fontweight="bold")
    colorbar_axis = figure.add_axes([0.25, 0.045, 0.5, 0.022])
    colorbar = figure.colorbar(artist, cax=colorbar_axis, orientation="horizontal", extend=extend)
    colorbar.set_label(colorbar_label, fontweight="bold")
    _finish_figure(figure, output_path, dpi)
    return figure, axes


def plot_spatial_mean_maps(
    spatial_dataset,
    *,
    product_order: Sequence[str],
    levels: Sequence[float],
    extent=CONUS_EXTENT,
    cmap_name: str = "nipy_spectral",
    customize_low_end: bool = True,
    colorbar_ticks: Sequence[float] | None = None,
    colorbar_label: str = "Mean precipitation (mm h$^{-1}$)",
    figure_size: tuple[float, float] = (18, 4.8),
    colorbar_width_fraction: float = 0.54,
    colorbar_height: float = 0.020,
    colorbar_bottom: float = 0.075,
    dpi: int = 150,
    output_path: str | Path | None = None,
):
    """
    Plot mean precipitation maps using the established manuscript style.

    Expected layout:
        GPROF V7 | GPROF V8 | MRMS | Stage IV | ERA5

    All products share one discrete precipitation color scale.
    """

    # ------------------------------------------------------------------
    # Coordinates
    # ------------------------------------------------------------------
    longitude = np.asarray(
        spatial_dataset["longitude"].values,
        dtype=float,
    )

    latitude = np.asarray(
        spatial_dataset["latitude"].values,
        dtype=float,
    )

    if longitude.ndim != 1 or latitude.ndim != 1:
        raise ValueError(
            "Longitude and latitude must be one-dimensional."
        )

    # ------------------------------------------------------------------
    # Discrete precipitation colormap
    # ------------------------------------------------------------------
    boundaries = np.asarray(
        levels,
        dtype=float,
    )

    cmap, norm, boundaries = create_discrete_colormap(
        vmin=float(boundaries[0]),
        vmax=float(boundaries[-1]),
        cmap_name=cmap_name,
        levels=boundaries,
        customize_low_end=customize_low_end,
    )

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    nproducts = len(product_order)

    figure = plt.figure(
        figsize=figure_size,
        dpi=dpi,
    )

    grid = figure.add_gridspec(
        nrows=1,
        ncols=nproducts,
        left=0.045,
        right=0.985,
        top=0.90,
        bottom=0.24,
        wspace=0.035,
    )

    axes = np.empty(
        nproducts,
        dtype=object,
    )

    first_artist = None

    # ------------------------------------------------------------------
    # Product maps
    # ------------------------------------------------------------------
    for panel_index, product in enumerate(
        product_order
    ):

        axis = figure.add_subplot(
            grid[0, panel_index],
            projection=ccrs.PlateCarree(),
        )

        axes[panel_index] = axis

        variable = f"{product}_mean"

        if variable not in spatial_dataset:
            raise KeyError(
                f"Missing map variable: {variable}"
            )

        field = np.asarray(
            spatial_dataset[variable].values,
            dtype=float,
        ).squeeze()

        artist = axis.contourf(
            longitude,
            latitude,
            field,
            levels=boundaries,
            cmap=cmap,
            norm=norm,
            transform=ccrs.PlateCarree(),
            extend="max",
        )

        if first_artist is None:
            first_artist = artist

        # Reuse existing common map formatting.
        _format_map_axis(
            axis,
            extent=extent,
            left_labels=(panel_index == 0),
            bottom_labels=True,
        )

        axis.set_title(
            f"({chr(97 + panel_index)}) "
            f"{PRODUCT_LABELS[product]}",
            fontsize=12,
            fontweight="bold",
            pad=6,
        )

    # ------------------------------------------------------------------
    # Shared centered colorbar
    # ------------------------------------------------------------------
    figure.canvas.draw()

    left_position = axes[0].get_position()
    right_position = axes[-1].get_position()

    full_width = (
        right_position.x1
        - left_position.x0
    )

    colorbar_width = (
        full_width
        * colorbar_width_fraction
    )

    colorbar_left = (
        left_position.x0
        + 0.5
        * (
            full_width
            - colorbar_width
        )
    )

    colorbar_axis = figure.add_axes(
        [
            colorbar_left,
            colorbar_bottom,
            colorbar_width,
            colorbar_height,
        ]
    )

    colorbar = figure.colorbar(
        first_artist,
        cax=colorbar_axis,
        orientation="horizontal",
        extend="max",
    )

    # ------------------------------------------------------------------
    # Colorbar ticks
    # ------------------------------------------------------------------
    if colorbar_ticks is None:

        tick_indices = np.linspace(
            0,
            len(boundaries) - 1,
            min(7, len(boundaries)),
            dtype=int,
        )

        colorbar_ticks = boundaries[
            tick_indices
        ]

    colorbar_ticks = np.asarray(
        colorbar_ticks,
        dtype=float,
    )

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
        labelsize=10,
        length=5,
        width=1.0,
    )

    colorbar.set_label(
        colorbar_label,
        fontsize=12,
        fontweight="bold",
        labelpad=4,
    )

    _finish_figure(
        figure,
        output_path,
        dpi,
    )

    return figure, axes


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
    Create a discrete colormap and BoundaryNorm.

    For nipy_spectral and similar colormaps, the terminal source-cmap
    color is intentionally excluded. This preserves the manuscript-style
    high-end precipitation colors and prevents the largest values from
    being rendered gray.

    When explicit levels are supplied, each interval receives one color.
    """

    # ------------------------------------------------------------------
    # Resolve boundaries
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Sample source colormap
    #
    # IMPORTANT:
    # Request one extra color and discard the terminal color.
    # The terminal end of nipy_spectral is gray, which we do not want
    # for the highest precipitation values.
    # ------------------------------------------------------------------
    source_cmap = plt.get_cmap(
        cmap_name,
        number_of_colors + 1,
    )

    color_array = source_cmap(
        np.linspace(
            0,
            1,
            number_of_colors + 1,
        )
    )[:-1]

    # ------------------------------------------------------------------
    # Manuscript-style low-end modification
    # ------------------------------------------------------------------
    if (
        customize_low_end
        and number_of_colors >= 3
    ):

        color_array[0] = [
            128 / 256,
            100 / 256,
            128 / 256,
            1,
        ]

        # Preserve the original second source color as the third bin.
        color_array[2] = (
            color_array[1].copy()
        )

        # Replace the second bin with manuscript magenta.
        color_array[1] = [
            0.7,
            0.1,
            0.7,
            1,
        ]

    # ------------------------------------------------------------------
    # Discrete cmap and normalization
    # ------------------------------------------------------------------
    discrete_cmap = ListedColormap(
        color_array
    )

    discrete_cmap.set_bad(
        color="white"
    )

    discrete_cmap.set_under(
        discrete_cmap.colors[0]
    )

    # Highest / over-range values use the last REAL color,
    # not the discarded gray nipy_spectral endpoint.
    discrete_cmap.set_over(
        discrete_cmap.colors[-1]
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

def plot_spatial_fraction_maps(
    spatial_dataset,
    *,
    product_order: Sequence[str],
    levels: Sequence[float] = tuple(np.linspace(0, 1, 11)),
    dpi: int = 150,
    output_path: str | Path | None = None,
):
    """Plot wet/valid precipitation fractions (0–1) with a shared scale."""

    variables = [[f"{product}_precipitation_fraction" for product in product_order]]
    return _plot_map_matrix(
        spatial_dataset, variables, row_labels=None,
        column_labels=[PRODUCT_LABELS[product] for product in product_order],
        levels=levels, cmap="cividis", colorbar_label="Wet-observation fraction",
        extend="neither", dpi=dpi, output_path=output_path,
    )


def plot_spatial_difference_maps(
    difference_dataset,
    *,
    variable_order: Sequence[str],
    labels: Mapping[str, str],
    levels: Sequence[float],
    dpi: int = 150,
    output_path: str | Path | None = None,
):
    """Plot paired mean precipitation differences with a common diverging scale."""

    return _plot_map_matrix(
        difference_dataset, [list(variable_order)], row_labels=None,
        column_labels=[labels[variable] for variable in variable_order], levels=levels,
        cmap="RdBu_r", colorbar_label="Paired mean difference (mm h$^{-1}$)",
        extend="both", dpi=dpi, output_path=output_path,
    )


def plot_seasonal_mean_maps(
    seasonal_dataset,
    *,
    product_order: Sequence[str],
    season_order: Sequence[str],
    levels: Sequence[float],
    extent=CONUS_EXTENT,
    cmap_name: str = "nipy_spectral",
    customize_low_end: bool = True,
    colorbar_ticks: Sequence[float] | None = None,
    figure_size: tuple[float, float] = (24.0, 10.5),
    dpi: int = 150,
    output_path: str | Path | None = None,
):
    """
    Plot seasonal mean precipitation.

    Rows
    ----
    DJF, MAM, JJA, SON

    Columns
    -------
    Products in product_order.

    Uses the same discrete manuscript-style precipitation colormap
    as the Section 2.1 mean precipitation maps.
    """

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

    # ------------------------------------------------------------------
    # Manuscript-style discrete precipitation colormap
    # ------------------------------------------------------------------
    cmap, norm, boundaries = (
        create_discrete_colormap(
            vmin=float(boundaries[0]),
            vmax=float(boundaries[-1]),
            cmap_name=cmap_name,
            levels=boundaries,
            customize_low_end=customize_low_end,
        )
    )

    cmap.set_bad("white")
    cmap.set_under(cmap.colors[0])
    cmap.set_over(cmap.colors[-1])

    # ------------------------------------------------------------------
    # Coordinates
    # ------------------------------------------------------------------
    longitude_original = np.asarray(
        seasonal_dataset["longitude"].values,
        dtype=float,
    )

    latitude_original = np.asarray(
        seasonal_dataset["latitude"].values,
        dtype=float,
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
    # Figure layout
    # ------------------------------------------------------------------
    nrows = len(season_order)
    ncols = len(product_order)

    projection = ccrs.PlateCarree()

    figure = plt.figure(
        figsize=figure_size,
        dpi=dpi,
    )

    grid = figure.add_gridspec(
            nrows=nrows,
            ncols=ncols,
            left=0.042,
            right=0.985,
            top=0.970,
            bottom=0.155,
            wspace=0.025,
            hspace=0.075,
        )

    axes = np.empty(
        (nrows, ncols),
        dtype=object,
    )

    first_artist = None
    panel_index = 0

    # ------------------------------------------------------------------
    # Plot seasons x products
    # ------------------------------------------------------------------
    for row_index, season in enumerate(
        season_order
    ):

        if season not in seasonal_dataset["season"].values:
            raise KeyError(
                f"Season {season!r} not found in seasonal_dataset."
            )

        seasonal_slice = seasonal_dataset.sel(
            season=season
        )

        for column_index, product in enumerate(
            product_order
        ):

            variable = (
                f"{product}_mean"
            )

            if variable not in seasonal_slice:
                raise KeyError(
                    f"Missing seasonal variable: {variable}"
                )

            axis = figure.add_subplot(
                grid[row_index, column_index],
                projection=projection,
            )

            axes[
                row_index,
                column_index,
            ] = axis

            field = np.asarray(
                seasonal_slice[variable].values,
                dtype=float,
            ).squeeze()

            expected_shape = (
                len(latitude_original),
                len(longitude_original),
            )

            if field.shape != expected_shape:
                raise ValueError(
                    f"{season} {product}: "
                    f"field shape {field.shape} "
                    f"does not match expected shape "
                    f"{expected_shape}."
                )

            field = field[
                latitude_sort,
                :
            ]

            field = field[
                :,
                longitude_sort
            ]

            artist = axis.contourf(
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

            _format_map_axis(
                axis,
                extent=map_extent,
                left_labels=(column_index == 0),
                bottom_labels=True,
            )

            # Product labels only at top
            if row_index == 0:
                axis.set_title(
                    PRODUCT_LABELS[product],
                    fontsize=12,
                    fontweight="bold",
                    pad=5,
                )

            # Panel labels
            panel_index += 1

            axis.text(
                0.985,
                0.045,
                f"({chr(96 + panel_index)})",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=12.5,
                fontweight="bold",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                    "pad": 1.2,
                },
                zorder=10,
            )

        # Season label at left of row
        axes[row_index, 0].text(
            -0.105,
            0.50,
            season,
            transform=axes[row_index, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            clip_on=False,
        )

    # ------------------------------------------------------------------
    # Shared horizontal colorbar below the seasonal maps
    # ------------------------------------------------------------------
    figure.canvas.draw()

    left_position = axes[-1, 0].get_position()
    right_position = axes[-1, -1].get_position()

    full_map_width = (
        right_position.x1
        - left_position.x0
    )

    colorbar_width = (
        0.58 * full_map_width
    )

    colorbar_left = (
        left_position.x0
        + 0.5
        * (
            full_map_width
            - colorbar_width
        )
    )

    colorbar_axis = figure.add_axes(
        [
            colorbar_left,
            0.055,
            colorbar_width,
            0.020,
        ]
    )

    colorbar = figure.colorbar(
        first_artist,
        cax=colorbar_axis,
        orientation="horizontal",
        extend="max",
    )

    if colorbar_ticks is None:
        colorbar_ticks = np.array(
            [
                boundaries[0],
                0.035,
                0.075,
                0.150,
                0.300,
                boundaries[-1],
            ],
            dtype=float,
        )

    colorbar_ticks = np.asarray(
        colorbar_ticks,
        dtype=float,
    )

    colorbar_ticks = colorbar_ticks[
        (
            colorbar_ticks >= boundaries[0]
        )
        &
        (
            colorbar_ticks <= boundaries[-1]
        )
    ]

    colorbar.set_ticks(
        colorbar_ticks
    )

    colorbar.set_ticklabels(
        [
            f"{tick:.3f}"
            for tick in colorbar_ticks
        ]
    )

    colorbar.ax.tick_params(
        labelsize=10.5,
        length=5,
        width=1.0,
    )

    colorbar.set_label(
        "Seasonal mean precipitation (mm h$^{-1}$)",
        fontsize=11.5,
        fontweight="bold",
        labelpad=5,
    )

    _finish_figure(
        figure,
        output_path,
        dpi,
    )

    return figure, axes


def plot_spatial_mean_and_fraction_maps(
    spatial_dataset,
    *,
    product_order: Sequence[str],
    mean_levels: Sequence[float],
    fraction_levels: Sequence[float] = (
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
        ),
    extent=CONUS_EXTENT,
    cmap_name: str = "nipy_spectral",
    customize_low_end: bool = True,
    mean_colorbar_ticks: Sequence[float] | None = None,
    fraction_colorbar_ticks: Sequence[float] = (
            0,
            5,
            10,
            20,
            30,
            40,
            50,
            60,
            80,
        ),
    figure_size: tuple[float, float] = (28.0, 5.8),
    dpi: int = 150,
    output_path: str | Path | None = None,
):
    """
    Plot mean precipitation and precipitation fraction.

    Rows
    ----
    1. Mean precipitation
    2. Precipitation fraction

    Columns
    -------
    Products in product_order.

    Precipitation fraction is stored internally as 0-1 and is
    converted to percent only for plotting.
    """

    longitude = np.asarray(
        spatial_dataset["longitude"].values,
        dtype=float,
    )

    latitude = np.asarray(
        spatial_dataset["latitude"].values,
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Color scales
    # ------------------------------------------------------------------
    mean_boundaries = np.asarray(
        mean_levels,
        dtype=float,
    )

    mean_cmap, mean_norm, mean_boundaries = (
        create_discrete_colormap(
            vmin=float(mean_boundaries[0]),
            vmax=float(mean_boundaries[-1]),
            cmap_name=cmap_name,
            levels=mean_boundaries,
            customize_low_end=customize_low_end,
        )
    )

    fraction_boundaries = np.asarray(
        fraction_levels,
        dtype=float,
    )

    fraction_cmap, fraction_norm, fraction_boundaries = (
        create_discrete_colormap(
            vmin=float(fraction_boundaries[0]),
            vmax=float(fraction_boundaries[-1]),
            cmap_name=cmap_name,
            levels=fraction_boundaries,
            customize_low_end=customize_low_end,
        )
    )

    for cmap in (
        mean_cmap,
        fraction_cmap,
    ):
        cmap.set_bad("white")
        cmap.set_under(cmap.colors[0])
        cmap.set_over(cmap.colors[-1])

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    nproducts = len(product_order)

    figure = plt.figure(
        figsize=figure_size,
        dpi=dpi,
    )

    grid = figure.add_gridspec(
        nrows=2,
        ncols=nproducts,
        left=0.038,
        right=0.945,
        top=0.965,
        bottom=0.060,
        wspace=0.025,
        hspace=0.025,
    )

    axes = np.empty(
        (2, nproducts),
        dtype=object,
    )

    mean_artist = None
    fraction_artist = None

    panel_index = 0

    # ------------------------------------------------------------------
    # Maps
    # ------------------------------------------------------------------
    for row_index in range(2):

        for column_index, product in enumerate(
            product_order
        ):

            axis = figure.add_subplot(
                grid[
                    row_index,
                    column_index,
                ],
                projection=ccrs.PlateCarree(),
            )

            axes[
                row_index,
                column_index,
            ] = axis

            if row_index == 0:

                variable = (
                    f"{product}_mean"
                )

                field = np.asarray(
                    spatial_dataset[
                        variable
                    ].values,
                    dtype=float,
                ).squeeze()

                panel_cmap = mean_cmap
                panel_norm = mean_norm
                panel_levels = mean_boundaries
                panel_extend = "max"

            else:

                variable = (
                    f"{product}_precipitation_fraction"
                )

                # Stored as fraction 0-1.
                # Plot as percentage.
                field = (
                    100.0
                    * np.asarray(
                        spatial_dataset[
                            variable
                        ].values,
                        dtype=float,
                    ).squeeze()
                )

                panel_cmap = fraction_cmap
                panel_norm = fraction_norm
                panel_levels = fraction_boundaries
                panel_extend = "max"

            artist = axis.contourf(
                longitude,
                latitude,
                field,
                levels=panel_levels,
                cmap=panel_cmap,
                norm=panel_norm,
                transform=ccrs.PlateCarree(),
                extend=panel_extend,
            )

            if (
                row_index == 0
                and mean_artist is None
            ):
                mean_artist = artist

            if (
                row_index == 1
                and fraction_artist is None
            ):
                fraction_artist = artist

            _format_map_axis(
                axis,
                extent=extent,
                left_labels=(
                    column_index == 0
                ),
                bottom_labels=True,
                )

            # Product names only across top row.
            if row_index == 0:
                axis.set_title(
                    PRODUCT_LABELS[
                        product
                    ],
                    fontsize=14,
                    fontweight="bold",
                    pad=6,
                )

            panel_index += 1

            axis.text(
                0.985,
                0.045,
                f"({chr(96 + panel_index)})",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=12.5,
                fontweight="bold",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                    "pad": 1.2,
                },
                zorder=10,
            )

    # Row labels
    axes[0, 0].text(
        -0.105,
        0.5,
        "Mean precipitation",
        transform=axes[0, 0].transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )

    axes[1, 0].text(
        -0.105,
        0.5,
        "Precipitation fraction",
        transform=axes[1, 0].transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )

    # ------------------------------------------------------------------
    # Vertical colorbars on the right of each map row
    # ------------------------------------------------------------------
    figure.canvas.draw()

    row_colorbar_configuration = (
        {
            "row": 0,
            "artist": mean_artist,
            "ticks": mean_colorbar_ticks,
            "label": "Mean precipitation\n[mm h$^{-1}$]",
            "extend": "max",
            "tick_format": "mean",
        },
        {
            "row": 1,
            "artist": fraction_artist,
            "ticks": fraction_colorbar_ticks,
            "label": "Precipitation fraction\n[%]",
            "extend": "max",
            "tick_format": "fraction",
        },
    )

    for config in row_colorbar_configuration:

        row_index = config["row"]

        right_map_position = (
            axes[row_index, -1].get_position()
        )

        left_map_position = (
            axes[row_index, 0].get_position()
        )

        # Same approximate vertical extent as the maps in this row.
        colorbar_bottom = (
            right_map_position.y0
        )

        colorbar_height = (
            right_map_position.height
        )

        colorbar_left = (
            right_map_position.x1
            + 0.010
        )

        colorbar_width = 0.012

        colorbar_axis = figure.add_axes(
            [
                colorbar_left,
                colorbar_bottom,
                colorbar_width,
                colorbar_height,
            ]
        )

        colorbar = figure.colorbar(
            config["artist"],
            cax=colorbar_axis,
            orientation="vertical",
            extend=config["extend"],
        )

        ticks = config["ticks"]

        if ticks is None:
            if row_index == 0:
                ticks = np.array(
                        [
                            0.000,
                            0.035,
                            0.075,
                            0.150,
                            0.300,
                            0.450,
                        ]
                    )
            else:
                ticks = np.array(
                    [
                        0,
                        10,
                        20,
                        40,
                        60,
                        80,
                        100,
                    ]
                )

        ticks = np.asarray(
            ticks,
            dtype=float,
        )

        colorbar.set_ticks(
            ticks
        )

        if config["tick_format"] == "mean":

            colorbar.set_ticklabels(
                [
                    f"{tick:.3f}"
                    for tick in ticks
                ]
            )

        else:

            colorbar.set_ticklabels(
                [
                    f"{tick:g}"
                    for tick in ticks
                ]
            )

        colorbar.ax.tick_params(
            labelsize=11,
            length=4,
            width=1.0,
        )

        colorbar.set_label(
            config["label"],
            fontsize=12,
            fontweight="bold",
            rotation=90,
            labelpad=12,
        )

    _finish_figure(
        figure,
        output_path,
        dpi,
    )

    return figure, axes


def plot_spatial_mean_and_fraction_difference_maps(
    mean_difference_dataset,
    fraction_difference_dataset,
    *,
    variable_order: Sequence[str] = (
        "GPROF_V8_minus_MRMS",
        "GPROF_V8_minus_StageIV",
        "GPROF_V7_minus_MRMS",
        "GPROF_V7_minus_StageIV",
    ),
    labels: Mapping[str, str] | None = None,
    extent=CONUS_EXTENT,

    mean_difference_levels: Sequence[float] = (
        -0.50,
        -0.40,
        -0.30,
        -0.20,
        -0.15,
        -0.10,
        -0.075,
        -0.05,
        -0.025,
        0.00,
        0.025,
        0.05,
        0.075,
        0.10,
        0.15,
        0.20,
        0.30,
        0.40,
        0.50,
    ),

    fraction_difference_levels: Sequence[float] = (
            -30,
            -25,
            -20,
            -15,
            -10,
            -7.5,
            -5,
            -2.5,
            0,
            2.5,
            5,
            7.5,
            10,
            15,
            20,
            25,
            30,
        ),

    mean_difference_ticks: Sequence[float] = (
                -0.30,
                -0.15,
                -0.05,
                0.00,
                0.05,
                0.15,
                0.30,
            ),

    fraction_difference_ticks: Sequence[float] = (
            -30,
            -15,
            -5,
            0,
            5,
            15,
            30,
        ),

    difference_cmap_name: str = "RdBu_r",

    figure_size: tuple[float, float] = (24.0, 5.8),
    dpi: int = 150,
    output_path: str | Path | None = None,
):
    """
    Plot mean-precipitation and precipitation-fraction differences.

    Layout
    ------
    Row 1:
        Mean precipitation differences

    Row 2:
        Precipitation-fraction differences

    Columns:
        GPROF V8 - MRMS
        GPROF V8 - Stage IV
        GPROF V7 - MRMS
        GPROF V7 - Stage IV

    Positive values indicate that the first product in the label
    exceeds the second product.
    """

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------
    if labels is None:
        labels = {
            "GPROF_V8_minus_MRMS": "GPROF V8 − MRMS",
            "GPROF_V8_minus_StageIV": "GPROF V8 − Stage IV",
            "GPROF_V7_minus_MRMS": "GPROF V7 − MRMS",
            "GPROF_V7_minus_StageIV": "GPROF V7 − Stage IV",
        }

    # ------------------------------------------------------------------
    # Coordinates
    # ------------------------------------------------------------------
    longitude_original = np.asarray(
        mean_difference_dataset["longitude"].values,
        dtype=float,
    )

    latitude_original = np.asarray(
        mean_difference_dataset["latitude"].values,
        dtype=float,
    )

    longitude_sort = np.argsort(longitude_original)
    latitude_sort = np.argsort(latitude_original)

    longitude = longitude_original[longitude_sort]
    latitude = latitude_original[latitude_sort]

    map_extent = [
        max(float(extent[0]), float(np.nanmin(longitude))),
        min(float(extent[1]), float(np.nanmax(longitude))),
        max(float(extent[2]), float(np.nanmin(latitude))),
        min(float(extent[3]), float(np.nanmax(latitude))),
    ]

    # ------------------------------------------------------------------
    # Mean-difference colormap
    # ------------------------------------------------------------------
    mean_boundaries = np.asarray(
        mean_difference_levels,
        dtype=float,
    )

    mean_cmap, mean_norm, mean_boundaries = (
        create_discrete_colormap(
            vmin=float(mean_boundaries[0]),
            vmax=float(mean_boundaries[-1]),
            cmap_name=difference_cmap_name,
            levels=mean_boundaries,
        )
    )

    mean_cmap.set_bad("white")
    mean_cmap.set_under(mean_cmap.colors[0])
    mean_cmap.set_over(mean_cmap.colors[-1])

    # ------------------------------------------------------------------
    # Fraction-difference colormap
    # ------------------------------------------------------------------
    fraction_boundaries = np.asarray(
        fraction_difference_levels,
        dtype=float,
    )

    fraction_cmap, fraction_norm, fraction_boundaries = (
        create_discrete_colormap(
            vmin=float(fraction_boundaries[0]),
            vmax=float(fraction_boundaries[-1]),
            cmap_name=difference_cmap_name,
            levels=fraction_boundaries,
        )
    )

    fraction_cmap.set_bad("white")
    fraction_cmap.set_under(fraction_cmap.colors[0])
    fraction_cmap.set_over(fraction_cmap.colors[-1])

    # ------------------------------------------------------------------
    # Figure geometry
    #
    # IMPORTANT:
    # Keep this relatively WIDE and SHORT.
    # Cartopy preserves geographic map aspect ratio.
    # ------------------------------------------------------------------
    projection = ccrs.PlateCarree()

    figure = plt.figure(
        figsize=figure_size,
        dpi=dpi,
    )

    grid = figure.add_gridspec(
            nrows=2,
            ncols=len(variable_order),
            left=0.042,
            right=0.935,
            top=0.965,
            bottom=0.065,
            wspace=0.025,
            hspace=0.025,
        )

    axes = np.empty(
        (2, len(variable_order)),
        dtype=object,
    )

    mean_artist = None
    fraction_artist = None
    panel_index = 0

    # ------------------------------------------------------------------
    # Plot rows
    # ------------------------------------------------------------------
    for row_index in range(2):

        if row_index == 0:
            source_dataset = mean_difference_dataset
            cmap = mean_cmap
            norm = mean_norm
            boundaries = mean_boundaries

        else:
            source_dataset = fraction_difference_dataset
            cmap = fraction_cmap
            norm = fraction_norm
            boundaries = fraction_boundaries

        for column_index, variable in enumerate(variable_order):

            if variable not in source_dataset:
                raise KeyError(
                    f"Missing map variable: {variable}"
                )

            axis = figure.add_subplot(
                grid[row_index, column_index],
                projection=projection,
            )

            axes[row_index, column_index] = axis

            field = np.asarray(
                source_dataset[variable].values,
                dtype=float,
            ).squeeze()

            expected_shape = (
                len(latitude_original),
                len(longitude_original),
            )

            if field.shape != expected_shape:
                raise ValueError(
                    f"{variable}: field shape {field.shape} "
                    f"does not match expected shape "
                    f"{expected_shape}."
                )

            field = field[
                latitude_sort,
                :
            ]

            field = field[
                :,
                longitude_sort
            ]

            artist = axis.contourf(
                longitude,
                latitude,
                field,
                levels=boundaries,
                cmap=cmap,
                norm=norm,
                transform=projection,
                extend="both",
            )

            if row_index == 0 and mean_artist is None:
                mean_artist = artist

            if row_index == 1 and fraction_artist is None:
                fraction_artist = artist

            _format_map_axis(
                axis,
                extent=map_extent,
                left_labels=(column_index == 0),
                bottom_labels=True,
            )

            # Column titles only once
            if row_index == 0:
                axis.set_title(
                    labels[variable],
                    fontsize=12,
                    fontweight="bold",
                    pad=5,
                )

            # Panel labels
            panel_index += 1

            axis.text(
                0.012,
                0.975,
                f"({chr(96 + panel_index)})",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=10.5,
                fontweight="bold",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.65,
                    "pad": 1.0,
                },
                zorder=10,
            )

    # ------------------------------------------------------------------
    # Row labels
    # ------------------------------------------------------------------
    axes[0, 0].text(
        -0.105,
        0.50,
        "Mean difference",
        transform=axes[0, 0].transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        clip_on=False,
    )

    axes[1, 0].text(
        -0.105,
        0.50,
        "Fraction difference",
        transform=axes[1, 0].transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        clip_on=False,
    )

    # Need finalized map positions before manually placing colorbars.
    figure.canvas.draw()

    # ------------------------------------------------------------------
    # Vertical colorbar: mean difference
    # ------------------------------------------------------------------
    right_position = axes[0, -1].get_position()

    mean_colorbar_axis = figure.add_axes(
        [
            right_position.x1 + 0.007,
            right_position.y0,
            0.012,
            right_position.height,
        ]
    )

    mean_colorbar = figure.colorbar(
        mean_artist,
        cax=mean_colorbar_axis,
        orientation="vertical",
        extend="both",
    )

    mean_ticks = np.asarray(
        mean_difference_ticks,
        dtype=float,
    )

    mean_colorbar.set_ticks(
        mean_ticks
    )

    mean_colorbar.set_ticklabels(
        [
            f"{tick:.2f}"
            for tick in mean_ticks
        ]
    )

    mean_colorbar.ax.tick_params(
        labelsize=10.5,
        length=4,
    )

    mean_colorbar.set_label(
        "Mean precipitation\ndifference\n(mm h$^{-1}$)",
        fontsize=11.5,
        fontweight="bold",
        rotation=90,
        labelpad=9,
    )

    # ------------------------------------------------------------------
    # Vertical colorbar: fraction difference
    # ------------------------------------------------------------------
    right_position = axes[1, -1].get_position()

    fraction_colorbar_axis = figure.add_axes(
        [
            right_position.x1 + 0.007,
            right_position.y0,
            0.012,
            right_position.height,
        ]
    )

    fraction_colorbar = figure.colorbar(
        fraction_artist,
        cax=fraction_colorbar_axis,
        orientation="vertical",
        extend="both",
    )

    fraction_ticks = np.asarray(
        fraction_difference_ticks,
        dtype=float,
    )

    fraction_colorbar.set_ticks(
        fraction_ticks
    )

    fraction_colorbar.set_ticklabels(
        [
            f"{tick:g}"
            for tick in fraction_ticks
        ]
    )

    fraction_colorbar.ax.tick_params(
        labelsize=10.5,
        length=4,
    )

    fraction_colorbar.set_label(
        "Precipitation-fraction\ndifference\n(percentage points)",
        fontsize=11.5,
        fontweight="bold",
        rotation=90,
        labelpad=9,
    )

    _finish_figure(
        figure,
        output_path,
        dpi,
    )

    return figure, axes

__all__ = [
    "plot_multireference_performance_diagram",
    "plot_occurrence_and_volume_distributions",
    "plot_quantitative_metric_bars",
    "plot_reference_density",
    "plot_reference_robustness_summary",
    "plot_seasonal_mean_maps",
    "plot_spatial_difference_maps",
    "plot_spatial_fraction_maps",
    "plot_spatial_mean_maps",
    "plot_spatial_mean_and_fraction_maps",
    "plot_spatial_mean_and_fraction_difference_maps",
    "plot_temperature_binned_means",
    "plot_twet_binned_v8_v7_metric_improvements",
    "plot_twet_binned_raw_metrics",
]

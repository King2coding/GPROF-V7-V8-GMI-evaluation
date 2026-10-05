#%%
"""
GPROF GMI V7/V8 evaluation against MRMS over CONUS.

This script is intentionally organized for interactive, section-by-section
execution. RAQI is retained in the matchup archive but is not used anywhere
in the present analysis.
"""
import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gprof_v7_v8_analysis_functions_v1 import (
    add_time_coordinates,
    calculate_dynamic_temperature_binned_means,
    calculate_monthly_mean_timeseries,
    calculate_product_categorical_table,
    calculate_product_metric_table,
    calculate_seasonal_spatial_mean_fields,
    calculate_spatial_categorical_metrics,
    calculate_spatial_continuous_metrics,
    calculate_spatial_mean_fields,
    calculate_spatial_precipitation_fraction,
    calculate_surface_group_means,
    compute_product_pdf_bundle,
    create_spatial_grid,
    discover_matchup_files,
    load_matchup_archive,
    load_matchup_cache,
    save_matchup_cache,
    select_analysis_sample,
    calculate_spatial_mean_differences,
    calculate_spatial_precipitation_fraction_differences,
    add_imerg_land_mask_to_matchups
)

from gprof_v7_v8_plotting_functions_v1 import (
    plot_monthly_mean_timeseries,
    plot_mrms_product_scatter_density,
    plot_pdf_bundle,
    plot_performance_diagram,
    plot_quantitative_metric_bars,
    plot_seasonal_spatial_mean_maps,
    plot_spatial_categorical_metric_maps,
    plot_spatial_continuous_metric_maps,
    plot_spatial_mean_maps,
    plot_spatial_precipitation_fraction_maps,
    plot_surface_group_means,
    plot_temperature_binned_means,
    plot_spatial_means_and_differences,
    plot_spatial_precipitation_fractions_and_differences,
)


#%%
# =============================================================================
# 1. User configuration
# =============================================================================

CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent

MATCHUP_DIR = Path(
    "/scratch/kkumah/"
    "GPROF_V7_V8_MRMS_RAQI_ERA5_MERRA2_AutoSnow_matchups/"
    "matchups/GMI"
)

LAND_SEA_MASK_PATH = Path(
    "/ra1/pubdat/AVHRR_CloudSat_proj/IMERG/"
    "ancillary_imerg_data/"
    "GPM_IMERG_LandSeaMask.2.nc4"
)

OUTPUT_DIR = PROJECT_DIR / "analysis_outputs_v1"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
CACHE_DIR = OUTPUT_DIR / "cache"
NETCDF_DIR = OUTPUT_DIR / "netcdf"

for directory in (OUTPUT_DIR, FIGURE_DIR, TABLE_DIR, CACHE_DIR, NETCDF_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CACHE_FILE = CACHE_DIR / "gprof_v7_v8_mrms_era5_common_sample.parquet"
LOAD_SUMMARY_FILE = TABLE_DIR / "matchup_file_loading_summary.csv"

# Rebuild after new production files have been added to the archive.
REBUILD_CACHE = True
SKIP_FAILED_FILES = True
RETAIN_IDENTIFIERS = True

PRODUCT_COLUMNS = ["GPROF_V7", "GPROF_V8", "MRMS", "ERA5"]
EVALUATED_PRODUCTS = ["GPROF_V7", "GPROF_V8", "ERA5"]
REFERENCE_COLUMN = "MRMS"

WET_THRESHOLD = 0.1
CONUS_EXTENT = (-125.0, -66.0, 24.0, 50.0)
SPATIAL_GRID_RESOLUTION = 0.5 #0.25
MINIMUM_SPATIAL_SAMPLE_COUNT = 1
MINIMUM_SPATIAL_EVENT_COUNT = 1
MINIMUM_REFERENCE_MEAN_FOR_RELATIVE_BIAS = 0.01

# User-defined PDF upper-bin limits. Values above 64 mm h-1 are not included.
PDF_BINS = np.array([0.1, 0.2, 0.5, 1, 2, 4, 8, 16, 32, 64], dtype=float)

# Two-kelvin temperature bins.
TEMPERATURE_BIN_WIDTH = 2.0
FULL_TEMPERATURE_RANGE = (230.0, 315.0)
COLD_TEMPERATURE_RANGE = (230.0, 275.0)
MINIMUM_TEMPERATURE_BIN_COUNT = 1


#%%
# =============================================================================
# 2. Discover matchup files
# =============================================================================

matchup_files = discover_matchup_files(MATCHUP_DIR)

if len(matchup_files) == 0:
    raise FileNotFoundError(f"No matchup files found in {MATCHUP_DIR}")


#%%
# =============================================================================
# 3. Build or load compact matchup dataframe
# =============================================================================

if REBUILD_CACHE or not CACHE_FILE.exists():
    df_matchups, load_summary = load_matchup_archive(
        matchup_files,
        retain_only_base_reference_rows=True,
        retain_identifiers=RETAIN_IDENTIFIERS,
        skip_failed_files=SKIP_FAILED_FILES,
        show_progress=False,
    )
    save_matchup_cache(df_matchups, CACHE_FILE)
    load_summary.to_csv(LOAD_SUMMARY_FILE, index=False)
else:
    df_matchups = load_matchup_cache(CACHE_FILE)

# RAQI is deliberately not required here or in any analysis below.
df_matchups = add_imerg_land_mask_to_matchups(
    df_matchups,
    land_sea_mask_path=LAND_SEA_MASK_PATH,
    mask_variable="landseamask",
    longitude_column="longitude",
    latitude_column="latitude",
    output_column="land_mask",
    land_threshold=25.0,
)

print(
    "\nStatic IMERG land-mask counts "
    "(1=land, 0=water):"
)

print(
    df_matchups["land_mask"]
    .value_counts(dropna=False)
    .sort_index()
)

print(
    "\nStatic IMERG land-mask percentages:"
)

print(
    (
        100.0
        * df_matchups["land_mask"]
        .value_counts(
            normalize=True,
            dropna=False,
        )
        .sort_index()
    ).round(2)
)

# The core comparison sample requires identical finite MRMS, V7, V8, and ERA5.
df_common = select_analysis_sample(
    df_matchups,
    require_era5=True,
)

df_common = df_common.loc[
    df_common["land_mask"] == 1
].copy()

required_common_columns = [
    REFERENCE_COLUMN,
    *EVALUATED_PRODUCTS,
]

assert (
    df_common[
        required_common_columns
    ]
    .notna()
    .all()
    .all()
)

assert (
    df_common["land_mask"] == 1
).all()

print(
    f"Land-only common sample: "
    f"{len(df_common):,} footprints"
)

#%%
# =============================================================================
# 4. Common-sample inventory
# =============================================================================

sample_inventory = pd.DataFrame(
    {
        "sample": [
            "Loaded matchup footprints",
            "Static-mask land footprints",
            "Common land MRMS–V7–V8–ERA5 footprints",
            "Common land sample with T2M",
            "Common land sample with AutoSnow",
        ],
        "count": [
            len(df_matchups),
            int(
                (
                    df_matchups["land_mask"] == 1
                ).sum()
            ),
            len(df_common),
            len(
                select_analysis_sample(
                    df_common,
                    require_era5=True,
                    require_t2m=True,
                )
            ),
            len(
                select_analysis_sample(
                    df_common,
                    require_era5=True,
                    require_autosnow=True,
                )
            ),
        ],
    }
)

sample_inventory.to_csv(TABLE_DIR / "common_sample_inventory.csv", index=False)

gc.collect()

#%%
# =============================================================================
# 5. Summary quantitative metrics
# =============================================================================
assert (
    df_common["land_mask"] == 1
).all()

print(
    f"Calculating quantitative metrics from "
    f"{len(df_common):,} land-only footprints."
)

quantitative_metrics = calculate_product_metric_table(
    df_common,
    reference_column=REFERENCE_COLUMN,
    product_columns=EVALUATED_PRODUCTS,
)

assert quantitative_metrics["N"].nunique() == 1
quantitative_metrics.to_csv(TABLE_DIR / "summary_quantitative_metrics.csv", index=False)

fig, axes = plot_quantitative_metric_bars(
    quantitative_metrics,
    product_order=EVALUATED_PRODUCTS,
    metrics=("correlation", "relative_bias_percent", "RMSE", "MAE"),
    title=None,
    figure_size=(7.2, 8.5),
    output_path=FIGURE_DIR / "summary_quantitative_metrics_barplot.png",
)
plt.show()


#%%
# =============================================================================
# 6. Summary categorical metrics and Roebber diagram
# =============================================================================

categorical_metrics = calculate_product_categorical_table(
    df_common,
    reference_column=REFERENCE_COLUMN,
    product_columns=EVALUATED_PRODUCTS,
    threshold=WET_THRESHOLD,
)

assert categorical_metrics["N"].nunique() == 1
categorical_metrics.to_csv(TABLE_DIR / "summary_categorical_metrics.csv", index=False)

fig, ax = plot_performance_diagram(
    categorical_metrics,
    product_order=EVALUATED_PRODUCTS,
    title="",
    threshold_text=None,
    output_path=FIGURE_DIR / "summary_categorical_roebber_diagram.png",
)
plt.show()


#%%
# =============================================================================
# 7. Fixed CONUS spatial grid
# =============================================================================

conus_grid = create_spatial_grid(
    extent=CONUS_EXTENT,
    resolution=SPATIAL_GRID_RESOLUTION,
)


#%%
# =============================================================================
# 8. Spatial quantitative metrics
# =============================================================================

spatial_quantitative = calculate_spatial_continuous_metrics(
    df_common,
    grid=conus_grid,
    reference_column=REFERENCE_COLUMN,
    product_columns=EVALUATED_PRODUCTS,
    minimum_sample_count=MINIMUM_SPATIAL_SAMPLE_COUNT,
    minimum_reference_mean_for_relative_bias=(
        MINIMUM_REFERENCE_MEAN_FOR_RELATIVE_BIAS
    ),
)

spatial_quantitative.to_netcdf(NETCDF_DIR / "spatial_quantitative_metrics_0p5deg.nc")

fig, axes = plot_spatial_continuous_metric_maps(
    spatial_quantitative,
    product_order=EVALUATED_PRODUCTS,
    extent=CONUS_EXTENT,
    title=None,
    colorbar_width_fraction=0.44,
    colorbar_height=0.014,
    colorbar_gap=0.045,
    output_path=(
        FIGURE_DIR
        / "spatial_quantitative_metrics_0p5deg.png"
    ),
)

plt.show()
gc.collect()

#%%
# =============================================================================
# 9. Spatial categorical metrics
# =============================================================================

spatial_categorical = calculate_spatial_categorical_metrics(
    df_common,
    grid=conus_grid,
    reference_column=REFERENCE_COLUMN,
    product_columns=EVALUATED_PRODUCTS,
    wet_threshold=WET_THRESHOLD,
    minimum_event_count=MINIMUM_SPATIAL_EVENT_COUNT,
)

spatial_categorical.to_netcdf(NETCDF_DIR / "spatial_categorical_metrics_0p5deg.nc")

fig, axes = plot_spatial_categorical_metric_maps(
    spatial_categorical,
    product_order=EVALUATED_PRODUCTS,
    extent=CONUS_EXTENT,
    title=None,
    output_path=(
        FIGURE_DIR
        / "spatial_categorical_metrics_0p5deg.png"
    ),
)

plt.show()
gc.collect()


#%%
# =============================================================================
# 10. Spatial mean precipitation
# =============================================================================

spatial_means = calculate_spatial_mean_fields(
    df_common,
    grid=conus_grid,
    product_columns=PRODUCT_COLUMNS,
    minimum_sample_count=MINIMUM_SPATIAL_SAMPLE_COUNT,
)

spatial_means.to_netcdf(NETCDF_DIR / "spatial_mean_precipitation_0p5deg.nc")

fig, axes = plot_spatial_mean_maps(
    spatial_means,
    product_order=PRODUCT_COLUMNS,
    extent=CONUS_EXTENT,
    title=None,
    output_path=FIGURE_DIR / "spatial_mean_precipitation_0p5deg.png",
)
plt.show()
gc.collect()

#%%
# =============================================================================
# 10a. Spatial mean precipitation differences
# =============================================================================

spatial_mean_differences = (
    calculate_spatial_mean_differences(
        spatial_means
    )
)

spatial_mean_differences.to_netcdf(
    NETCDF_DIR
    / "spatial_mean_precipitation_differences_0p5deg.nc"
)

fig, axes = plot_spatial_means_and_differences(
    spatial_mean_dataset=spatial_means,
    spatial_difference_dataset=(
        spatial_mean_differences
    ),
    extent=CONUS_EXTENT,
    mean_vmin=0.0,
    mean_vmax=0.325,
    difference_vmin=-0.25,
    difference_vmax=0.25,
    difference_step=0.015,
    mean_colorbar_ticks=(
        0.000,
        0.088,
        0.175,
        0.263,
        0.325
        # 0.350,
        # 0.438,
        # 0.525,
    ),
    difference_colorbar_ticks=(
        -0.25,
        -0.20,
        -0.15,
        -0.10,
        -0.05,
        0.00,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
    ),
    title=None,
    output_path=(
        FIGURE_DIR
        / "spatial_mean_and_difference_maps_0p5deg.png"
    ),
)

plt.show()
gc.collect()
#%%
# =============================================================================
# 11. Seasonal spatial mean precipitation
# =============================================================================

seasonal_spatial_means = calculate_seasonal_spatial_mean_fields(
    df_common,
    grid=conus_grid,
    product_columns=PRODUCT_COLUMNS,
    date_column="orbit_date",
    minimum_sample_count=MINIMUM_SPATIAL_SAMPLE_COUNT,
)

seasonal_spatial_means.to_netcdf(
    NETCDF_DIR / "seasonal_spatial_mean_precipitation_0p5deg.nc"
)

fig, axes = plot_seasonal_spatial_mean_maps(
    seasonal_spatial_means,
    product_order=PRODUCT_COLUMNS,
    season_order=(
        "DJF",
        "MAM",
        "JJA",
        "SON",
    ),
    extent=CONUS_EXTENT,
    title=None,
    colorbar_width_fraction=0.50,
    colorbar_height=0.018,
    colorbar_bottom=0.045,
    output_path=(
        FIGURE_DIR
        / "seasonal_spatial_mean_precipitation_0p5deg.png"
    ),
)

plt.show()
gc.collect()


#%%
# =============================================================================
# 12. Monthly mean precipitation time series
# =============================================================================

monthly_means = calculate_monthly_mean_timeseries(
    df_common,
    product_columns=PRODUCT_COLUMNS,
    date_column="orbit_date",
)

monthly_means.to_csv(TABLE_DIR / "monthly_mean_precipitation_timeseries.csv", index=False)

fig, ax = plot_monthly_mean_timeseries(
    monthly_means,
    product_columns=PRODUCT_COLUMNS,
    output_path=FIGURE_DIR / "monthly_mean_precipitation_timeseries.png",
)
plt.show()
gc.collect()


#%%
# =============================================================================
# 13. Spatial precipitation fraction
# =============================================================================

spatial_fraction = calculate_spatial_precipitation_fraction(
    df_common,
    grid=conus_grid,
    product_columns=PRODUCT_COLUMNS,
    wet_threshold=WET_THRESHOLD,
    minimum_sample_count=MINIMUM_SPATIAL_SAMPLE_COUNT,
)

spatial_fraction.to_netcdf(NETCDF_DIR / "spatial_precipitation_fraction_0p5deg.nc")

fig, axes = plot_spatial_precipitation_fraction_maps(
    spatial_fraction,
    product_order=PRODUCT_COLUMNS,
    extent=CONUS_EXTENT,
    output_path=FIGURE_DIR / "spatial_precipitation_fraction_0p5deg.png",
)
plt.show()
gc.collect()

#%%
# =============================================================================
# Spatial precipitation-fraction differences
# =============================================================================

spatial_fraction_differences = (
    calculate_spatial_precipitation_fraction_differences(
        spatial_fraction
    )
)

spatial_fraction_differences.to_netcdf(
    NETCDF_DIR
    / "spatial_precipitation_fraction_differences_0p5deg.nc"
)

fig, axes = plot_spatial_precipitation_fractions_and_differences(
    spatial_fraction_dataset=spatial_fraction,
    spatial_difference_dataset=spatial_fraction_differences,
    extent=CONUS_EXTENT,
    fraction_levels=(
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
    ),
    fraction_colorbar_ticks=(
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
    ),
    difference_vmin=-20.0,
    difference_vmax=20.0,
    difference_step=2.5,
    difference_colorbar_ticks=(
        # -30,
        # -25,
        -20,
        -15,
        -10,
        -5,
        0,
        5,
        10,
        15,
        20,
        # 25,
        # 30,
    ),
    title=None,
    output_path=(
        FIGURE_DIR
        / "spatial_precipitation_fraction_and_differences_0p5deg.png"
    ),
)

plt.show()
gc.collect()
#%%
# =============================================================================
# 14. Mean precipitation by AutoSnow surface type
# =============================================================================

df_surface = select_analysis_sample(
    df_common,
    require_era5=True,
    require_autosnow=True,
)

assert (
    df_surface["land_mask"] == 1
).all()

surface_means = calculate_surface_group_means(
    df_surface,
    product_columns=PRODUCT_COLUMNS,
    include_all_surfaces=True,
    minimum_sample_count=1,
)

# Retain only the overall land sample and the two land
# AutoSnow surface categories.
surface_means = surface_means.loc[
    surface_means["surface_code"].isin(
        [
            -1,
            1,
            2,
        ]
    )
].copy()

# Preserve the desired plotting order.
surface_code_order = {
    -1: 0,
    1: 1,
    2: 2,
}

surface_means["_plot_order"] = (
    surface_means["surface_code"]
    .map(surface_code_order)
)

surface_means = (
    surface_means
    .sort_values("_plot_order")
    .drop(columns="_plot_order")
    .reset_index(drop=True)
)

print(
    surface_means[
        [
            "surface_code",
            "surface_group",
            "sample_count",
        ]
    ]
)

surface_means.to_csv(
    TABLE_DIR
    / "mean_precipitation_by_surface_type.csv",
    index=False,
)

fig, ax = plot_surface_group_means(
    surface_means,
    product_columns=PRODUCT_COLUMNS,
    title="",
    annotate_sample_counts=False,
    output_path=(
        FIGURE_DIR
        / "mean_precipitation_by_surface_type.png"
    ),
)

plt.show()
gc.collect()
#%%
# =============================================================================
# 15. PDF by precipitation volume
# =============================================================================

pdf_bundle = compute_product_pdf_bundle(
    df_common,
    product_columns=PRODUCT_COLUMNS,
    bins=PDF_BINS,
)

pdf_table = pd.concat(
    [table.assign(product=product) for product, table in pdf_bundle.items()],
    ignore_index=True,
)
pdf_table.to_csv(TABLE_DIR / "precipitation_pdf_by_volume_upto_32mmhr.csv", index=False)

fig, ax = plot_pdf_bundle(
    pdf_bundle,
    product_order=PRODUCT_COLUMNS,
    pdf_kind="pdfv",
    title="",
    output_path=FIGURE_DIR / "precipitation_pdf_by_volume_upto_32mmhr.png",
)
plt.show()
gc.collect()


#%%
# =============================================================================
# 16. Temperature-bin analysis: full temperature range
# =============================================================================

# T2M is required only for temperature-dependent analysis.
df_temperature = select_analysis_sample(
    df_common,
    require_era5=True,
    require_t2m=True,
)

temperature_full = calculate_dynamic_temperature_binned_means(
    df_temperature,
    product_columns=PRODUCT_COLUMNS,
    temperature_column="T2M",
    bin_width=TEMPERATURE_BIN_WIDTH,
    temperature_range=FULL_TEMPERATURE_RANGE,
    minimum_sample_count=MINIMUM_TEMPERATURE_BIN_COUNT,
)

temperature_full.to_csv(TABLE_DIR / "temperature_binned_means_full_range.csv", index=False)

fig, ax = plot_temperature_binned_means(
    temperature_full,
    product_columns=PRODUCT_COLUMNS,
    title=None,
    x_limits=FULL_TEMPERATURE_RANGE,
    x_tick_interval=10.0,
    output_path=FIGURE_DIR / "temperature_binned_means_full_range.png",
)
plt.show()
gc.collect()


#%%
# =============================================================================
# 17. Temperature-bin analysis: cold temperature range
# =============================================================================

temperature_cold = calculate_dynamic_temperature_binned_means(
    df_temperature,
    product_columns=PRODUCT_COLUMNS,
    temperature_column="T2M",
    bin_width=TEMPERATURE_BIN_WIDTH,
    temperature_range=COLD_TEMPERATURE_RANGE,
    minimum_sample_count=MINIMUM_TEMPERATURE_BIN_COUNT,
)

temperature_cold.to_csv(TABLE_DIR / "temperature_binned_means_cold_range.csv", index=False)

fig, ax = plot_temperature_binned_means(
    temperature_cold,
    product_columns=PRODUCT_COLUMNS,
    title=None,
    x_limits=COLD_TEMPERATURE_RANGE,
    x_tick_interval=5.0,
    output_path=FIGURE_DIR / "temperature_binned_means_cold_range.png",
)
plt.show()
gc.collect()


#%%
# =============================================================================
# 18. MRMS-versus-product scatter-density plots
# =============================================================================

fig, axes = plot_mrms_product_scatter_density(
    df_common,
    products=EVALUATED_PRODUCTS,
    reference_column=REFERENCE_COLUMN,
    axis_limit=(0.0, 12.0),
    tick_values=(
        0.0,
        1.0,
        2.0,
        4.0,
        8.0,
        12.0,
    ),
    title=None,
    output_path=(
        FIGURE_DIR
        / "mrms_product_scatter_density.png"
    ),
)

plt.show()
gc.collect()

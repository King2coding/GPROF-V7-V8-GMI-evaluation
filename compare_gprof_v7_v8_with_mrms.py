#%%
"""
Comparative analysis of GPROF V7 and V8 using MRMS as reference.

Products:
    - GPROF V7
    - GPROF V8
    - MRMS Pass2
    - ERA5 precipitation

Conditioning variables:
    - MRMS RAQI
    - MERRA-2 T2M
    - AutoSnow surface class

Main outputs:
    - Matchup sample inventory
    - Mean precipitation by surface class
    - Mean precipitation by temperature
    - Continuous metrics
    - Categorical metrics
    - Precipitation PDFs
    - MRMS-versus-product density plots
    - Roebber performance diagrams
    - Taylor diagrams
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gprof_v7_v8_analysis_functions import (
    add_autosnow_labels,
    add_temperature_bins,
    calculate_categorical_metrics_by_group,
    calculate_dynamic_temperature_binned_means,
    calculate_metrics_by_group,
    calculate_product_categorical_table,
    calculate_product_metric_table,
    calculate_surface_group_means,
    dataframe_memory_mib,
    discover_matchup_files,
    load_matchup_archive,
    load_matchup_cache,
    save_matchup_cache,
    select_analysis_sample,
    summarize_sample_counts, 
    compute_pdf_elements,
    compute_product_pdf_bundle,
    create_spatial_grid,
    calculate_spatial_mean_fields,
    calculate_spatial_continuous_metrics,
    calculate_spatial_categorical_metrics,
    
)

from gprof_v7_v8_plotting_functions import (
    plot_mrms_product_scatter_density,
    plot_pdf_bundle,
    plot_performance_diagram,
    plot_quantitative_metric_bars,
    plot_surface_group_means,
    plot_temperature_binned_means,
    plot_spatial_mean_maps,
    plot_spatial_continuous_metric_maps,
    plot_spatial_categorical_metric_maps,
    plot_spatial_sample_count,
)


#%%
# =============================================================================
# 1. USER CONFIGURATION
# =============================================================================

CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent

OUTPUT_DIR = PROJECT_DIR / "analysis_outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
CACHE_DIR = OUTPUT_DIR / "cache"

MATCHUP_DIR = Path(
    "/scratch/kkumah/"
    "GPROF_V7_V8_MRMS_RAQI_ERA5_MERRA2_AutoSnow_matchups/"
    "matchups/GMI"
)

for directory in (
    OUTPUT_DIR,
    FIGURE_DIR,
    TABLE_DIR,
    CACHE_DIR,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# -----------------------------------------------------------------------------
# Data loading controls
# -----------------------------------------------------------------------------

CACHE_FILE = (
    CACHE_DIR
    / "gprof_v7_v8_mrms_matchups.parquet"
)

LOAD_SUMMARY_FILE = (
    TABLE_DIR
    / "matchup_file_loading_summary.csv"
)

# Set True whenever you want to rebuild the cache from the current NetCDF files.
# Because production is still running, use True when you want a new snapshot.
REBUILD_CACHE = False

# Skip individual corrupt/incomplete files instead of stopping the analysis.
SKIP_FAILED_FILES = False

# Keep source_file, orbit, orbit_date, scan, and pixel identifiers.
# These are useful for tracing questionable samples.
RETAIN_IDENTIFIERS = True


# -----------------------------------------------------------------------------
# Scientific settings
# -----------------------------------------------------------------------------

PRODUCT_COLUMNS = [
    "GPROF_V7",
    "GPROF_V8",
    "MRMS",
    "ERA5",
]

EVALUATED_PRODUCTS = [
    "GPROF_V7",
    "GPROF_V8",
    "ERA5",
]

REFERENCE_COLUMN = "MRMS"

RAQI_THRESHOLDS = [
    None,
    0.5,
    0.8,
]

PRIMARY_RAQI_THRESHOLD = 0.8

# -----------------------------------------------------------------------------
# Spatial-analysis settings
# -----------------------------------------------------------------------------

CONUS_EXTENT = (
    -125.0,
    -66.0,
    24.0,
    50.0,
)

SPATIAL_GRID_RESOLUTION = 0.5

# Exploratory spatial maps: retain every populated grid cell.
MINIMUM_SPATIAL_SAMPLE_COUNT = 1

# Retain categorical metrics whenever their denominator contains
# at least one event.
MINIMUM_SPATIAL_EVENT_COUNT = 1

# Relative bias is unstable when the local mean MRMS precipitation
# is almost zero. This threshold does not remove cells from CC or RMSE.
MINIMUM_REFERENCE_MEAN_FOR_RELATIVE_BIAS = 0.01

LATITUDE_COLUMN = "latitude"
LONGITUDE_COLUMN = "longitude"

SPATIAL_OUTPUT_DIR = (
    OUTPUT_DIR
    / "spatial_analysis"
)

SPATIAL_FIGURE_DIR = (
    SPATIAL_OUTPUT_DIR
    / "figures"
)

SPATIAL_TABLE_DIR = (
    SPATIAL_OUTPUT_DIR
    / "netcdf"
)

for directory in (
    SPATIAL_OUTPUT_DIR,
    SPATIAL_FIGURE_DIR,
    SPATIAL_TABLE_DIR,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

# -----------------------------------------------------------------------------
# Precipitation threshold
# -----------------------------------------------------------------------------

# GPROF is reported as mm hr-1, while MRMS and ERA5 are one-hour
# accumulations in mm. Their numerical values are compared on a common
# hourly precipitation scale.
WET_THRESHOLD = 0.1


# -----------------------------------------------------------------------------
# Precipitation-intensity distribution bins
# -----------------------------------------------------------------------------

# Resolve intensities through 32 mm h-1 and place all larger values
# in a final >=32 mm h-1 class.
PRECIPITATION_INTENSITY_BIN_EDGES = np.array(
    [
        0.0,
        0.5,
        1.0,
        2.0,
        4.0,
        8.0,
        16.0,
        32.0,
        np.inf,
    ],
    dtype=float,
)


# -----------------------------------------------------------------------------
# Broad temperature classes
# -----------------------------------------------------------------------------

TEMPERATURE_BIN_EDGES = np.array(
    [
        -np.inf,
        253.15,
        263.15,
        273.15,
        283.15,
        293.15,
        np.inf,
    ],
    dtype=float,
)

TEMPERATURE_BIN_LABELS = [
    "T2M < 253 K",
    "253–263 K",
    "263–273 K",
    "273–283 K",
    "283–293 K",
    "T2M ≥ 293 K",
]


# -----------------------------------------------------------------------------
# Fine temperature curves
# -----------------------------------------------------------------------------

TEMPERATURE_CURVE_RANGE = (
    235.0,
    315.0,
)

TEMPERATURE_CURVE_BIN_WIDTH = 2.0

# Minimum number of footprints required to display a group.
MINIMUM_GROUP_SAMPLE_COUNT = 100

# Surface classes used in the principal land comparison.
PRIMARY_SURFACE_CLASSES = [
    1,  # snow-free land
    2,  # snow-covered land
]

PRELIMINARY_LABEL = (
    "PRELIMINARY — production archive may be incomplete"
)


# -----------------------------------------------------------------------------
# Plot settings
# -----------------------------------------------------------------------------

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
    }
)


#%%
# =============================================================================
# 2. DISCOVER MATCHUP FILES
# =============================================================================

all_matchup_files = discover_matchup_files(
    MATCHUP_DIR,
    recursive=False,
)

print(
    f"Completed NetCDF matchup files found: "
    f"{len(all_matchup_files):,}"
)

if not all_matchup_files:
    raise RuntimeError(
        f"No completed NetCDF files were found in:\n"
        f"{MATCHUP_DIR}"
    )

print("First matchup file:")
print(all_matchup_files[0])

print("\nLast matchup file:")
print(all_matchup_files[-1])


#%%
# =============================================================================
# 3. BUILD OR LOAD COMPACT MATCHUP DATAFRAME
# =============================================================================

if REBUILD_CACHE or not CACHE_FILE.exists():

    print(
        "\nBuilding compact matchup DataFrame "
        "from NetCDF files..."
    )

    df_matchups, load_summary = load_matchup_archive(
        all_matchup_files,
        retain_only_base_reference_rows=True,
        retain_identifiers=RETAIN_IDENTIFIERS,
        skip_failed_files=SKIP_FAILED_FILES,
        show_progress=True,
    )

    save_matchup_cache(
        df_matchups,
        CACHE_FILE,
    )

    load_summary.to_csv(
        LOAD_SUMMARY_FILE,
        index=False,
    )

    print(
        f"\nSaved matchup cache:\n{CACHE_FILE}"
    )

    print(
        f"\nSaved file-loading summary:\n"
        f"{LOAD_SUMMARY_FILE}"
    )

else:

    print(
        "\nLoading existing compact matchup cache..."
    )

    df_matchups = load_matchup_cache(
        CACHE_FILE
    )

    if LOAD_SUMMARY_FILE.exists():
        load_summary = pd.read_csv(
            LOAD_SUMMARY_FILE
        )
    else:
        load_summary = pd.DataFrame()


print(
    f"\nCompact DataFrame rows: "
    f"{len(df_matchups):,}"
)

print(
    f"Compact DataFrame memory: "
    f"{dataframe_memory_mib(df_matchups):,.1f} MiB"
)

print("\nDataFrame columns:")
print(df_matchups.columns.tolist())

print("\nDataFrame data types:")
print(df_matchups.dtypes)


#%%
# =============================================================================
# 4. BASIC DATA INSPECTION
# =============================================================================

print(
    df_matchups[
        [
            "GPROF_V7",
            "GPROF_V8",
            "MRMS",
            "RAQI",
            "ERA5",
            "T2M",
            "AutoSnow",
        ]
    ].describe()
)

print("\nMissing-value counts:")

print(
    df_matchups[
        [
            "GPROF_V7",
            "GPROF_V8",
            "MRMS",
            "RAQI",
            "ERA5",
            "T2M",
            "AutoSnow",
        ]
    ]
    .isna()
    .sum()
)


#%%
# =============================================================================
# 5. FILE-LOADING QUALITY CONTROL
# =============================================================================

if not load_summary.empty:

    print("\nFile-loading status:")

    print(
        load_summary[
            "status"
        ].value_counts(
            dropna=False
        )
    )

    failed_files = load_summary.loc[
        load_summary["status"] != "PASS"
    ].copy()

    if not failed_files.empty:
        print("\nFiles that failed to load:")

        print(
            failed_files[
                [
                    "source_file",
                    "message",
                ]
            ]
        )

    print(
        "\nTotal rows retained according to "
        "file summaries:"
    )

    print(
        load_summary[
            "rows_retained"
        ].sum()
    )


#%%
# =============================================================================
# 6. SAMPLE INVENTORY BY RAQI THRESHOLD
# =============================================================================

sample_inventory = summarize_sample_counts(
    df_matchups,
    raqi_thresholds=RAQI_THRESHOLDS,
)

print(sample_inventory)

sample_inventory.to_csv(
    TABLE_DIR
    / "sample_inventory_by_raqi_threshold.csv",
    index=False,
)


#%%
# =============================================================================
# 7. CREATE MAIN ANALYSIS SAMPLES
# =============================================================================

# Base common sample:
# finite V7, V8, MRMS, and RAQI with mrms_valid_flag == 1.
df_base = select_analysis_sample(
    df_matchups
)

# Common sample with RAQI >= 0.5.
df_raqi05 = select_analysis_sample(
    df_matchups,
    raqi_threshold=0.5,
)

# Main high-quality MRMS sample.
df_raqi08 = select_analysis_sample(
    df_matchups,
    raqi_threshold=PRIMARY_RAQI_THRESHOLD,
)

# Common sample for comparisons that include ERA5.
df_raqi08_era5 = select_analysis_sample(
    df_matchups,
    raqi_threshold=PRIMARY_RAQI_THRESHOLD,
    require_era5=True,
)

# Common sample for temperature analysis.
df_raqi08_t2m = select_analysis_sample(
    df_matchups,
    raqi_threshold=PRIMARY_RAQI_THRESHOLD,
    require_era5=True,
    require_t2m=True,
)

# Common sample for surface-type analysis.
df_raqi08_surface = select_analysis_sample(
    df_matchups,
    raqi_threshold=PRIMARY_RAQI_THRESHOLD,
    require_era5=True,
    require_autosnow=True,
)


print(
    f"Base common sample: "
    f"{len(df_base):,}"
)

print(
    f"RAQI >= 0.5 sample: "
    f"{len(df_raqi05):,}"
)

print(
    f"RAQI >= 0.8 sample: "
    f"{len(df_raqi08):,}"
)

print(
    f"RAQI >= 0.8 + ERA5 sample: "
    f"{len(df_raqi08_era5):,}"
)

print(
    f"RAQI >= 0.8 + ERA5 + T2M sample: "
    f"{len(df_raqi08_t2m):,}"
)

print(
    f"RAQI >= 0.8 + ERA5 + AutoSnow sample: "
    f"{len(df_raqi08_surface):,}"
)


#%%
# =============================================================================
# 8. SURFACE-CLASS SAMPLE INVENTORY
# =============================================================================

surface_inventory = (
    df_raqi08_surface[
        "AutoSnow"
    ]
    .value_counts(
        dropna=False
    )
    .sort_index()
    .rename_axis(
        "AutoSnow"
    )
    .reset_index(
        name="sample_count"
    )
)

surface_names = {
    0: "Clear water",
    1: "Snow-free land",
    2: "Snow-covered land",
    3: "Ice-covered water",
}

surface_inventory[
    "surface_name"
] = surface_inventory[
    "AutoSnow"
].map(surface_names)

print(surface_inventory)

surface_inventory.to_csv(
    TABLE_DIR
    / "surface_class_sample_inventory_raqi_ge_0p8.csv",
    index=False,
)


#%%
# =============================================================================
# 9. MEAN PRECIPITATION BY SURFACE CLASS
# =============================================================================

surface_means_all = calculate_surface_group_means(
    df_raqi08_surface,
    product_columns=PRODUCT_COLUMNS,
    include_all_surfaces=True,
    minimum_sample_count=MINIMUM_GROUP_SAMPLE_COUNT,
)

print(surface_means_all)

surface_means_all.to_csv(
    TABLE_DIR
    / "mean_precipitation_by_surface_raqi_ge_0p8.csv",
    index=False,
)


#%%
# Principal surface comparison:
# all surfaces, snow-free land, and snow-covered land.

principal_surface_names = [
    "all surfaces",
    "AutoSnow 0: clear water",
    "AutoSnow 1: snow-free land",
    "AutoSnow 2: snow-covered land",
]

surface_means_land = surface_means_all.loc[
    surface_means_all[
        "surface_group"
    ].isin(principal_surface_names)
].copy()

fig, ax = plot_surface_group_means(
    surface_mean_data=surface_means_land,
    product_columns=PRODUCT_COLUMNS,
    title=(
        "Mean precipitation over land surface classes\n"
        f"MRMS RAQI ≥ {PRIMARY_RAQI_THRESHOLD:g}"
    ),
    ylabel="Mean hourly precipitation (mm h$^{-1}$)",
    annotate_sample_counts=True,
    output_path=(
        FIGURE_DIR
        / "mean_precipitation_land_surface_classes_raqi_ge_0p8.png"
    ),
    preliminary_label=PRELIMINARY_LABEL,
)

plt.show()


#%%
# =============================================================================
# 11A. FULL TEMPERATURE CURVE
# =============================================================================
temperature_means_fine = (
    calculate_dynamic_temperature_binned_means(
        df_raqi08_t2m,
        product_columns=PRODUCT_COLUMNS,
        temperature_column="T2M",
        bin_width=TEMPERATURE_CURVE_BIN_WIDTH,
        temperature_range=TEMPERATURE_CURVE_RANGE,
        minimum_sample_count=(
            MINIMUM_GROUP_SAMPLE_COUNT
        ),
    )
)

print(temperature_means_fine.head())

temperature_means_fine.to_csv(
    TABLE_DIR
    / "mean_precipitation_temperature_curve_raqi_ge_0p8.csv",
    index=False,
)


#%%
# =============================================================================
# 11A. FULL TEMPERATURE CURVE
# =============================================================================

fig, ax = plot_temperature_binned_means(
    temperature_mean_data=temperature_means_fine,
    product_columns=PRODUCT_COLUMNS,
    temperature_column="temperature_midpoint",
    title=(
        "Mean precipitation as a function of MERRA-2 T2M\n"
        f"{TEMPERATURE_CURVE_BIN_WIDTH:g}-K bins; "
        f"MRMS RAQI ≥ {PRIMARY_RAQI_THRESHOLD:g}"
    ),
    xlabel="MERRA-2 2-m air temperature (K)",
    ylabel="Mean precipitation (mm h$^{-1}$)",
    x_limits=(240, 302),
    x_tick_interval=4,
    output_path=(
        FIGURE_DIR
        / "mean_precipitation_temperature_curve_raqi_ge_0p8.png"
    ),
    preliminary_label=PRELIMINARY_LABEL,
)

plt.show()

#%%
# =============================================================================
# COLD-REGIME TEMPERATURE CURVE: T2M <= 270 K
# =============================================================================
temperature_means_cold = (
    temperature_means_fine.loc[
        temperature_means_fine[
            "temperature_midpoint"
        ] <= 270.0
    ]
    .copy()
)

fig, ax = plot_temperature_binned_means(
    temperature_mean_data=temperature_means_cold,
    product_columns=PRODUCT_COLUMNS,
    temperature_column="temperature_midpoint",
    title=(
        "Mean precipitation in the cold-temperature regime\n"
        f"MERRA-2 T2M ≤ 270 K; "
        f"{TEMPERATURE_CURVE_BIN_WIDTH:g}-K bins; "
        f"MRMS RAQI ≥ {PRIMARY_RAQI_THRESHOLD:g}"
    ),
    x_limits=(240, 270),
    x_tick_interval=4,
    output_path=(
        FIGURE_DIR
        / "mean_precipitation_temperature_curve_t2m_le_270K.png"
    ),
    preliminary_label=PRELIMINARY_LABEL,
)

plt.show()


#%%
# =============================================================================
# 12. CONTINUOUS METRICS
# =============================================================================

# Because ERA5 is included in this table, use the common sample that requires
# finite V7, V8, MRMS, RAQI, and ERA5.
continuous_metrics = (
    calculate_product_metric_table(
        df_raqi08_era5,
        reference_column=REFERENCE_COLUMN,
        product_columns=EVALUATED_PRODUCTS,
    )
)

print(continuous_metrics)

continuous_metrics.to_csv(
    TABLE_DIR
    / "continuous_metrics_mrms_reference_raqi_ge_0p8.csv",
    index=False,
)


#%%
# Rounded copy for convenient viewing only.

continuous_metrics_display = (
    continuous_metrics.copy()
)

numeric_metric_columns = [
    column
    for column in continuous_metrics_display.columns
    if column
    not in [
        "reference",
        "product",
    ]
]

continuous_metrics_display[
    numeric_metric_columns
] = continuous_metrics_display[
    numeric_metric_columns
].round(4)

print(continuous_metrics_display)


#%%
# =============================================================================
# 13. CONTINUOUS METRICS BY RAQI THRESHOLD
# =============================================================================

continuous_metrics_by_raqi = []

for raqi_threshold in RAQI_THRESHOLDS:

    df_threshold = select_analysis_sample(
        df_matchups,
        raqi_threshold=raqi_threshold,
        require_era5=True,
    )

    threshold_metrics = (
        calculate_product_metric_table(
            df_threshold,
            reference_column=REFERENCE_COLUMN,
            product_columns=EVALUATED_PRODUCTS,
        )
    )

    threshold_metrics[
        "raqi_threshold"
    ] = (
        "all"
        if raqi_threshold is None
        else raqi_threshold
    )

    continuous_metrics_by_raqi.append(
        threshold_metrics
    )

continuous_metrics_by_raqi = pd.concat(
    continuous_metrics_by_raqi,
    ignore_index=True,
)

continuous_metrics_by_raqi.to_csv(
    TABLE_DIR
    / "continuous_metrics_by_raqi_threshold.csv",
    index=False,
)

print(continuous_metrics_by_raqi)


#%%
# =============================================================================
# 14. CATEGORICAL METRICS
# =============================================================================

categorical_metrics = (
    calculate_product_categorical_table(
        df_raqi08_era5,
        reference_column=REFERENCE_COLUMN,
        product_columns=EVALUATED_PRODUCTS,
        threshold=WET_THRESHOLD,
    )
)

print(categorical_metrics)

categorical_metrics.to_csv(
    TABLE_DIR
    / "categorical_metrics_mrms_reference_raqi_ge_0p8.csv",
    index=False,
)


#%%
# =============================================================================
# 15. ROEBBER PERFORMANCE DIAGRAM
# =============================================================================

fig, ax = plot_performance_diagram(
    categorical_metrics=categorical_metrics,
    product_order=EVALUATED_PRODUCTS,
    title=(
        "Precipitation detection performance relative to MRMS"
    ),
    threshold_text=(
        f"Wet threshold = {WET_THRESHOLD:g}; "
        f"MRMS RAQI ≥ {PRIMARY_RAQI_THRESHOLD:g}"
    ),
    annotate_points=True,
    output_path=(
        FIGURE_DIR
        / "roebber_performance_diagram_raqi_ge_0p8.png"
    ),
    preliminary_label=PRELIMINARY_LABEL,
)

plt.show()


#%%
# =============================================================================
# 16. QUANTITATIVE PERFORMANCE METRIC BARS
# =============================================================================

fig, axes = plot_quantitative_metric_bars(
    metric_data=continuous_metrics,
    product_order=EVALUATED_PRODUCTS,
    metrics=[
        "correlation",
        "relative_bias_percent",
        "RMSE",
    ],
    title=(
        "Quantitative performance relative to MRMS\n"
        f"MRMS RAQI ≥ {PRIMARY_RAQI_THRESHOLD:g}"
    ),
    output_path=(
        FIGURE_DIR
        / "quantitative_metric_bars_raqi_ge_0p8.png"
    ),
    preliminary_label=PRELIMINARY_LABEL,
)

plt.show()


#%%
# =============================================================================
# 17. PRECIPITATION INTENSITY PDF
# =============================================================================

PDF_BIN_VALUES = np.array(
    [
        0.5,
        1.0,
        2.0,
        4.0,
        8.0,
        16.0,
        32.0,
    ],
    dtype=float,
)

# All products use the same common-valid DataFrame.
# No independent wet-sample filtering is applied.
pdf_bundle = compute_product_pdf_bundle(
    df_raqi08_era5,
    product_columns=PRODUCT_COLUMNS,
    bins=PDF_BIN_VALUES,
)

for product, product_pdf in (
    pdf_bundle.items()
):
    product_pdf.to_csv(
        TABLE_DIR
        / (
            f"precipitation_pdf_"
            f"{product.lower()}_"
            f"raqi_ge_0p8.csv"
        ),
        index=False,
    )

#%%
fig, ax = plot_pdf_bundle(
    pdf_bundle,
    product_order=PRODUCT_COLUMNS,
    pdf_kind="pdfc",
    title=(
        "Precipitation intensity PDF by occurrence\n"
        f"MRMS RAQI ≥ {PRIMARY_RAQI_THRESHOLD:g}"
    ),
    ylabel="PDF by occurrence (%)",
    output_path=(
        FIGURE_DIR
        / "precipitation_intensity_pdf_occurrence_raqi_ge_0p8.png"
    ),
    preliminary_label=PRELIMINARY_LABEL,
)

plt.show()

#%%
fig, ax = plot_pdf_bundle(
    pdf_bundle,
    product_order=PRODUCT_COLUMNS,
    pdf_kind="pdfv",
    title=(
        "Precipitation intensity PDF by volume\n"
        f"MRMS RAQI ≥ {PRIMARY_RAQI_THRESHOLD:g}"
    ),
    ylabel="PDF by volume (%)",
    output_path=(
        FIGURE_DIR
        / "precipitation_intensity_pdf_volume_raqi_ge_0p8.png"
    ),
    preliminary_label=PRELIMINARY_LABEL,
)

plt.show()

#%%
# =============================================================================
# 19. MRMS-VERSUS-PRODUCT DENSITY PLOTS
# =============================================================================

# These scatter-density panels use identical common samples for V7, V8, and
# ERA5 because df_raqi08_era5 requires all four precipitation fields.
fig, axes = plot_mrms_product_scatter_density(
    data=df_raqi08_era5,
    products=EVALUATED_PRODUCTS,
    reference_column="MRMS",
    axis_limit=(0.0, 10.0),
    tick_values=[
        0,        
        2,        
        4,
        6,
        8,
        10,
    ],
    density_vmin=1,
    density_vmax=8000,
    add_regression_line=False,
    title=(
        "Product precipitation versus MRMS\n"
        f"MRMS RAQI ≥ {PRIMARY_RAQI_THRESHOLD:g}"
    ),
    output_path=(
        FIGURE_DIR
        / "mrms_product_scatter_density_raqi_ge_0p8.png"
    ),
    preliminary_label=PRELIMINARY_LABEL,
)

plt.show()


#%%
# =============================================================================
# 20. ADD SURFACE AND TEMPERATURE LABELS FOR STRATIFIED ANALYSIS
# =============================================================================

df_stratified = add_autosnow_labels(
    df_raqi08_surface,
    source_column="AutoSnow",
    output_column="surface_class",
)

df_stratified = add_temperature_bins(
    df_stratified,
    temperature_column="T2M",
    bin_edges=TEMPERATURE_BIN_EDGES,
    bin_labels=TEMPERATURE_BIN_LABELS,
    output_column="temperature_bin",
)

print(
    df_stratified[
        [
            "surface_class",
            "temperature_bin",
        ]
    ].head()
)


#%%
# =============================================================================
# 21. CONTINUOUS METRICS BY SURFACE CLASS
# =============================================================================

continuous_by_surface = calculate_metrics_by_group(
    df_stratified,
    group_column="surface_class",
    reference_column=REFERENCE_COLUMN,
    product_columns=EVALUATED_PRODUCTS,
    minimum_sample_count=MINIMUM_GROUP_SAMPLE_COUNT,
)

continuous_by_surface.to_csv(
    TABLE_DIR
    / "continuous_metrics_by_surface_raqi_ge_0p8.csv",
    index=False,
)

print(continuous_by_surface)


#%%
# =============================================================================
# 22. CATEGORICAL METRICS BY SURFACE CLASS
# =============================================================================

categorical_by_surface = (
    calculate_categorical_metrics_by_group(
        df_stratified,
        group_column="surface_class",
        threshold=WET_THRESHOLD,
        reference_column=REFERENCE_COLUMN,
        product_columns=EVALUATED_PRODUCTS,
        minimum_sample_count=(
            MINIMUM_GROUP_SAMPLE_COUNT
        ),
    )
)

categorical_by_surface.to_csv(
    TABLE_DIR
    / "categorical_metrics_by_surface_raqi_ge_0p8.csv",
    index=False,
)

print(categorical_by_surface)


#%%
# =============================================================================
# 23. CONTINUOUS METRICS BY TEMPERATURE CLASS
# =============================================================================

continuous_by_temperature = (
    calculate_metrics_by_group(
        df_stratified.dropna(
            subset=["temperature_bin"]
        ),
        group_column="temperature_bin",
        reference_column=REFERENCE_COLUMN,
        product_columns=EVALUATED_PRODUCTS,
        minimum_sample_count=(
            MINIMUM_GROUP_SAMPLE_COUNT
        ),
    )
)

continuous_by_temperature.to_csv(
    TABLE_DIR
    / "continuous_metrics_by_temperature_raqi_ge_0p8.csv",
    index=False,
)

print(continuous_by_temperature)


#%%
# =============================================================================
# 24. CATEGORICAL METRICS BY TEMPERATURE CLASS
# =============================================================================

categorical_by_temperature = (
    calculate_categorical_metrics_by_group(
        df_stratified.dropna(
            subset=["temperature_bin"]
        ),
        group_column="temperature_bin",
        threshold=WET_THRESHOLD,
        reference_column=REFERENCE_COLUMN,
        product_columns=EVALUATED_PRODUCTS,
        minimum_sample_count=(
            MINIMUM_GROUP_SAMPLE_COUNT
        ),
    )
)

categorical_by_temperature.to_csv(
    TABLE_DIR
    / "categorical_metrics_by_temperature_raqi_ge_0p8.csv",
    index=False,
)

print(categorical_by_temperature)


#%%
# =============================================================================
# 25. SPATIAL ANALYSIS: CREATE FIXED CONUS GRID
# =============================================================================

spatial_grid = create_spatial_grid(
    extent=CONUS_EXTENT,
    resolution=SPATIAL_GRID_RESOLUTION,
)

print(
    f"Spatial grid: "
    f"{spatial_grid['nlat']} latitude cells × "
    f"{spatial_grid['nlon']} longitude cells"
)

#%%
# =============================================================================
# 26. SPATIAL MEAN PRECIPITATION
# =============================================================================
# Common finite sample without applying an RAQI threshold.
# RAQI must still be finite because it is part of the base matchup mask,
# but no minimum RAQI value is imposed.
df_spatial_all_raqi = select_analysis_sample(
    df_matchups,
    raqi_threshold=None,
    require_era5=True,
)

print(
    f"Spatial sample without RAQI threshold: "
    f"{len(df_spatial_all_raqi):,}"
)

spatial_mean_dataset = (
    calculate_spatial_mean_fields(
        df_spatial_all_raqi,
        grid=spatial_grid,
        product_columns=PRODUCT_COLUMNS,
        latitude_column=LATITUDE_COLUMN,
        longitude_column=LONGITUDE_COLUMN,
        minimum_sample_count=(
            MINIMUM_SPATIAL_SAMPLE_COUNT
        ),
    )
)

spatial_mean_dataset.to_netcdf(
    SPATIAL_TABLE_DIR
    / "spatial_mean_precipitation_raqi_ge.nc"
)

print(spatial_mean_dataset)


print("\nSpatial continuous-metric coverage:")

print("\nSpatial continuous-metric coverage:")

for product in EVALUATED_PRODUCTS:
    for metric in [
        "CC",
        "relative_bias",
        "RMSE",
    ]:
        variable_name = (
            f"{product}_{metric}"
        )

        if (
            variable_name
            not in spatial_mean_dataset
        ):
            print(
                f"Missing variable: "
                f"{variable_name}"
            )
            continue

        values = (
            spatial_mean_dataset[
                variable_name
            ].values
        )

        valid_count = int(
            np.isfinite(values).sum()
        )

        total_count = values.size

        print(
            f"{product:10s} "
            f"{metric:15s}: "
            f"{valid_count:,}/"
            f"{total_count:,} cells valid "
            f"({100.0 * valid_count / total_count:.1f}%)"
        )

#%%
# =============================================================================
# 27. SPATIAL QUANTITATIVE METRICS
# =============================================================================

spatial_continuous_dataset = (
    calculate_spatial_continuous_metrics(
        df_spatial_all_raqi,
        grid=spatial_grid,
        reference_column="MRMS",
        product_columns=EVALUATED_PRODUCTS,
        latitude_column=LATITUDE_COLUMN,
        longitude_column=LONGITUDE_COLUMN,
        minimum_sample_count=(
            MINIMUM_SPATIAL_SAMPLE_COUNT
        ),
        minimum_reference_mean_for_relative_bias=(
            MINIMUM_REFERENCE_MEAN_FOR_RELATIVE_BIAS
        ),
    )
)

spatial_continuous_dataset.to_netcdf(
    SPATIAL_TABLE_DIR
    / "spatial_continuous_metrics_all_finite_raqi.nc"
)

print(spatial_continuous_dataset)

print("\nSpatial categorical-metric coverage:")


for product in EVALUATED_PRODUCTS:
    for metric in [
        "POD",
        "FAR",
        "CSI",
    ]:
        variable_name = (
            f"{product}_{metric}"
        )

        if (
            variable_name
            not in spatial_continuous_dataset
        ):
            print(
                f"Missing variable: "
                f"{variable_name}"
            )
            continue

        values = (
            spatial_continuous_dataset[
                variable_name
            ].values
        )

        valid_count = int(
            np.isfinite(values).sum()
        )

        total_count = values.size

        print(
            f"{product:10s} "
            f"{metric:5s}: "
            f"{valid_count:,}/"
            f"{total_count:,} cells valid "
            f"({100.0 * valid_count / total_count:.1f}%)"
        )

#%%
# =============================================================================
# 28. SPATIAL CATEGORICAL METRICS
# =============================================================================

spatial_categorical_dataset = (
    calculate_spatial_categorical_metrics(
        df_spatial_all_raqi,
        grid=spatial_grid,
        reference_column="MRMS",
        product_columns=EVALUATED_PRODUCTS,
        wet_threshold=WET_THRESHOLD,
        latitude_column=LATITUDE_COLUMN,
        longitude_column=LONGITUDE_COLUMN,
        minimum_event_count=(
            MINIMUM_SPATIAL_EVENT_COUNT
        ),
    )
)

spatial_categorical_dataset.to_netcdf(
    SPATIAL_TABLE_DIR
    / "spatial_categorical_metrics_all_finite_raqi.nc"
)

print(spatial_categorical_dataset)

#%%
fig, axes = plot_spatial_mean_maps(
    spatial_mean_dataset,
    product_order=PRODUCT_COLUMNS,
    extent=CONUS_EXTENT,
    vmin=0.0,
    vmax=0.50,
    levels=np.arange(
        0.0,
        0.525,
        0.025,
    ),
    cmap_name="jet",
    customize_low_end=False,
    title=(
            "Mean hourly precipitation over CONUS\n"
            "All finite MRMS RAQI values"
        ),
    output_path=(
        SPATIAL_FIGURE_DIR
        / "spatial_mean_precipitation_all_finite_raqi.png"
    ),
    preliminary_label=PRELIMINARY_LABEL,
)

plt.show()

#%%
fig, axes = plot_spatial_continuous_metric_maps(
    spatial_continuous_dataset,
    product_order=EVALUATED_PRODUCTS,
    extent=CONUS_EXTENT,
    metric_configuration={
        "CC": {
            "levels": np.arange(
                0.0,
                1.01,
                0.1,
            ),
        },
        "relative_bias": {
            "levels": np.array(
                [
                    -100,
                    -75,
                    -50,
                    -30,
                    -20,
                    -10,
                    0,
                    10,
                    20,
                    30,
                    50,
                    75,
                    100,
                ]
            ),
        },
        "RMSE": {
            "levels": np.arange(
                0.0,
                1.05,
                0.05,
            ),
        },
    },
    title=(
        "Spatial quantitative performance relative to MRMS\n"
        f"RAQI ≥ {PRIMARY_RAQI_THRESHOLD:g}; "
        f"N ≥ {MINIMUM_SPATIAL_SAMPLE_COUNT}"
    ),
    output_path=(
        SPATIAL_FIGURE_DIR
        / "spatial_continuous_metrics_all_finite_raqi.png"
    ),
    preliminary_label=PRELIMINARY_LABEL,
)

plt.show()

#%%
fig, axes = (
    plot_spatial_categorical_metric_maps(
        spatial_categorical_dataset,
        product_order=EVALUATED_PRODUCTS,
        extent=CONUS_EXTENT,
        title=(
            "Spatial categorical performance relative to MRMS\n"
            f"Wet threshold = "
            f"{WET_THRESHOLD:g} mm h$^{{-1}}$; "
            "all finite RAQI values"
        ),
        output_path=(
            SPATIAL_FIGURE_DIR
            / "spatial_categorical_metrics_raqi_ge_0p8.png"
        ),
        preliminary_label=PRELIMINARY_LABEL,
    )
)

plt.show()


#%%
fig, ax = plot_spatial_sample_count(
    spatial_mean_dataset,
    variable="sample_count",
    extent=CONUS_EXTENT,
    title=(
            "Common-valid footprint count over CONUS\n"
            f"All finite RAQI values; "
            f"{SPATIAL_GRID_RESOLUTION:g}° grid"
        ),
    output_path=(
        SPATIAL_FIGURE_DIR
        / "spatial_sample_count_raqi_ge_0p8.png"
    ),
    preliminary_label=PRELIMINARY_LABEL,
)

plt.show()


#%%
# =============================================================================
# 25. V7-TO-V8 DIRECT CHANGE SUMMARY
# =============================================================================

v7_mean = df_raqi08_era5[
    "GPROF_V7"
].mean()

v8_mean = df_raqi08_era5[
    "GPROF_V8"
].mean()

v8_minus_v7 = (
    v8_mean - v7_mean
)

v8_percent_change = (
    100.0
    * v8_minus_v7
    / v7_mean
    if not np.isclose(
        v7_mean,
        0.0,
    )
    else np.nan
)

v7_v8_summary = pd.DataFrame(
    {
        "sample_count": [
            len(df_raqi08_era5)
        ],
        "GPROF_V7_mean": [
            v7_mean
        ],
        "GPROF_V8_mean": [
            v8_mean
        ],
        "V8_minus_V7": [
            v8_minus_v7
        ],
        "V8_percent_change_from_V7": [
            v8_percent_change
        ],
    }
)

print(v7_v8_summary)

v7_v8_summary.to_csv(
    TABLE_DIR
    / "gprof_v7_v8_direct_change_summary_raqi_ge_0p8.csv",
    index=False,
)


#%%
# =============================================================================
# 26. FINAL OUTPUT INVENTORY
# =============================================================================

generated_tables = sorted(
    TABLE_DIR.glob("*.csv")
)

generated_figures = sorted(
    FIGURE_DIR.glob("*.png")
)

print(
    f"\nGenerated tables: "
    f"{len(generated_tables):,}"
)

for filepath in generated_tables:
    print(
        f"  {filepath.name}"
    )

print(
    f"\nGenerated figures: "
    f"{len(generated_figures):,}"
)

for filepath in generated_figures:
    print(
        f"  {filepath.name}"
    )

print(
    "\nComparative analysis completed."
)
#%%
"""Interactive manuscript assessment using the ERA5 interval-matched archive.

Run this file cell-by-cell in VS Code.  The safe default loads only a short
orbit subset and leaves manuscript analysis blocks disabled.  Set
USE_SHORT_TEST_SUBSET = False only when ready to build the full dated cache.
"""

from datetime import datetime
from pathlib import Path
import gc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gprof_v7_v8_mrms_stageiv_analysis_functions_footprint_aligned_20260925 import (
    PRECIPITATION_COLUMNS,
    SEASON_ORDER,
    assign_phase_regime,
    calculate_phase_assessment_metrics,
    calculate_precipitation_distributions,
    calculate_reference_agreement,
    calculate_reference_dependence,
    calculate_seasonal_spatial_means,
    calculate_spatial_fraction_differences,
    calculate_spatial_mean_differences,
    calculate_spatial_precipitation_characteristics,
    calculate_spatial_sampling_coverage,
    calculate_temperature_binned_means,
    calculate_twet_binned_v8_v7_metrics,
    create_reference_consistency_summary_table,
    calculate_reference_performance_tables,
    create_spatial_grid,
    discover_matchup_files,
    load_footprint_cache,
    load_matchup_archive,
    save_footprint_cache,
    summarize_sample_counts,
    select_conus_land_footprints,
)
from gprof_v7_v8_mrms_stageiv_plotting_functions_footprint_aligned_20260925 import (
    plot_multireference_performance_diagram,
    plot_multiphase_quantitative_metric_bars,
    plot_occurrence_and_volume_distributions,
    plot_quantitative_metric_bars,
    plot_reference_density_by_phase,
    plot_reference_density,
    plot_reference_robustness_summary,
    plot_seasonal_mean_maps,
    plot_spatial_difference_maps,
    plot_spatial_fraction_maps,
    plot_spatial_mean_and_fraction_difference_maps,
    plot_spatial_mean_and_fraction_maps,
    plot_spatial_mean_maps,
    plot_spatial_sampling_coverage,
    plot_temperature_binned_means_comparison,
    plot_twet_binned_v8_v7_metric_improvements,
    plot_twet_binned_raw_metrics,
)


#%%
# =============================================================================
# ANALYSIS CONFIGURATION
# =============================================================================

CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent

MATCHUP_DIR = Path(
    "/scratch/kkumah/"
    "GPROF_V7_V8_MRMS_RAQI_ERA5_MERRA2_AutoSnow_StageIV_matchups_footprint_aligned_20260925/"
    "data/matchups/GMI"
)

# Large reusable footprint cache is stored on scratch.
CACHE_DIR = Path(
    "/scratch/kkumah/"
    "GPROF_V7_V8_MRMS_RAQI_ERA5_MERRA2_AutoSnow_StageIV_matchups_footprint_aligned_20260925/"
    "data/cache"
)

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_DIR = Path(
    "/scratch/kkumah/"
    "GPROF_V7_V8_MRMS_RAQI_ERA5_MERRA2_AutoSnow_StageIV_"
    "matchups_footprint_aligned_20260925/results/"
    "fullarchive_CONUS_land_20260927"
)
PLOTS_DIR = RESULTS_DIR / "plots"
DFS_DIR = RESULTS_DIR / "dfs"

METHODS_PLOT_DIR = PLOTS_DIR / "methods_sampling_coverage"
METHODS_DF_DIR = DFS_DIR / "methods_sampling_coverage"

SECTION1_PLOT_DIR = PLOTS_DIR / "section1_rain_snow"
SECTION2_PLOT_DIR = PLOTS_DIR / "section2_mean_precipitation"
SECTION3_PLOT_DIR = PLOTS_DIR / "section3_reference_robustness"
SECTION1_DF_DIR = DFS_DIR / "section1_rain_snow"
SECTION2_DF_DIR = DFS_DIR / "section2_mean_precipitation"
SECTION3_DF_DIR = DFS_DIR / "section3_reference_robustness"

for output_directory in (
    METHODS_PLOT_DIR,
    METHODS_DF_DIR,
    SECTION1_PLOT_DIR,
    SECTION2_PLOT_DIR,
    SECTION3_PLOT_DIR,
    SECTION1_DF_DIR,
    SECTION2_DF_DIR,
    SECTION3_DF_DIR,
):
    output_directory.mkdir(parents=True, exist_ok=True)

RUN_DATE_TAG = "20260927"
# RUN_DATE_TAG = "20260825"
CACHE_DATE_TAG = "20260927"
EXPERIMENT_TAG = "footprint_aligned_fullarchive_CONUS_land"


def output_name(descriptor: str, extension: str) -> str:
    """Return descriptor[_experiment]_YYYYMMDD.extension."""
    experiment = f"_{EXPERIMENT_TAG.strip()}" if EXPERIMENT_TAG.strip() else ""
    return f"{descriptor}{experiment}_{RUN_DATE_TAG}.{extension.lstrip('.')}"


GRID_RESOLUTION_DEGREES = 0.25
CONUS_EXTENT = (-125.0, -66.0, 24.0, 50.0)

PRECIPITATION_THRESHOLD_MM_H = 0.1

# =============================================================================
# Rain / snow phase thresholds

# Sims–Liu / IMERG wet-bulb rain–snow partition
# High-confidence snow: T2MWET <= 273.15 K (0 C)
# High-confidence rain: T2MWET >= 275.15 K (2 C)
# Transition/mixed:     273.15 K < T2MWET < 275.15 K
# Unknown:              non-finite T2MWET
# =============================================================================
SNOW_TWET_MAX_K = 273.15
RAIN_TWET_MIN_K = 275.15


# =============================================================================
# TEMPERATURE-BIN SETTINGS
# =============================================================================

TEMPERATURE_BIN_MIN_K = 230.0
TEMPERATURE_BIN_MAX_K = 316.0
TEMPERATURE_BIN_WIDTH_K = 2.0

COLD_TEMPERATURE_MAX_K = 275.0

TWET_BIN_MIN_K = 250.0
TWET_BIN_MAX_K = 290.0
TWET_BIN_WIDTH_K = 2.0
MINIMUM_TWET_BIN_COUNT = 1000
MINIMUM_REFERENCE_EVENT_COUNT = 100

MINIMUM_SPATIAL_SAMPLE_COUNT = None
FIGURE_DPI = 150

PRECIPITATION_INTENSITY_BINS_MM_H = np.array(
    [0.1, 0.2, 0.5, 1, 2, 4, 8, 16, 32, 64, np.inf], dtype=float
)
# Annual spatial mean: detailed color intervals, sparse labels.
ANNUAL_SPATIAL_MEAN_LEVELS_MM_H = np.linspace(
    0.0,
    0.32,
    21,
).round(2)

ANNUAL_SPATIAL_MEAN_COLORBAR_TICKS = np.array(
    [
        0.00,
        0.06,
        0.13,
        0.20,
        0.26,
        0.32,
    ],
    dtype=float,
)

# Retain the broader range for seasonal maps.
SEASONAL_MEAN_LEVELS_MM_H = np.linspace(0.0, 0.45, 21).round(4)
SEASONAL_MEAN_COLORBAR_TICKS = SEASONAL_MEAN_LEVELS_MM_H[[0, 4, 8, 12, 16, 20]]
SPATIAL_MEAN_LEVELS_MM_H = np.array(
    [
        0.000,
        0.010,
        0.020,
        0.035,
        0.050,
        0.075,
        0.100,
        0.125,
        0.150,
        0.200,
        0.250,
        0.300,
        0.375,
        0.450,
    ],
    dtype=float,
)

SPATIAL_FRACTION_LEVELS_PERCENT = np.array(
    [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        8,
        10,
        12,
        15,
        18,
        20,
        22,
        25,
        28,
        30,
        32,
        35,
    ],
    dtype=float,
)

SPATIAL_FRACTION_COLORBAR_TICKS = np.array(
    [
        0,
        2,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
    ],
    dtype=float,
)

SPATIAL_DIFFERENCE_LEVELS_MM_H = np.array(
    [-0.50, -0.30, -0.20, -0.10, -0.05, -0.02, 0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
)

# No scientific tolerance is assumed.  Populate metric-specific values here if
# a negligible-change category is later justified (units are metric-specific).
REFERENCE_IMPROVEMENT_TOLERANCES = None

# =============================================================================
# INPUT SUBSET SETTINGS
# =============================================================================

# Analysis input mode:
#   "short_test" = use a small predefined/file-count subset for debugging
#   "date_range" = use every matchup file within the requested date range
#   "full_archive" = use every available matchup file
INPUT_MODE = "date_range"

# -------------------------------------------------------------------------
# Short-test settings
# -------------------------------------------------------------------------
SHORT_TEST_FILE_COUNT = 100

SHORT_TEST_FILE_NAMES = (
    "GMI_GPROF_V7_V8_MRMS_RAQI_ERA5_MERRA2_AutoSnow_StageIV_038889_20210101.nc",
    "GMI_GPROF_V7_V8_MRMS_RAQI_ERA5_MERRA2_AutoSnow_StageIV_038890_20210101.nc",
    "GMI_GPROF_V7_V8_MRMS_RAQI_ERA5_MERRA2_AutoSnow_StageIV_038891_20210101.nc",
)

# -------------------------------------------------------------------------
# Date-range settings
# -------------------------------------------------------------------------
# Inclusive date range.
ANALYSIS_START_DATE = "20210101"
ANALYSIS_END_DATE   = "20241231"
EXPECTED_MATCHUP_FILE_COUNT = 21_971

# -------------------------------------------------------------------------
# Force this manuscript run to read the matched NetCDF source archive rather
# than reusing any earlier footprint cache. The rebuilt cache remains on scratch.
REBUILD_FOOTPRINT_CACHE = True
SKIP_FAILED_FILES = False
SHOW_LOAD_PROGRESS = True
LOAD_PROGRESS_INTERVAL_FILES = 1000
# Change to 2.5 or 5.0 for sensitivity tests. Because every aligned pair and
# its separation are retained in the NetCDF files, this does not require a
# collocation rerun; each threshold gets a separate scratch cache.
MAXIMUM_PAIR_SEPARATION_KM = 1.0

RUN_SECTION_1_1 = True
RUN_SECTION_1_2 = True
RUN_SECTION_2_1 = True
RUN_SECTION_2_2 = True
RUN_SECTION_2_3 = True
RUN_SECTION_2_4 = True
RUN_SECTION_3_1 = True
RUN_SECTION_3_2 = True
RUN_METHODS_COVERAGE_MAP = True

# =============================================================================
# CACHE NAMING
# =============================================================================

if INPUT_MODE == "short_test":

    CACHE_DESCRIPTOR = "footprint_matchups_shorttest"

elif INPUT_MODE == "date_range":

    CACHE_DESCRIPTOR = (
        f"footprint_matchups_"
        f"{ANALYSIS_START_DATE}_"
        f"{ANALYSIS_END_DATE}"
    )

elif INPUT_MODE == "full_archive":

    CACHE_DESCRIPTOR = "footprint_matchups_fullarchive"

else:

    raise ValueError(
        "INPUT_MODE must be one of: "
        "'short_test', 'date_range', or 'full_archive'."
    )

PAIR_SEPARATION_TAG = f"pairsep_{MAXIMUM_PAIR_SEPARATION_KM:g}km".replace(".", "p")
CACHE_DESCRIPTOR = f"{CACHE_DESCRIPTOR}_{PAIR_SEPARATION_TAG}"


FOOTPRINT_CACHE_FILE = (
    CACHE_DIR
    / f"{CACHE_DESCRIPTOR}_{CACHE_DATE_TAG}.pkl"
)

LOAD_SUMMARY_FILE = (
    CACHE_DIR
    / f"{CACHE_DESCRIPTOR}_load_summary_{CACHE_DATE_TAG}.csv"
)

SAMPLE_COUNTS_FILE = (
    DFS_DIR
    / output_name(
        f"{CACHE_DESCRIPTOR}_sample_counts",
        "csv",
    )
)

print(
    "Footprint cache target:",
    FOOTPRINT_CACHE_FILE,
)

print(
    "Cache inventory target:",
    LOAD_SUMMARY_FILE,
)

#%%
# =============================================================================
# INPUT DISCOVERY AND FOOTPRINT CACHE
# =============================================================================

matchup_files = discover_matchup_files(MATCHUP_DIR)
if not matchup_files:
    raise FileNotFoundError(f"No orbit matchup files found in {MATCHUP_DIR}")

# -----------------------------------------------------------------------------
# Select matchup files according to requested analysis mode
# -----------------------------------------------------------------------------

if INPUT_MODE == "short_test":

    files_by_name = {
        path.name: path
        for path in matchup_files
    }

    selected_matchup_files = [
        files_by_name[name]
        for name in SHORT_TEST_FILE_NAMES
        if name in files_by_name
    ]

    # Fall back to the first N files if the explicitly named
    # short-test files are not available.
    if not selected_matchup_files:

        selected_matchup_files = (
            matchup_files[
                :SHORT_TEST_FILE_COUNT
            ]
        )


elif INPUT_MODE == "date_range":

    selected_matchup_files = []

    for path in matchup_files:

        # Filename convention:
        #
        # ..._StageIV_<orbit>_YYYYMMDD.nc
        #
        # Therefore the final underscore-separated token
        # before ".nc" is the observation date.
        date_string = (
            path.stem
            .split("_")[-1]
        )

        if (
            ANALYSIS_START_DATE
            <= date_string
            <= ANALYSIS_END_DATE
        ):

            selected_matchup_files.append(
                path
            )

    if not selected_matchup_files:

        raise FileNotFoundError(
            "No matchup files were found between "
            f"{ANALYSIS_START_DATE} and "
            f"{ANALYSIS_END_DATE}."
        )


elif INPUT_MODE == "full_archive":

    selected_matchup_files = (
        matchup_files
    )


else:

    raise ValueError(
        "INPUT_MODE must be one of: "
        "'short_test', 'date_range', or 'full_archive'."
    )


# Fail closed if the intended manuscript archive is incomplete or if selection
# drifts away from the verified 2021-2024 matched-file inventory.
if INPUT_MODE == "date_range":
    selected_date_strings = [path.stem.split("_")[-1] for path in selected_matchup_files]
    selected_start_date = min(selected_date_strings)
    selected_end_date = max(selected_date_strings)
    if selected_start_date != ANALYSIS_START_DATE:
        raise RuntimeError(
            f"Selected archive starts at {selected_start_date}, not {ANALYSIS_START_DATE}."
        )
    if selected_end_date != ANALYSIS_END_DATE:
        raise RuntimeError(
            f"Selected archive ends at {selected_end_date}, not {ANALYSIS_END_DATE}."
        )
    if len(selected_matchup_files) != EXPECTED_MATCHUP_FILE_COUNT:
        raise RuntimeError(
            "Matched archive inventory changed: "
            f"expected {EXPECTED_MATCHUP_FILE_COUNT:,} files but selected "
            f"{len(selected_matchup_files):,}. Re-audit the archive before running."
        )


print(
    f"Discovered {len(matchup_files):,} orbit files "
    f"in {MATCHUP_DIR}"
)

print(
    f"Selected {len(selected_matchup_files):,} orbit files "
    f"using INPUT_MODE={INPUT_MODE!r}"
)

if INPUT_MODE == "date_range":

    print(
        "Requested analysis period: "
        f"{ANALYSIS_START_DATE} to "
        f"{ANALYSIS_END_DATE}"
    )

    print(
        "First selected file:",
        selected_matchup_files[0].name,
    )

    print(
        "Last selected file:",
        selected_matchup_files[-1].name,
    )

print(f"Discovered {len(matchup_files):,} orbit files in {MATCHUP_DIR}")
print(f"Selected {len(selected_matchup_files):,} orbit files for this interactive run")

cached_load_summary = (
    pd.read_csv(LOAD_SUMMARY_FILE) if LOAD_SUMMARY_FILE.exists() else None
)
expected_source_files = {path.name for path in selected_matchup_files}
cached_source_files = (
    set(cached_load_summary["source_file"].astype(str))
    if cached_load_summary is not None and "source_file" in cached_load_summary
    else set()
)
cache_matches_selected_archive = (
    FOOTPRINT_CACHE_FILE.exists()
    and cached_source_files == expected_source_files
)

if REBUILD_FOOTPRINT_CACHE:
    print(
        "Footprint-cache read disabled: rebuilding directly from "
        f"{len(selected_matchup_files):,} matched NetCDF files."
    )

if cache_matches_selected_archive and not REBUILD_FOOTPRINT_CACHE:
    df_footprints = load_footprint_cache(FOOTPRINT_CACHE_FILE)
    load_summary = cached_load_summary
    print(f"Reused footprint cache: {FOOTPRINT_CACHE_FILE}")
else:
    if FOOTPRINT_CACHE_FILE.exists() and not REBUILD_FOOTPRINT_CACHE:
        print("Cached source-file inventory differs from the selected archive; rebuilding.")
    df_footprints, load_summary = load_matchup_archive(
        selected_matchup_files,
        land_only=True,
        extent=CONUS_EXTENT,
        retain_identifiers=True,
        maximum_pair_separation_km=MAXIMUM_PAIR_SEPARATION_KM,
        skip_failed_files=SKIP_FAILED_FILES,
        show_progress=SHOW_LOAD_PROGRESS,
        progress_interval=LOAD_PROGRESS_INTERVAL_FILES,
    )
    save_footprint_cache(df_footprints, FOOTPRINT_CACHE_FILE)
    load_summary.to_csv(LOAD_SUMMARY_FILE, index=False)
    print(f"Saved footprint cache: {FOOTPRINT_CACHE_FILE}")

#%%
# Apply the geographic selection AFTER saving/loading the reusable full cache,
# and BEFORE every downstream summary, metric, distribution and map.
df_footprints, conus_selection_counts = select_conus_land_footprints(df_footprints)
conus_selection_counts.to_csv(
    RESULTS_DIR / output_name("CONUS_land_selection_counts", "csv"), index=False,
)
print(conus_selection_counts.to_string(index=False))

df_footprints = assign_phase_regime(
    df_footprints,
    snow_twet_max_k=SNOW_TWET_MAX_K,
    rain_twet_min_k=RAIN_TWET_MIN_K,
)

total_footprints_seen = (
    int(load_summary["rows_total"].sum())
    if load_summary is not None and "rows_total" in load_summary
    else None
)
sample_counts = summarize_sample_counts(
    df_footprints,
    total_footprints_seen=total_footprints_seen,
)
sample_counts.rename_axis("population").reset_index().to_csv(SAMPLE_COUNTS_FILE, index=False)
print(sample_counts.to_string())
print(f"Footprint table memory: {df_footprints.memory_usage(deep=True).sum() / 1024**2:,.1f} MiB")


#%%
# =============================================================================
# METHODS FIGURE: COMMON V7-V8 GMI SAMPLING COVERAGE
# =============================================================================

if RUN_METHODS_COVERAGE_MAP:
    coverage_grid = create_spatial_grid(
        extent=CONUS_EXTENT,
        resolution=GRID_RESOLUTION_DEGREES,
    )
    sampling_coverage = calculate_spatial_sampling_coverage(
        df_footprints,
        grid=coverage_grid,
    )
    sampling_coverage.to_dataframe().reset_index().to_pickle(
        METHODS_DF_DIR
        / output_name("common_v7_v8_gmi_sampling_coverage_025deg", "pkl")
    )
    fig_sampling_coverage, ax_sampling_coverage = plot_spatial_sampling_coverage(
        sampling_coverage,
        period_label="2021–2024",
        dpi=FIGURE_DPI,
        output_path=(
            METHODS_PLOT_DIR
            / output_name("common_v7_v8_gmi_sampling_coverage_025deg", "png")
        ),
    )


#%%
# =============================================================================
# 1. RAIN–SNOW COMPARATIVE ASSESSMENT
# =============================================================================

# 1.1 Detection and quantitative performance
if RUN_SECTION_1_1:
    phase_continuous_metrics, phase_categorical_metrics = calculate_phase_assessment_metrics(
        df_footprints,
        precipitation_threshold=PRECIPITATION_THRESHOLD_MM_H,
    )
    phase_continuous_metrics.to_csv(
        SECTION1_DF_DIR / output_name("rain_snow_continuous_metrics", "csv"), index=False
    )
    phase_categorical_metrics.to_csv(
        SECTION1_DF_DIR / output_name("rain_snow_categorical_metrics", "csv"), index=False
    )

    fig_phase_roebber, axes_phase_roebber = plot_multireference_performance_diagram(
        phase_categorical_metrics,
        phase_order=("rain", "snow"),
        reference_order=("MRMS", "StageIV"),
        threshold=PRECIPITATION_THRESHOLD_MM_H,
        dpi=FIGURE_DPI,
        output_path=SECTION1_PLOT_DIR / output_name("rain_snow_roebber_multireference", "png"),
    )
    fig_phase_metrics, axes_phase_metrics = plot_multiphase_quantitative_metric_bars(
        phase_continuous_metrics,
        phase_order=("rain", "snow"),
        reference_order=("MRMS", "StageIV"),
        dpi=FIGURE_DPI,
        output_path=SECTION1_PLOT_DIR / output_name("rain_snow_quantitative_metrics", "png"),
    )
    print(phase_continuous_metrics[["phase", "reference_label", "product", "N", "correlation", "relative_bias_percent", "RMSE", "MAE"]])


#%%
# 1.2 Distributional assessment by precipitation intensity
if RUN_SECTION_1_2:
    rain_distributions = calculate_precipitation_distributions(
        df_footprints,
        product_columns=(
            "GPROF_V7",
            "GPROF_V8",
            "MRMS",
            "StageIV",
            "ERA5",
        ),
        bin_edges=PRECIPITATION_INTENSITY_BINS_MM_H,
        phase="rain",
        precipitation_threshold=PRECIPITATION_THRESHOLD_MM_H,
    )
    snow_distributions = calculate_precipitation_distributions(
        df_footprints,
        product_columns=(
            "GPROF_V7",
            "GPROF_V8",
            "MRMS",
            "StageIV",
            "ERA5",
        ),
        bin_edges=PRECIPITATION_INTENSITY_BINS_MM_H,
        phase="snow",
        precipitation_threshold=PRECIPITATION_THRESHOLD_MM_H,
    )
    rain_distributions.to_csv(
        SECTION1_DF_DIR / output_name("rain_precipitation_distributions", "csv"), index=False
    )
    snow_distributions.to_csv(
        SECTION1_DF_DIR / output_name("snow_precipitation_distributions", "csv"), index=False
    )
    fig_distributions, axes_distributions = (
    plot_occurrence_and_volume_distributions(
        rain_distributions,
        snow_distributions,
        rain_product_order=(
            "GPROF_V7",
            "GPROF_V8",
            "MRMS",
            "StageIV",
            "ERA5",
        ),
        snow_product_order=(
            "GPROF_V7",
            "GPROF_V8",
            "MRMS",
            "StageIV",
            "ERA5",
        ),
        dpi=FIGURE_DPI,
        output_path=(
            SECTION1_PLOT_DIR
            / output_name(
                "rain_snow_occurrence_volume_distributions",
                "png",
            )
        ),
    )
)

# plt.show()

gc.collect()
#%%
# =============================================================================
# 2. ASSESSMENT OF MEAN PRECIPITATION CHARACTERISTICS
# =============================================================================

analysis_grid = create_spatial_grid(extent=CONUS_EXTENT, resolution=GRID_RESOLUTION_DEGREES)

# 2.1 Mean precipitation intensity and fraction
if RUN_SECTION_2_1:
    spatial_characteristics = calculate_spatial_precipitation_characteristics(
        df_footprints,
        grid=analysis_grid,
        product_columns=PRECIPITATION_COLUMNS,
        precipitation_threshold=PRECIPITATION_THRESHOLD_MM_H,
        minimum_sample_count=MINIMUM_SPATIAL_SAMPLE_COUNT,
    )
    spatial_characteristics.to_dataframe().reset_index().to_pickle(
        SECTION2_DF_DIR / output_name("spatial_precipitation_characteristics_025deg", "pkl")
    )
    # Subtract the displayed means: all products use the same common-valid
    # footprint population, rather than a separate pairwise-valid sample.
    spatial_mean_differences = spatial_characteristics[[]].copy()
    for first, second in (
            ("GPROF_V8", "MRMS"),
            ("GPROF_V8", "StageIV"),
            ("GPROF_V7", "MRMS"),
            ("GPROF_V7", "StageIV"),
            ("ERA5", "MRMS"),
            ("ERA5", "StageIV"),
    ):
        difference_name = f"{first}_minus_{second}"
        spatial_mean_differences[difference_name] = (
            spatial_characteristics[f"{first}_mean"]
            - spatial_characteristics[f"{second}_mean"]
        )
        spatial_mean_differences[difference_name].attrs.update(
            units="mm h-1",
            sign_convention=f"positive means {first} exceeds {second}",
        )
        spatial_mean_differences[f"{difference_name}_sample_count"] = (
            spatial_characteristics["common_sample_count"]
        )
    spatial_mean_differences.attrs["difference_method"] = (
        "difference of spatial means from the five-product common-valid sample"
    )
    spatial_mean_differences.to_dataframe().reset_index().to_pickle(
        SECTION2_DF_DIR / output_name("spatial_mean_differences_025deg", "pkl")
    )

    spatial_fraction_differences = (
        calculate_spatial_fraction_differences(
            spatial_characteristics,
            pairs=(
                ("GPROF_V8", "MRMS"),
                ("GPROF_V8", "StageIV"),
                ("GPROF_V7", "MRMS"),
                ("GPROF_V7", "StageIV"),
                ("ERA5", "MRMS"),
                ("ERA5", "StageIV"),
            ),
        )
    )

    spatial_fraction_differences.to_dataframe().reset_index().to_pickle(
        SECTION2_DF_DIR
        / output_name(
            "spatial_fraction_differences_025deg",
            "pkl",
        )
    )

    fig_spatial_characteristics, axes_spatial_characteristics = (
            plot_spatial_mean_and_fraction_maps(
                spatial_characteristics,
                product_order=PRECIPITATION_COLUMNS,
                mean_levels=ANNUAL_SPATIAL_MEAN_LEVELS_MM_H,
                mean_colorbar_ticks=(
                    ANNUAL_SPATIAL_MEAN_COLORBAR_TICKS
                ),
                fraction_levels=(
                    SPATIAL_FRACTION_LEVELS_PERCENT
                ),
                fraction_colorbar_ticks=(
                    SPATIAL_FRACTION_COLORBAR_TICKS
                ),
                dpi=FIGURE_DPI,
                output_path=(
                    SECTION2_PLOT_DIR
                    / output_name(
                        "mean_precipitation_and_fraction_maps_025deg",
                        "png",
                    )
                ),
            )
        )

    fig_spatial_differences, axes_spatial_differences = (
        plot_spatial_mean_and_fraction_difference_maps(
            spatial_mean_differences,
            spatial_fraction_differences,
            dpi=FIGURE_DPI,
            output_path=(
                SECTION2_PLOT_DIR
                / output_name(
                    "mean_precipitation_and_fraction_difference_maps_025deg",
                    "png",
                )
            ),
        )
    )
    
    print("Maximum per-product valid sample count by grid cell:")
    print({product: int(spatial_characteristics[f"{product}_sample_count"].max()) for product in PRECIPITATION_COLUMNS})

gc.collect()
#%%
# 2.2 Seasonal mean precipitation
if RUN_SECTION_2_2:
    seasonal_spatial_means = calculate_seasonal_spatial_means(
        df_footprints,
        grid=analysis_grid,
        product_columns=PRECIPITATION_COLUMNS,
        minimum_sample_count=MINIMUM_SPATIAL_SAMPLE_COUNT,
    )
    seasonal_spatial_means.to_dataframe().reset_index().to_pickle(
        SECTION2_DF_DIR / output_name("seasonal_mean_precipitation_025deg", "pkl")
    )
    fig_seasonal_means, axes_seasonal_means = (
        plot_seasonal_mean_maps(
            seasonal_spatial_means,
            product_order=PRECIPITATION_COLUMNS,
            season_order=SEASON_ORDER,
            levels=SEASONAL_MEAN_LEVELS_MM_H,
            colorbar_ticks=SEASONAL_MEAN_COLORBAR_TICKS,
            dpi=FIGURE_DPI,
            output_path=(
                SECTION2_PLOT_DIR
                / output_name(
                    "seasonal_mean_precipitation_maps_025deg",
                    "png",
                )
            ),
        )
    )

gc.collect()
#%%
# 2.3 Mean precipitation as a function of temperature
# =============================================================================

if RUN_SECTION_2_3:

    # -------------------------------------------------------------------------
    # Full temperature-bin definition
    # -------------------------------------------------------------------------
    temperature_bin_edges = np.arange(
        TEMPERATURE_BIN_MIN_K,
        TEMPERATURE_BIN_MAX_K + TEMPERATURE_BIN_WIDTH_K,
        TEMPERATURE_BIN_WIDTH_K,
    )

    # -------------------------------------------------------------------------
    # Calculate full-range temperature-binned precipitation means
    # -------------------------------------------------------------------------
    temperature_binned_means = calculate_temperature_binned_means(
        df_footprints,
        temperature_bin_edges=temperature_bin_edges,
        product_columns=PRECIPITATION_COLUMNS,
    )

    # Save the full table.
    temperature_binned_means.to_csv(
        SECTION2_DF_DIR
        / output_name(
            "temperature_binned_mean_precipitation_full",
            "csv",
        ),
        index=False,
    )

    # -------------------------------------------------------------------------
    # Cold-temperature subset
    #
    # Use bins whose midpoint is <= 275 K.
    # This preserves exactly the same bin statistics used in the full plot.
    # -------------------------------------------------------------------------
    temperature_binned_means_cold = (
        temperature_binned_means.loc[
            temperature_binned_means[
                "temperature_midpoint"
            ]
            <= COLD_TEMPERATURE_MAX_K
        ]
        .copy()
        .reset_index(drop=True)
    )

    temperature_binned_means_cold.to_csv(
        SECTION2_DF_DIR
        / output_name(
            "temperature_binned_mean_precipitation_cold_le275K",
            "csv",
        ),
        index=False,
    )

    # -------------------------------------------------------------------------
    # Combined full-range and cold-temperature plot
    # -------------------------------------------------------------------------
    fig_temperature, axes_temperature = (
        plot_temperature_binned_means_comparison(
            temperature_binned_means,
            temperature_binned_means_cold,
            product_order=PRECIPITATION_COLUMNS,
            dpi=FIGURE_DPI,
            output_path=(
                SECTION2_PLOT_DIR
                / output_name(
                    "mean_precipitation_by_t2m_full_and_cold_le275K",
                    "png",
                )
            ),
        )
    )

    print(temperature_binned_means)

gc.collect()

#%%
# 2.4 V7/V8 performance as a function of wet-bulb temperature
if RUN_SECTION_2_4:
    twet_bin_edges = np.arange(
        TWET_BIN_MIN_K,
        TWET_BIN_MAX_K + TWET_BIN_WIDTH_K,
        TWET_BIN_WIDTH_K,
    )
    twet_metric_tables = []
    for reference in ("MRMS", "StageIV"):
        table = calculate_twet_binned_v8_v7_metrics(
            df_footprints,
            twet_bin_edges=twet_bin_edges,
            reference_column=reference,
            precipitation_threshold=PRECIPITATION_THRESHOLD_MM_H,
            minimum_bin_count=MINIMUM_TWET_BIN_COUNT,
            minimum_reference_event_count=MINIMUM_REFERENCE_EVENT_COUNT,
        )
        twet_metric_tables.append(table)
    twet_metrics = pd.concat(twet_metric_tables, ignore_index=True)
    twet_metrics.to_csv(
        SECTION2_DF_DIR / output_name("twet_binned_v8_v7_metrics_by_autosnow", "csv"),
        index=False,
    )
    fig_twet, axes_twet = (
        plot_twet_binned_v8_v7_metric_improvements(
            twet_metrics,
            reference_order=(
                "MRMS",
                "StageIV",
            ),
            dpi=FIGURE_DPI,
            output_path=(
                SECTION2_PLOT_DIR
                / output_name(
                    "twet_binned_v8_v7_metric_improvements_by_autosnow",
                    "png",
                )
            ),
        )
    )

    # ------------------------------------------------------------------
    # Raw V7 and V8 metric values
    # ------------------------------------------------------------------
    fig_twet_raw, axes_twet_raw = (
        plot_twet_binned_raw_metrics(
            twet_metrics,
            reference_order=(
                "MRMS",
                "StageIV",
            ),
            dpi=FIGURE_DPI,
            output_path=(
                SECTION2_PLOT_DIR
                / output_name(
                    "twet_binned_raw_v7_v8_era5_metrics_by_snow_cover",
                    "png",
                )
            ),
        )
    )
    # print(twet_metrics[["reference_label", "surface_regime", "twet_midpoint", "N"]])
gc.collect()
#%%
# =============================================================================
# 3. REFERENCE CONSISTENCY AND ROBUSTNESS OF V8–V7 DIFFERENCES
# =============================================================================

# 3.1 MRMS–Stage IV agreement
if RUN_SECTION_3_1:
    reference_samples = {}
    continuous_tables = []
    categorical_tables = []
    consistency_tables = []

    for phase in ("rain", "snow"):
        sample, continuous, categorical = calculate_reference_agreement(
            df_footprints,
            precipitation_threshold=PRECIPITATION_THRESHOLD_MM_H,
            phase=phase,
        )
        reference_samples[phase] = sample
        continuous_tables.append(continuous)
        categorical_tables.append(categorical)
        summary = create_reference_consistency_summary_table(continuous, categorical)
        summary.insert(0, "phase", phase)
        consistency_tables.append(summary)

    reference_continuous_metrics = pd.concat(continuous_tables, ignore_index=True)
    reference_categorical_metrics = pd.concat(categorical_tables, ignore_index=True)
    reference_consistency_table = pd.concat(consistency_tables, ignore_index=True)

    reference_continuous_metrics.to_csv(
        SECTION3_DF_DIR / output_name("mrms_stageiv_continuous_metrics_by_phase", "csv"),
        index=False,
    )
    reference_categorical_metrics.to_csv(
        SECTION3_DF_DIR / output_name("mrms_stageiv_categorical_metrics_by_phase", "csv"),
        index=False,
    )
    reference_consistency_table.to_csv(
        SECTION3_DF_DIR / output_name("mrms_stageiv_consistency_by_phase", "csv"),
        index=False,
    )

    fig_reference_density, axes_reference_density = plot_reference_density_by_phase(
        reference_samples,
        dpi=FIGURE_DPI,
        output_path=SECTION3_PLOT_DIR / output_name(
            "mrms_stageiv_rain_snow_density", "png"
        ),
    )
    print(reference_continuous_metrics)
    print(reference_categorical_metrics)

    print(
    "\nMRMS–Stage IV reference consistency\n"
    )

    print(
        reference_consistency_table.to_string(
            index=False,
        )
    )
gc.collect()

#%%
# =============================================================================
# 3.2 Reference dependence of GPROF V8 performance improvement
# =============================================================================

if RUN_SECTION_3_2:
    reference_performance_tables_by_phase = {}

    metric_columns = ["GPROF V7", "GPROF V8", "ERA5"]
    improvement_columns = ["ΔV8 versus V7", "ΔV8 versus ERA5"]

    for phase in ("rain", "snow"):
        reference_performance_tables = calculate_reference_performance_tables(
            df_footprints,
            precipitation_threshold=PRECIPITATION_THRESHOLD_MM_H,
            phase=phase,
        )
        reference_performance_tables_by_phase[phase] = (
            reference_performance_tables
        )

        mrms_performance_table = reference_performance_tables["MRMS"]
        stageiv_performance_table = reference_performance_tables["StageIV"]

        # Retain the established rain filenames and add an explicit snow suffix.
        phase_suffix = "_rain" if phase == "rain" else "_snow"

        mrms_performance_table.to_csv(
            SECTION3_DF_DIR
            / output_name(
                f"gprof_v8_improvement_mrms_reference{phase_suffix}",
                "csv",
            ),
            index=False,
            float_format="%.2f",
        )

        stageiv_performance_table.to_csv(
            SECTION3_DF_DIR
            / output_name(
                f"gprof_v8_improvement_stageiv_reference{phase_suffix}",
                "csv",
            ),
            index=False,
            float_format="%.2f",
        )

        for reference_label, performance_table in (
            ("MRMS", mrms_performance_table),
            ("STAGE IV", stageiv_performance_table),
        ):
            display_table = performance_table.copy()
            display_table[metric_columns] = (
                display_table[metric_columns].round(2)
            )
            display_table[improvement_columns] = (
                display_table[improvement_columns].round(2)
            )

            print(
                "\n"
                + "=" * 80
                + f"\n{phase.upper()} — {reference_label} REFERENCE\n"
                + "=" * 80
            )
            print(display_table.to_string(index=False, float_format=lambda value: f"{value:.2f}"))

        print(
            f"\nCommon {phase} sample N = "
            f"{mrms_performance_table.attrs['common_sample_count']:,}"
        )

gc.collect()
#%%
# =============================================================================
# OPTIONAL INTERACTIVE CLEANUP
# =============================================================================

# Close figures only when finished inspecting them interactively.
# plt.close("all")
# del large_temporary_object
gc.collect()

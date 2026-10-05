"""
Analysis utilities for the GMI GPROF V7/V8 comparison with MRMS.

This module contains:
    - Matchup-file discovery
    - Compact NetCDF loading
    - Common-sample filtering
    - Surface and temperature stratification
    - Continuous and categorical statistics
    - Precipitation PDFs
    - Optional Parquet caching

Plotting functions should be placed in:
    gprof_v7_v8_plotting_functions.py
"""

from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr


# =============================================================================
# Constants and configuration
# =============================================================================

NETCDF_TO_ANALYSIS_COLUMNS: dict[str, str] = {
    "surfacePrecipitation_V7": "GPROF_V7",
    "surfacePrecipitation_V8": "GPROF_V8",
    "MRMS_Pass2": "MRMS",
    "RAQI": "RAQI",
    "ERA5_precipitation": "ERA5",
    "MERRA2_T2M": "T2M",
    "AutoSnow": "AutoSnow",
    "mrms_valid_flag": "mrms_valid_flag",
    "latitude": "latitude",
    "longitude": "longitude",
}

REQUIRED_NETCDF_VARIABLES: tuple[str, ...] = (
    "surfacePrecipitation_V7",
    "surfacePrecipitation_V8",
    "MRMS_Pass2",
    "ERA5_precipitation",
    "MERRA2_T2M",
    "AutoSnow",
    "mrms_valid_flag",
    "latitude",
    "longitude",
)

BASE_REFERENCE_COLUMNS: tuple[str, ...] = (
    "GPROF_V7",
    "GPROF_V8",
    "MRMS",
    "mrms_valid_flag",
)

CONTINUOUS_COLUMNS: tuple[str, ...] = (
    "GPROF_V7",
    "GPROF_V8",
    "MRMS",
    "RAQI",
    "ERA5",
    "T2M",
)

AUTOSNOW_CLASS_NAMES: dict[int, str] = {
    0: "clear water",
    1: "snow-free land",
    2: "snow-covered land",
    3: "ice-covered water",
}

DEFAULT_PRODUCT_COLUMNS: tuple[str, ...] = (
    "GPROF_V7",
    "GPROF_V8",
    "MRMS",
    "ERA5",
)

DEFAULT_TEMPERATURE_BIN_EDGES: np.ndarray = np.array(
    [
        -np.inf,
        263.15,
        273.15,
        283.15,
        np.inf,
    ],
    dtype=float,
)

DEFAULT_TEMPERATURE_BIN_LABELS: tuple[str, ...] = (
    "T2M < 263.15 K",
    "263.15 <= T2M < 273.15 K",
    "273.15 <= T2M < 283.15 K",
    "T2M >= 283.15 K",
)

DEFAULT_PRECIPITATION_BIN_EDGES: np.ndarray = np.array(
    [
        0.0,
        0.01,
        0.03,
        0.10,
        0.20,
        0.50,
        1.00,
        2.00,
        4.00,
        8.00,
        16.00,
        32.00,
        64.00,
        128.00,
        np.inf,
    ],
    dtype=float,
)


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass(frozen=True)
class MatchupLoadSummary:
    """Summary of one matchup-file loading operation."""

    source_file: str
    source_path: str
    rows_total: int
    rows_retained: int
    status: str
    message: str = ""


# =============================================================================
# General helpers
# =============================================================================

def ensure_columns(
    data: pd.DataFrame,
    required_columns: Sequence[str],
) -> None:
    """
    Raise KeyError when required columns are absent.
    """

    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise KeyError(
            "Missing required DataFrame columns: "
            + ", ".join(missing)
        )


def replace_infinite_with_nan(
    data: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Replace positive and negative infinity with NaN.
    """

    result = data.copy()

    if columns is None:
        columns = result.select_dtypes(
            include=[np.number]
        ).columns.tolist()

    existing = [
        column
        for column in columns
        if column in result.columns
    ]

    result.loc[:, existing] = result.loc[:, existing].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return result


def dataframe_memory_mib(data: pd.DataFrame) -> float:
    """
    Return deep DataFrame memory usage in MiB.
    """

    return float(
        data.memory_usage(index=True, deep=True).sum()
        / (1024.0 ** 2)
    )


def downcast_matchup_dataframe(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reduce DataFrame memory usage while preserving scientific values.
    """

    result = data.copy()

    float_columns = [
        column
        for column in CONTINUOUS_COLUMNS
        if column in result.columns
    ]

    for column in float_columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
            downcast="float",
        )

    if "AutoSnow" in result.columns:
        autosnow = pd.to_numeric(
            result["AutoSnow"],
            errors="coerce",
        )

        # Keep nullable UInt8 so missing values can remain missing.
        result["AutoSnow"] = autosnow.astype("UInt8")

    if "mrms_valid_flag" in result.columns:
        result["mrms_valid_flag"] = (
            pd.to_numeric(
                result["mrms_valid_flag"],
                errors="coerce",
            )
            .fillna(0)
            .astype("uint8")
        )

    if "scan" in result.columns:
        result["scan"] = pd.to_numeric(
            result["scan"],
            errors="coerce",
            downcast="unsigned",
        )

    if "pixel" in result.columns:
        result["pixel"] = pd.to_numeric(
            result["pixel"],
            errors="coerce",
            downcast="unsigned",
        )

    if "source_file" in result.columns:
        result["source_file"] = result[
            "source_file"
        ].astype("category")

    if "orbit" in result.columns:
        result["orbit"] = result[
            "orbit"
        ].astype("category")

    return result


def extract_orbit_and_date_from_filename(
    filepath: str | os.PathLike[str],
) -> tuple[str | None, str | None]:
    """
    Extract orbit number and YYYYMMDD date from a matchup filename.

    Expected pattern includes:
        ..._<orbit>_<YYYYMMDD>.nc
    """

    filename = Path(filepath).name

    match = re.search(
        r"_(\d{6})_(\d{8})\.nc$",
        filename,
    )

    if match is None:
        return None, None

    orbit, date_string = match.groups()
    return orbit, date_string


# =============================================================================
# Matchup-file discovery and loading
# =============================================================================

def discover_matchup_files(
    matchup_directory: str | os.PathLike[str],
    *,
    recursive: bool = False,
) -> list[Path]:
    """
    Discover completed NetCDF matchup files.

    Files ending in '.partial' are excluded automatically.
    """

    matchup_directory = Path(matchup_directory)

    if not matchup_directory.exists():
        raise FileNotFoundError(
            f"Matchup directory does not exist: "
            f"{matchup_directory}"
        )

    pattern = "**/*.nc" if recursive else "*.nc"

    files = sorted(
        filepath
        for filepath in matchup_directory.glob(pattern)
        if filepath.is_file()
        and not filepath.name.endswith(".partial")
    )

    return files


def validate_dataset_variables(
    dataset: xr.Dataset,
    required_variables: Sequence[str] = REQUIRED_NETCDF_VARIABLES,
) -> None:
    """
    Confirm that a matchup Dataset includes all required variables.
    """

    missing = [
        variable
        for variable in required_variables
        if variable not in dataset.variables
    ]

    if missing:
        raise KeyError(
            "Dataset is missing required variables: "
            + ", ".join(missing)
        )


def _stack_matchup_dataset(
    dataset: xr.Dataset,
) -> xr.Dataset:
    """
    Stack the two-dimensional scan/pixel swath into one footprint dimension.
    """

    if "scan" not in dataset.dims or "pixel" not in dataset.dims:
        raise ValueError(
            "Expected matchup dimensions 'scan' and 'pixel'. "
            f"Found: {dict(dataset.sizes)}"
        )

    return dataset.stack(
        footprint=("scan", "pixel")
    )


def load_matchup_file(
    filepath: str | os.PathLike[str],
    *,
    retain_only_base_reference_rows: bool = True,
    retain_identifiers: bool = True,
) -> tuple[pd.DataFrame, MatchupLoadSummary]:
    """
    Load one matchup file into a compact footprint-level DataFrame.

    Only the variables required for the current analysis are read.

    When retain_only_base_reference_rows is True, rows are retained only when:
        - GPROF V7 is finite
        - GPROF V8 is finite
        - MRMS is finite
        - mrms_valid_flag == 1

    ERA5, T2M, and AutoSnow are not required at this loading stage because
    they are enforced later by analysis-specific sample masks.
    """

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(filepath)

    try:
        with xr.open_dataset(
            filepath,
            decode_cf=True,
            mask_and_scale=True,
        ) as dataset:

            validate_dataset_variables(dataset)

            selected = dataset[
                list(REQUIRED_NETCDF_VARIABLES)
            ]

            stacked = _stack_matchup_dataset(selected)

            total_rows = int(stacked.sizes["footprint"])

            if retain_only_base_reference_rows:
                base_mask = (
                    np.isfinite(
                        stacked["surfacePrecipitation_V7"]
                    )
                    & np.isfinite(
                        stacked["surfacePrecipitation_V8"]
                    )
                    & np.isfinite(
                        stacked["MRMS_Pass2"]
                    )
                    & (
                        stacked["mrms_valid_flag"] == 1
                    )
                )

                stacked = stacked.where(
                    base_mask,
                    drop=True,
                )

            # Convert to DataFrame. The stacked xarray object may already contain
            # scan and pixel as coordinate columns, so calling reset_index() directly
            # can produce duplicate-column errors such as:
            #     ValueError: cannot insert pixel, already exists
            frame = stacked.to_dataframe()

            # Preserve index coordinates only when they are not already columns.
            for index_name in frame.index.names:
                if (
                    index_name is not None
                    and index_name not in frame.columns
                ):
                    frame[index_name] = (
                        frame.index.get_level_values(index_name)
                    )

            # Discard the index after its useful coordinate values have been retained.
            frame = frame.reset_index(drop=True)

        frame = frame.rename(
            columns=NETCDF_TO_ANALYSIS_COLUMNS
        )

        # RAQI is preserved when available but is not required for the
        # present analysis workflow.
        if "RAQI" not in frame.columns:
            frame["RAQI"] = np.nan

        # Enforce row-wise validity again after conversion.
        if retain_only_base_reference_rows:
            frame = frame.dropna(
                subset=[
                    "GPROF_V7",
                    "GPROF_V8",
                    "MRMS",
                ]
            )

            frame = frame.loc[
                frame["mrms_valid_flag"] == 1
            ].copy()

        if retain_identifiers:
            orbit, date_string = (
                extract_orbit_and_date_from_filename(
                    filepath
                )
            )

            frame["source_file"] = filepath.name
            frame["orbit"] = orbit
            frame["orbit_date"] = date_string
        else:
            removable = [
                column
                for column in (
                    "scan",
                    "pixel",
                    "footprint",
                )
                if column in frame.columns
            ]

            frame = frame.drop(
                columns=removable,
                errors="ignore",
            )

        # Remove stack-generated index if present.
        frame = frame.drop(
            columns=["footprint"],
            errors="ignore",
        )

        frame = replace_infinite_with_nan(
            frame,
            columns=CONTINUOUS_COLUMNS,
        )

        frame = downcast_matchup_dataframe(frame)

        summary = MatchupLoadSummary(
            source_file=filepath.name,
            source_path=str(filepath),
            rows_total=total_rows,
            rows_retained=len(frame),
            status="PASS",
        )

        return frame, summary

    except Exception as exc:
        summary = MatchupLoadSummary(
            source_file=filepath.name,
            source_path=str(filepath),
            rows_total=0,
            rows_retained=0,
            status="FAIL",
            message=str(exc),
        )

        raise RuntimeError(
            f"Failed to load matchup file "
            f"{filepath}: {exc}"
        ) from exc


def load_matchup_archive(
    matchup_files: Iterable[str | os.PathLike[str]],
    *,
    retain_only_base_reference_rows: bool = True,
    retain_identifiers: bool = True,
    skip_failed_files: bool = True,
    show_progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load multiple matchup files and concatenate retained footprints.

    Returns
    -------
    matchup_dataframe
        Compact footprint-level DataFrame.

    load_summary
        One row per input file, including loading status and retained counts.
    """

    matchup_files = [
        Path(filepath)
        for filepath in matchup_files
    ]

    if not matchup_files:
        raise ValueError(
            "No matchup files were supplied."
        )

    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []

    number_of_files = len(matchup_files)

    for file_number, filepath in enumerate(
        matchup_files,
        start=1,
    ):
        try:
            frame, summary = load_matchup_file(
                filepath,
                retain_only_base_reference_rows=(
                    retain_only_base_reference_rows
                ),
                retain_identifiers=retain_identifiers,
            )

            frames.append(frame)
            summaries.append(summary.__dict__)

            if show_progress:
                print(
                    f"[{file_number:>4}/{number_of_files}] "
                    f"{filepath.name}: "
                    f"{summary.rows_retained:,} retained "
                    f"of {summary.rows_total:,} footprints"
                )

        except Exception as exc:
            summaries.append(
                MatchupLoadSummary(
                    source_file=filepath.name,
                    source_path=str(filepath),
                    rows_total=0,
                    rows_retained=0,
                    status="FAIL",
                    message=str(exc),
                ).__dict__
            )

            if not skip_failed_files:
                raise

            warnings.warn(
                f"Skipping {filepath.name}: {exc}",
                RuntimeWarning,
            )

    if not frames:
        raise RuntimeError(
            "No matchup files were loaded successfully."
        )

    matchup_dataframe = pd.concat(
        frames,
        ignore_index=True,
        copy=False,
    )

    matchup_dataframe = (
        downcast_matchup_dataframe(
            matchup_dataframe
        )
    )

    summary_dataframe = pd.DataFrame(summaries)

    if show_progress:
        print()
        print(
            f"Loaded {len(frames):,} of "
            f"{number_of_files:,} files."
        )
        print(
            f"Retained footprints: "
            f"{len(matchup_dataframe):,}"
        )
        print(
            f"DataFrame memory: "
            f"{dataframe_memory_mib(matchup_dataframe):,.1f} MiB"
        )

    return matchup_dataframe, summary_dataframe


def save_matchup_cache(
    data: pd.DataFrame,
    filepath: str | os.PathLike[str],
) -> Path:
    """
    Save a compact matchup DataFrame as a Parquet file.
    """

    filepath = Path(filepath)
    filepath.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_parquet(
        filepath,
        index=False,
    )

    return filepath


def load_matchup_cache(
    filepath: str | os.PathLike[str],
) -> pd.DataFrame:
    """
    Load a previously saved Parquet matchup cache.
    """

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(filepath)

    data = pd.read_parquet(filepath)

    return downcast_matchup_dataframe(data)


# =============================================================================
# Common-sample selection
# =============================================================================

def build_analysis_mask(
    data: pd.DataFrame,
    *,
    raqi_threshold: float | None = None,
    require_era5: bool = False,
    require_t2m: bool = False,
    require_autosnow: bool = False,
    autosnow_classes: Sequence[int] | None = None,
    mrms_wet_threshold: float | None = None,
    product_wet_column: str | None = None,
    product_wet_threshold: float | None = None,
) -> pd.Series:
    """
    Construct an explicit analysis-specific common-valid mask.

    The base mask always requires:
        finite GPROF V7
        finite GPROF V8
        finite MRMS
        mrms_valid_flag == 1
    """

    ensure_columns(
        data,
        BASE_REFERENCE_COLUMNS,
    )

    mask = (
        np.isfinite(data["GPROF_V7"])
        & np.isfinite(data["GPROF_V8"])
        & np.isfinite(data["MRMS"])
        & (data["mrms_valid_flag"] == 1)
    )

    if raqi_threshold is not None:
        ensure_columns(data, ["RAQI"])
        mask &= (
            np.isfinite(data["RAQI"])
            & (data["RAQI"] >= raqi_threshold)
        )

    if require_era5:
        ensure_columns(data, ["ERA5"])
        mask &= np.isfinite(data["ERA5"])

    if require_t2m:
        ensure_columns(data, ["T2M"])
        mask &= np.isfinite(data["T2M"])

    if require_autosnow:
        ensure_columns(data, ["AutoSnow"])
        mask &= data["AutoSnow"].isin(
            AUTOSNOW_CLASS_NAMES.keys()
        )

    if autosnow_classes is not None:
        ensure_columns(data, ["AutoSnow"])
        mask &= data["AutoSnow"].isin(
            list(autosnow_classes)
        )

    if mrms_wet_threshold is not None:
        mask &= data["MRMS"] >= mrms_wet_threshold

    if product_wet_column is not None:
        if product_wet_threshold is None:
            raise ValueError(
                "product_wet_threshold must be supplied "
                "when product_wet_column is used."
            )

        ensure_columns(
            data,
            [product_wet_column],
        )

        mask &= (
            data[product_wet_column]
            >= product_wet_threshold
        )

    return mask


def add_imerg_land_mask_to_matchups(
    data: pd.DataFrame,
    *,
    land_sea_mask_path: str | Path,
    mask_variable: str = "landseamask",
    longitude_column: str = "longitude",
    latitude_column: str = "latitude",
    output_column: str = "land_mask",
    land_threshold: float = 25.0,
) -> pd.DataFrame:
    """
    Sample the static IMERG land-sea mask at matchup locations.

    IMERG mask convention used in this project
    ------------------------------------------
    Native mask value < 25:
        land

    Native mask value >= 25:
        water/ocean

    Output convention
    -----------------
    1 = land
    0 = water/ocean

    Parameters
    ----------
    data
        Matchup dataframe containing longitude and latitude columns.

    land_sea_mask_path
        Path to GPM_IMERG_LandSeaMask.2.nc4.

    mask_variable
        Variable name in the land-sea-mask file.

    longitude_column, latitude_column
        Matchup coordinate-column names.

    output_column
        Name of the binary output land-mask column.

    land_threshold
        Native IMERG mask values below this threshold are land.

    Returns
    -------
    pandas.DataFrame
        Copy of the matchup dataframe with the binary land-mask column.
    """

    ensure_columns(
        data,
        [
            longitude_column,
            latitude_column,
        ],
    )

    output = data.copy()

    with xr.open_dataset(
        land_sea_mask_path
    ) as land_sea_dataset:
        if mask_variable not in land_sea_dataset:
            raise KeyError(
                f"{mask_variable!r} is not present in "
                f"{land_sea_mask_path}."
            )

        source_mask = (
            land_sea_dataset[mask_variable]
            .squeeze(drop=True)
            .load()
        )

    rename_mapping = {}

    if "longitude" in source_mask.dims:
        rename_mapping["longitude"] = "lon"

    if "latitude" in source_mask.dims:
        rename_mapping["latitude"] = "lat"

    if rename_mapping:
        source_mask = source_mask.rename(
            rename_mapping
        )

    if "lon" not in source_mask.dims:
        raise ValueError(
            "The IMERG land-sea mask does not contain a "
            "'lon' dimension."
        )

    if "lat" not in source_mask.dims:
        raise ValueError(
            "The IMERG land-sea mask does not contain a "
            "'lat' dimension."
        )

    source_mask = source_mask.transpose(
        "lat",
        "lon",
    )

    source_mask = source_mask.sortby(
        "lat"
    )

    source_mask = source_mask.sortby(
        "lon"
    )

    matchup_longitude = xr.DataArray(
        output[longitude_column].to_numpy(
            dtype=float
        ),
        dims="matchup",
    )

    matchup_latitude = xr.DataArray(
        output[latitude_column].to_numpy(
            dtype=float
        ),
        dims="matchup",
    )

    sampled_native_mask = source_mask.sel(
        lon=matchup_longitude,
        lat=matchup_latitude,
        method="nearest",
    ).to_numpy()

    output[output_column] = np.where(
        np.isfinite(sampled_native_mask)
        & (
            sampled_native_mask
            < land_threshold
        ),
        1,
        0,
    ).astype("uint8")

    return output


def select_analysis_sample(
    data: pd.DataFrame,
    **mask_kwargs,
) -> pd.DataFrame:
    """
    Return a copy of an analysis-specific common-valid sample.
    """

    mask = build_analysis_mask(
        data,
        **mask_kwargs,
    )

    return data.loc[mask].copy()


def summarize_sample_counts(
    data: pd.DataFrame,
    *,
    raqi_thresholds: Sequence[float | None] = (
        None,
        0.5,
        0.8,
    ),
) -> pd.DataFrame:
    """
    Summarize common-valid sample counts by RAQI threshold.
    """

    records: list[dict[str, object]] = []

    for threshold in raqi_thresholds:
        mask = build_analysis_mask(
            data,
            raqi_threshold=threshold,
        )

        label = (
            "all finite RAQI"
            if threshold is None
            else f"RAQI >= {threshold:g}"
        )

        records.append(
            {
                "sample": label,
                "raqi_threshold": threshold,
                "sample_count": int(mask.sum()),
            }
        )

    return pd.DataFrame(records)


# =============================================================================
# Surface-class analyses
# =============================================================================

def add_autosnow_labels(
    data: pd.DataFrame,
    *,
    source_column: str = "AutoSnow",
    output_column: str = "surface_class",
) -> pd.DataFrame:
    """
    Add readable AutoSnow surface-class labels.
    """

    ensure_columns(
        data,
        [source_column],
    )

    result = data.copy()

    result[output_column] = (
        result[source_column]
        .map(AUTOSNOW_CLASS_NAMES)
        .astype("category")
    )

    return result


def calculate_surface_group_means(
    data: pd.DataFrame,
    *,
    product_columns: Sequence[str] = DEFAULT_PRODUCT_COLUMNS,
    include_all_surfaces: bool = True,
    minimum_sample_count: int = 1,
) -> pd.DataFrame:
    """
    Calculate mean precipitation by AutoSnow class.

    Means are calculated from rows already selected using an appropriate
    common-valid mask.
    """

    ensure_columns(
        data,
        ["AutoSnow", *product_columns],
    )

    records: list[dict[str, object]] = []

    if include_all_surfaces:
        valid_all = data[
            ["AutoSnow", *product_columns]
        ].dropna(
            subset=list(product_columns)
        )

        if len(valid_all) >= minimum_sample_count:
            record: dict[str, object] = {
                "surface_code": -1,
                "surface_group": "all surfaces",
                "sample_count": len(valid_all),
            }

            for product in product_columns:
                record[product] = (
                    valid_all[product].mean()
                )

            records.append(record)

    for class_code, class_name in (
        AUTOSNOW_CLASS_NAMES.items()
    ):
        subset = data.loc[
            data["AutoSnow"] == class_code,
            list(product_columns),
        ].dropna(
            subset=list(product_columns)
        )

        if len(subset) < minimum_sample_count:
            continue

        record = {
            "surface_code": class_code,
            "surface_group": (
                f"AutoSnow {class_code}: "
                f"{class_name}"
            ),
            "sample_count": len(subset),
        }

        for product in product_columns:
            record[product] = subset[
                product
            ].mean()

        records.append(record)

    result = pd.DataFrame(records)

    if not result.empty:
        result["V8_minus_V7"] = (
            result["GPROF_V8"]
            - result["GPROF_V7"]
        )

        result["V8_percent_change_from_V7"] = np.where(
            result["GPROF_V7"] != 0,
            (
                100.0
                * result["V8_minus_V7"]
                / result["GPROF_V7"]
            ),
            np.nan,
        )

    return result


# =============================================================================
# Temperature analyses
# =============================================================================

def add_temperature_bins(
    data: pd.DataFrame,
    *,
    temperature_column: str = "T2M",
    bin_edges: Sequence[float] = DEFAULT_TEMPERATURE_BIN_EDGES,
    bin_labels: Sequence[str] = DEFAULT_TEMPERATURE_BIN_LABELS,
    output_column: str = "temperature_bin",
) -> pd.DataFrame:
    """
    Add categorical temperature bins.
    """

    ensure_columns(
        data,
        [temperature_column],
    )

    if len(bin_labels) != len(bin_edges) - 1:
        raise ValueError(
            "Number of temperature labels must equal "
            "number of bin intervals."
        )

    result = data.copy()

    result[output_column] = pd.cut(
        result[temperature_column],
        bins=np.asarray(bin_edges, dtype=float),
        labels=list(bin_labels),
        include_lowest=True,
        right=False,
        ordered=True,
    )

    return result


def calculate_temperature_binned_means(
    data: pd.DataFrame,
    *,
    product_columns: Sequence[str] = DEFAULT_PRODUCT_COLUMNS,
    temperature_column: str = "T2M",
    bin_edges: Sequence[float] = DEFAULT_TEMPERATURE_BIN_EDGES,
    bin_labels: Sequence[str] = DEFAULT_TEMPERATURE_BIN_LABELS,
    minimum_sample_count: int = 1,
) -> pd.DataFrame:
    """
    Calculate product means by predefined temperature bins.
    """

    ensure_columns(
        data,
        [temperature_column, *product_columns],
    )

    working = data[
        [temperature_column, *product_columns]
    ].copy()

    working = replace_infinite_with_nan(
        working,
        columns=[
            temperature_column,
            *product_columns,
        ],
    )

    working = working.dropna(
        subset=[
            temperature_column,
            *product_columns,
        ]
    )

    working = add_temperature_bins(
        working,
        temperature_column=temperature_column,
        bin_edges=bin_edges,
        bin_labels=bin_labels,
    )

    grouped = working.groupby(
        "temperature_bin",
        observed=False,
        sort=False,
    )

    means = grouped[
        list(product_columns)
    ].mean()

    counts = grouped.size().rename(
        "sample_count"
    )

    result = (
        means
        .join(counts)
        .reset_index()
    )

    result.loc[
        result["sample_count"] < minimum_sample_count,
        list(product_columns),
    ] = np.nan

    return result


def calculate_dynamic_temperature_binned_means(
    data: pd.DataFrame,
    *,
    product_columns: Sequence[str] = DEFAULT_PRODUCT_COLUMNS,
    temperature_column: str = "T2M",
    bin_width: float = 1.0,
    temperature_range: tuple[float, float] | None = None,
    minimum_sample_count: int = 1,
) -> pd.DataFrame:
    """
    Calculate product means using equal-width temperature bins.

    This is the improved form of the original
    do_intensity_level_precip_temp_analysis function.
    """

    if bin_width <= 0:
        raise ValueError(
            "bin_width must be positive."
        )

    ensure_columns(
        data,
        [temperature_column, *product_columns],
    )

    working = data[
        [temperature_column, *product_columns]
    ].copy()

    working = replace_infinite_with_nan(
        working
    )

    working = working.dropna(
        subset=[
            temperature_column,
            *product_columns,
        ]
    )

    if working.empty:
        return pd.DataFrame(
            columns=[
                "temperature_bin",
                "temperature_midpoint",
                "sample_count",
                *product_columns,
            ]
        )

    if temperature_range is None:
        lower = float(
            np.floor(
                working[temperature_column].min()
            )
        )
        upper = float(
            np.ceil(
                working[temperature_column].max()
            )
        )
    else:
        lower, upper = map(
            float,
            temperature_range,
        )

    if upper <= lower:
        raise ValueError(
            "temperature_range upper limit must "
            "exceed lower limit."
        )

    bin_edges = np.arange(
        lower,
        upper + bin_width,
        bin_width,
        dtype=float,
    )

    if bin_edges[-1] < upper:
        bin_edges = np.append(
            bin_edges,
            upper,
        )

    working["temperature_bin"] = pd.cut(
        working[temperature_column],
        bins=bin_edges,
        include_lowest=True,
        right=False,
        ordered=True,
    )

    grouped = working.groupby(
        "temperature_bin",
        observed=True,
        sort=True,
    )

    result = (
        grouped[list(product_columns)]
        .mean()
        .join(
            grouped.size().rename(
                "sample_count"
            )
        )
        .reset_index()
    )

    result["temperature_midpoint"] = result[
        "temperature_bin"
    ].map(
        lambda interval: (
            float(interval.mid)
            if pd.notna(interval)
            else np.nan
        )
    )

    result.loc[
        result["sample_count"] < minimum_sample_count,
        list(product_columns),
    ] = np.nan

    return result


# Backward-compatible alias for your original function name.
def do_intensity_level_precip_temp_analysis(
    data: pd.DataFrame,
    region: str | None = None,
    *,
    product_columns: Sequence[str] = DEFAULT_PRODUCT_COLUMNS,
    bin_width: float = 1.0,
    temperature_range: tuple[float, float] | None = None,
    minimum_sample_count: int = 1,
) -> pd.DataFrame:
    """
    Backward-compatible wrapper for dynamic temperature-bin means.

    The 'region' argument is retained for compatibility but is not used
    internally.
    """

    _ = region

    return calculate_dynamic_temperature_binned_means(
        data,
        product_columns=product_columns,
        bin_width=bin_width,
        temperature_range=temperature_range,
        minimum_sample_count=minimum_sample_count,
    )


# =============================================================================
# Continuous metrics
# =============================================================================

def pearson_correlation(
    reference: np.ndarray,
    product: np.ndarray,
) -> float:
    """
    Calculate Pearson correlation using paired finite values.
    """

    reference = np.asarray(
        reference,
        dtype=float,
    )

    product = np.asarray(
        product,
        dtype=float,
    )

    valid = (
        np.isfinite(reference)
        & np.isfinite(product)
    )

    reference = reference[valid]
    product = product[valid]

    if reference.size < 2:
        return np.nan

    if (
        np.nanstd(reference) == 0
        or np.nanstd(product) == 0
    ):
        return np.nan

    return float(
        np.corrcoef(
            reference,
            product,
        )[0, 1]
    )


def calculate_continuous_metrics(
    data: pd.DataFrame,
    reference_column: str,
    product_column: str,
) -> dict[str, float | int]:
    """
    Calculate continuous product-versus-reference statistics.

    Positive mean bias means the product exceeds the reference.
    """

    ensure_columns(
        data,
        [reference_column, product_column],
    )

    subset = (
        data[
            [reference_column, product_column]
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    sample_count = len(subset)

    if sample_count == 0:
        return {
            "reference": reference_column,
            "product": product_column,
            "N": 0,
            "reference_mean": np.nan,
            "product_mean": np.nan,
            "mean_bias": np.nan,
            "relative_bias_percent": np.nan,
            "MAE": np.nan,
            "RMSE": np.nan,
            "correlation": np.nan,
            "reference_std": np.nan,
            "product_std": np.nan,
            "normalized_std": np.nan,
            "centered_RMSE": np.nan,
        }

    reference = subset[
        reference_column
    ].to_numpy(dtype=float)

    product = subset[
        product_column
    ].to_numpy(dtype=float)

    residual = product - reference

    reference_mean = float(
        np.mean(reference)
    )

    product_mean = float(
        np.mean(product)
    )

    mean_bias = float(
        np.mean(residual)
    )

    relative_bias_percent = (
        100.0 * mean_bias / reference_mean
        if not np.isclose(reference_mean, 0.0)
        else np.nan
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

    reference_std = float(
        np.std(
            reference,
            ddof=0,
        )
    )

    product_std = float(
        np.std(
            product,
            ddof=0,
        )
    )

    normalized_std = (
        product_std / reference_std
        if not np.isclose(reference_std, 0.0)
        else np.nan
    )

    centered_residual = (
        (
            product
            - product_mean
        )
        -
        (
            reference
            - reference_mean
        )
    )

    centered_rmse = float(
        np.sqrt(
            np.mean(
                centered_residual ** 2
            )
        )
    )

    correlation = pearson_correlation(
        reference,
        product,
    )

    return {
        "reference": reference_column,
        "product": product_column,
        "N": int(sample_count),
        "reference_mean": reference_mean,
        "product_mean": product_mean,
        "mean_bias": mean_bias,
        "relative_bias_percent": relative_bias_percent,
        "MAE": mae,
        "RMSE": rmse,
        "correlation": correlation,
        "reference_std": reference_std,
        "product_std": product_std,
        "normalized_std": normalized_std,
        "centered_RMSE": centered_rmse,
    }


def calculate_metrics(
    data: pd.DataFrame,
    xcol: str,
    ycol: str,
) -> dict[str, float | int]:
    """
    Backward-compatible wrapper around calculate_continuous_metrics.

    xcol is treated as the reference and ycol as the product.
    """

    metrics = calculate_continuous_metrics(
        data,
        reference_column=xcol,
        product_column=ycol,
    )

    return {
        "N": metrics["N"],
        "Bias": metrics[
            "relative_bias_percent"
        ],
        "MeanBias": metrics["mean_bias"],
        "RMSE": metrics["RMSE"],
        "CC": metrics["correlation"],
        "MAE": metrics["MAE"],
    }


def calculate_product_metric_table(
    data: pd.DataFrame,
    *,
    reference_column: str = "MRMS",
    product_columns: Sequence[str] = (
        "GPROF_V7",
        "GPROF_V8",
        "ERA5",
    ),
) -> pd.DataFrame:
    """
    Calculate a continuous-metric table for multiple products.
    """

    records = [
        calculate_continuous_metrics(
            data,
            reference_column=reference_column,
            product_column=product,
        )
        for product in product_columns
    ]

    return pd.DataFrame(records)


# =============================================================================
# Categorical metrics
# =============================================================================

def calculate_contingency_counts(
    reference: np.ndarray,
    product: np.ndarray,
    *,
    threshold: float,
) -> dict[str, int]:
    """
    Calculate binary precipitation contingency counts.

    A value is classified as wet when value >= threshold.
    """

    reference = np.asarray(
        reference,
        dtype=float,
    )

    product = np.asarray(
        product,
        dtype=float,
    )

    valid = (
        np.isfinite(reference)
        & np.isfinite(product)
    )

    reference_wet = (
        reference[valid] >= threshold
    )

    product_wet = (
        product[valid] >= threshold
    )

    hits = int(
        np.sum(
            reference_wet
            & product_wet
        )
    )

    misses = int(
        np.sum(
            reference_wet
            & ~product_wet
        )
    )

    false_alarms = int(
        np.sum(
            ~reference_wet
            & product_wet
        )
    )

    correct_negatives = int(
        np.sum(
            ~reference_wet
            & ~product_wet
        )
    )

    return {
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_negatives": correct_negatives,
        "N": int(valid.sum()),
    }


def _safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    """
    Divide safely, returning NaN for a zero denominator.
    """

    if denominator == 0:
        return np.nan

    return float(numerator / denominator)


def calculate_categorical_metrics(
    data: pd.DataFrame,
    *,
    reference_column: str,
    product_column: str,
    threshold: float,
) -> dict[str, float | int | str]:
    """
    Calculate precipitation detection statistics.

    Metrics:
        POD
        FAR
        Success Ratio
        CSI
        Frequency Bias
        ETS
    """

    ensure_columns(
        data,
        [reference_column, product_column],
    )

    counts = calculate_contingency_counts(
        data[reference_column].to_numpy(),
        data[product_column].to_numpy(),
        threshold=threshold,
    )

    hits = counts["hits"]
    misses = counts["misses"]
    false_alarms = counts["false_alarms"]
    correct_negatives = counts[
        "correct_negatives"
    ]
    total = counts["N"]

    pod = _safe_ratio(
        hits,
        hits + misses,
    )

    far = _safe_ratio(
        false_alarms,
        hits + false_alarms,
    )

    success_ratio = _safe_ratio(
        hits,
        hits + false_alarms,
    )

    csi = _safe_ratio(
        hits,
        hits + misses + false_alarms,
    )

    frequency_bias = _safe_ratio(
        hits + false_alarms,
        hits + misses,
    )

    if total > 0:
        random_hits = (
            (hits + misses)
            * (hits + false_alarms)
            / total
        )
    else:
        random_hits = np.nan

    ets = (
        _safe_ratio(
            hits - random_hits,
            (
                hits
                + misses
                + false_alarms
                - random_hits
            ),
        )
        if np.isfinite(random_hits)
        else np.nan
    )

    return {
        "reference": reference_column,
        "product": product_column,
        "threshold": float(threshold),
        **counts,
        "POD": pod,
        "FAR": far,
        "success_ratio": success_ratio,
        "CSI": csi,
        "frequency_bias": frequency_bias,
        "ETS": ets,
    }


def calculate_product_categorical_table(
    data: pd.DataFrame,
    *,
    reference_column: str = "MRMS",
    product_columns: Sequence[str] = (
        "GPROF_V7",
        "GPROF_V8",
        "ERA5",
    ),
    threshold: float = 0.1,
) -> pd.DataFrame:
    """
    Calculate categorical metrics for multiple products.
    """

    records = [
        calculate_categorical_metrics(
            data,
            reference_column=reference_column,
            product_column=product,
            threshold=threshold,
        )
        for product in product_columns
    ]

    return pd.DataFrame(records)


# =============================================================================
# Precipitation PDFs
# =============================================================================

def _format_bin_value(
    value: float,
) -> str:
    """
    Format precipitation-bin limits for labels.
    """

    if np.isneginf(value):
        return "-inf"

    if np.isposinf(value):
        return "inf"

    return f"{value:g}"


def precipitation_bin_labels(
    bin_edges: Sequence[float],
) -> list[str]:
    """
    Create readable labels for conventional histogram bins.
    """

    edges = np.asarray(
        bin_edges,
        dtype=float,
    )

    labels: list[str] = []

    for left, right in zip(
        edges[:-1],
        edges[1:],
    ):
        if np.isposinf(right):
            labels.append(
                f">= {_format_bin_value(left)}"
            )
        else:
            labels.append(
                f"{_format_bin_value(left)}–"
                f"{_format_bin_value(right)}"
            )

    return labels


def calculate_precipitation_pdf(
    values: Sequence[float] | np.ndarray | pd.Series,
    *,
    positive_bin_edges: Sequence[float] = DEFAULT_PRECIPITATION_BIN_EDGES,
    include_exact_zero: bool = True,
) -> pd.DataFrame:
    """
    Calculate occurrence and precipitation-volume distributions.

    Exact zero is treated as a separate category when include_exact_zero=True.
    Positive precipitation is categorized using positive_bin_edges.

    The first positive edge should be zero.
    The final edge should normally be infinity.
    """

    values_array = np.asarray(
        values,
        dtype=float,
    )

    values_array = values_array[
        np.isfinite(values_array)
    ]

    if values_array.size == 0:
        return pd.DataFrame(
            columns=[
                "bin_index",
                "bin_label",
                "bin_left",
                "bin_right",
                "count",
                "occurrence_percent",
                "precipitation_volume",
                "volume_percent",
            ]
        )

    edges = np.asarray(
        positive_bin_edges,
        dtype=float,
    )

    if edges.ndim != 1 or edges.size < 2:
        raise ValueError(
            "positive_bin_edges must contain at "
            "least two ordered values."
        )

    if not np.all(
        np.diff(edges) > 0
    ):
        raise ValueError(
            "positive_bin_edges must be strictly increasing."
        )

    if edges[0] != 0.0:
        raise ValueError(
            "The first positive precipitation edge "
            "must be 0.0."
        )

    negative_count = int(
        np.sum(values_array < 0)
    )

    if negative_count > 0:
        warnings.warn(
            f"{negative_count:,} negative precipitation "
            "values were excluded from PDF calculation.",
            RuntimeWarning,
        )

    values_array = values_array[
        values_array >= 0
    ]

    total_count = values_array.size

    records: list[dict[str, float | int | str]] = []

    bin_index = 0

    if include_exact_zero:
        zero_mask = values_array == 0.0
        zero_count = int(zero_mask.sum())

        records.append(
            {
                "bin_index": bin_index,
                "bin_label": "exact zero",
                "bin_left": 0.0,
                "bin_right": 0.0,
                "count": zero_count,
                "occurrence_percent": (
                    100.0
                    * zero_count
                    / total_count
                ),
                "precipitation_volume": 0.0,
            }
        )

        bin_index += 1

    positive_values = values_array[
        values_array > 0.0
    ]

    # np.histogram excludes the rightmost infinity issue safely.
    counts, histogram_edges = np.histogram(
        positive_values,
        bins=edges,
    )

    weighted_volume, _ = np.histogram(
        positive_values,
        bins=edges,
        weights=positive_values,
    )

    labels = precipitation_bin_labels(edges)

    for index, (
        left,
        right,
        count,
        volume,
        label,
    ) in enumerate(
        zip(
            histogram_edges[:-1],
            histogram_edges[1:],
            counts,
            weighted_volume,
            labels,
        )
    ):
        records.append(
            {
                "bin_index": bin_index + index,
                "bin_label": label,
                "bin_left": float(left),
                "bin_right": float(right),
                "count": int(count),
                "occurrence_percent": (
                    100.0
                    * count
                    / total_count
                ),
                "precipitation_volume": float(volume),
            }
        )

    result = pd.DataFrame(records)

    total_volume = result[
        "precipitation_volume"
    ].sum()

    if total_volume > 0:
        result["volume_percent"] = (
            100.0
            * result["precipitation_volume"]
            / total_volume
        )
    else:
        result["volume_percent"] = 0.0

    return result


def compute_pdf_elements(
    data: pd.DataFrame,
    colname: str,
    bins: Sequence[float],
) -> pd.DataFrame:
    """
    Calculate precipitation PDFs by occurrence and volume.

    Parameters
    ----------
    data
        DataFrame containing the precipitation column.

    colname
        Name of the precipitation column.

    bins
        Successive upper limits of the precipitation classes.

        For example:
            [0.5, 1, 2, 4, 8, 16, 32]

        produces:
            <=0.5
            0.5-1
            1-2
            2-4
            4-8
            8-16
            16-32

    Returns
    -------
    pandas.DataFrame
        Columns:
            bin
            bin_label
            pdfc
            pdfv
            count
    """

    if colname not in data.columns:
        raise KeyError(
            f"Column '{colname}' is absent from the DataFrame."
        )

    values = pd.to_numeric(
        data[colname],
        errors="coerce",
    )

    values = values.loc[
        np.isfinite(values)
        & (values >= 0)
    ]

    total_count = len(values)

    if total_count == 0:
        return pd.DataFrame(
            columns=[
                "bin",
                "bin_label",
                "pdfc",
                "pdfv",
                "count",
            ]
        )

    bins = np.asarray(
        bins,
        dtype=float,
    )

    if bins.ndim != 1:
        raise ValueError(
            "bins must be one-dimensional."
        )

    if not np.all(
        np.diff(bins) > 0
    ):
        raise ValueError(
            "bins must be strictly increasing."
        )

    pdfc = []
    raw_volume = []
    counts = []
    bin_labels = []

    for index, upper_limit in enumerate(bins):

        if index == 0:
            bin_values = values.loc[
                values <= upper_limit
            ]

            bin_label = (
                f"≤{upper_limit:g}"
            )

        else:
            lower_limit = bins[index - 1]

            bin_values = values.loc[
                (values > lower_limit)
                & (values <= upper_limit)
            ]

            bin_label = (
                f"{lower_limit:g}–"
                f"{upper_limit:g}"
            )

        bin_count = len(bin_values)
        bin_volume = float(
            bin_values.sum()
        )

        counts.append(
            bin_count
        )

        pdfc.append(
            100.0
            * bin_count
            / total_count
        )

        raw_volume.append(
            bin_volume
        )

        bin_labels.append(
            bin_label
        )

    total_binned_volume = float(
        np.sum(raw_volume)
    )

    if total_binned_volume > 0:
        pdfv = (
            100.0
            * np.asarray(raw_volume)
            / total_binned_volume
        )
    else:
        pdfv = np.zeros(
            len(raw_volume),
            dtype=float,
        )

    return pd.DataFrame(
        {
            "bin": bins,
            "bin_label": bin_labels,
            "pdfc": pdfc,
            "pdfv": pdfv,
            "count": counts,
        }
    )


def compute_product_pdf_bundle(
    data: pd.DataFrame,
    *,
    product_columns: Sequence[str],
    bins: Sequence[float],
) -> dict[str, pd.DataFrame]:
    """
    Compute occurrence and volume PDFs for several products using the
    same input DataFrame.
    """

    missing = [
        product
        for product in product_columns
        if product not in data.columns
    ]

    if missing:
        raise KeyError(
            "Missing product columns: "
            + ", ".join(missing)
        )

    return {
        product: compute_pdf_elements(
            data,
            colname=product,
            bins=bins,
        )
        for product in product_columns
    }
# =============================================================================
# Combined stratified summaries
# =============================================================================

def calculate_metrics_by_group(
    data: pd.DataFrame,
    *,
    group_column: str,
    reference_column: str = "MRMS",
    product_columns: Sequence[str] = (
        "GPROF_V7",
        "GPROF_V8",
        "ERA5",
    ),
    minimum_sample_count: int = 1,
) -> pd.DataFrame:
    """
    Calculate continuous metrics by a categorical grouping column.
    """

    ensure_columns(
        data,
        [
            group_column,
            reference_column,
            *product_columns,
        ],
    )

    records: list[dict[str, object]] = []

    grouped = data.groupby(
        group_column,
        observed=True,
        sort=False,
    )

    for group_value, group_data in grouped:
        if len(group_data) < minimum_sample_count:
            continue

        for product in product_columns:
            metrics = calculate_continuous_metrics(
                group_data,
                reference_column=reference_column,
                product_column=product,
            )

            metrics[group_column] = group_value
            records.append(metrics)

    return pd.DataFrame(records)


def calculate_categorical_metrics_by_group(
    data: pd.DataFrame,
    *,
    group_column: str,
    threshold: float,
    reference_column: str = "MRMS",
    product_columns: Sequence[str] = (
        "GPROF_V7",
        "GPROF_V8",
        "ERA5",
    ),
    minimum_sample_count: int = 1,
) -> pd.DataFrame:
    """
    Calculate categorical precipitation metrics by group.
    """

    ensure_columns(
        data,
        [
            group_column,
            reference_column,
            *product_columns,
        ],
    )

    records: list[dict[str, object]] = []

    grouped = data.groupby(
        group_column,
        observed=True,
        sort=False,
    )

    for group_value, group_data in grouped:
        if len(group_data) < minimum_sample_count:
            continue

        for product in product_columns:
            metrics = calculate_categorical_metrics(
                group_data,
                reference_column=reference_column,
                product_column=product,
                threshold=threshold,
            )

            metrics[group_column] = group_value
            records.append(metrics)

    return pd.DataFrame(records)

# =============================================================================
# Spatial gridding and metric analysis
# =============================================================================

def create_spatial_grid(
    *,
    extent: tuple[float, float, float, float] = (
        -125.0,
        -66.0,
        24.0,
        50.0,
    ),
    resolution: float = 0.25,
) -> dict[str, np.ndarray | float | tuple]:
    """
    Create a regular CONUS latitude-longitude grid.

    Parameters
    ----------
    extent
        West, east, south, north.

    resolution
        Grid-cell width in degrees.
    """

    west, east, south, north = map(
        float,
        extent,
    )

    if resolution <= 0:
        raise ValueError(
            "resolution must be positive."
        )

    lon_edges = np.arange(
        west,
        east + resolution,
        resolution,
        dtype=float,
    )

    lat_edges = np.arange(
        south,
        north + resolution,
        resolution,
        dtype=float,
    )

    lon_centers = (
        lon_edges[:-1]
        + lon_edges[1:]
    ) / 2.0

    lat_centers = (
        lat_edges[:-1]
        + lat_edges[1:]
    ) / 2.0

    return {
        "extent": (
            west,
            east,
            south,
            north,
        ),
        "resolution": float(resolution),
        "lon_edges": lon_edges,
        "lat_edges": lat_edges,
        "lon_centers": lon_centers,
        "lat_centers": lat_centers,
        "nlon": len(lon_centers),
        "nlat": len(lat_centers),
    }


def add_spatial_grid_indices(
    data: pd.DataFrame,
    *,
    grid: Mapping[str, object],
    latitude_column: str = "latitude",
    longitude_column: str = "longitude",
) -> pd.DataFrame:
    """
    Assign each footprint to a regular latitude-longitude grid cell.

    Rows outside the requested grid are removed.
    """

    ensure_columns(
        data,
        [
            latitude_column,
            longitude_column,
        ],
    )

    west, east, south, north = grid[
        "extent"
    ]

    resolution = float(
        grid["resolution"]
    )

    working = data.copy()

    latitude = pd.to_numeric(
        working[latitude_column],
        errors="coerce",
    )

    longitude = pd.to_numeric(
        working[longitude_column],
        errors="coerce",
    )

    # Convert 0–360 longitudes to -180–180 when needed.
    longitude = np.where(
        longitude > 180.0,
        longitude - 360.0,
        longitude,
    )

    valid = (
        np.isfinite(latitude)
        & np.isfinite(longitude)
        & (longitude >= west)
        & (longitude < east)
        & (latitude >= south)
        & (latitude < north)
    )

    working = working.loc[
        valid
    ].copy()

    latitude = np.asarray(
        latitude[valid],
        dtype=float,
    )

    longitude = np.asarray(
        longitude[valid],
        dtype=float,
    )

    working["lat_index"] = np.floor(
        (latitude - south)
        / resolution
    ).astype(np.int32)

    working["lon_index"] = np.floor(
        (longitude - west)
        / resolution
    ).astype(np.int32)

    working["latitude"] = latitude
    working["longitude"] = longitude

    return working


def _series_to_spatial_array(
    series: pd.Series,
    *,
    grid: Mapping[str, object],
    fill_value: float = np.nan,
) -> np.ndarray:
    """
    Convert a Series indexed by (lat_index, lon_index) to a 2-D array.
    """

    output = np.full(
        (
            int(grid["nlat"]),
            int(grid["nlon"]),
        ),
        fill_value,
        dtype=float,
    )

    if series.empty:
        return output

    lat_index = series.index.get_level_values(
        "lat_index"
    ).to_numpy(dtype=int)

    lon_index = series.index.get_level_values(
        "lon_index"
    ).to_numpy(dtype=int)

    output[
        lat_index,
        lon_index,
    ] = series.to_numpy(dtype=float)

    return output


def calculate_spatial_mean_fields(
    data: pd.DataFrame,
    *,
    grid: Mapping[str, object],
    product_columns: Sequence[str] = (
        "GPROF_V7",
        "GPROF_V8",
        "MRMS",
        "ERA5",
    ),
    latitude_column: str = "latitude",
    longitude_column: str = "longitude",
    minimum_sample_count: int = 20,
) -> xr.Dataset:
    """
    Calculate gridded mean precipitation from common paired footprints.
    """

    ensure_columns(
        data,
        [
            latitude_column,
            longitude_column,
            *product_columns,
        ],
    )

    working = data[
        [
            latitude_column,
            longitude_column,
            *product_columns,
        ]
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna(
        subset=[
            latitude_column,
            longitude_column,
            *product_columns,
        ]
    )

    working = add_spatial_grid_indices(
        working,
        grid=grid,
        latitude_column=latitude_column,
        longitude_column=longitude_column,
    )

    grouped = working.groupby(
        [
            "lat_index",
            "lon_index",
        ],
        observed=True,
        sort=False,
    )

    count = grouped.size().rename(
        "sample_count"
    )

    dataset = xr.Dataset(
        coords={
            "latitude": np.asarray(
                grid["lat_centers"]
            ),
            "longitude": np.asarray(
                grid["lon_centers"]
            ),
        }
    )

    count_array = _series_to_spatial_array(
        count,
        grid=grid,
        fill_value=0,
    )

    dataset["sample_count"] = (
        (
            "latitude",
            "longitude",
        ),
        count_array.astype(np.int64),
    )

    for product in product_columns:
        mean_series = grouped[
            product
        ].mean()

        mean_array = _series_to_spatial_array(
            mean_series,
            grid=grid,
        )

        mean_array[
            count_array
            < minimum_sample_count
        ] = np.nan

        dataset[
            f"{product}_mean"
        ] = (
            (
                "latitude",
                "longitude",
            ),
            mean_array,
        )

    dataset.attrs.update(
        {
            "grid_resolution_degrees": float(
                grid["resolution"]
            ),
            "minimum_sample_count": int(
                minimum_sample_count
            ),
            "spatial_extent": str(
                grid["extent"]
            ),
        }
    )

    return dataset


def calculate_spatial_continuous_metrics(
    data: pd.DataFrame,
    *,
    grid: Mapping[str, object],
    reference_column: str = "MRMS",
    product_columns: Sequence[str] = (
        "GPROF_V7",
        "GPROF_V8",
        "ERA5",
    ),
    latitude_column: str = "latitude",
    longitude_column: str = "longitude",
    minimum_sample_count: int = 1,
    minimum_reference_mean_for_relative_bias: float = 0.01,
) -> xr.Dataset:
    """
    Calculate spatial CC, relative bias, mean bias, and RMSE.

    Metrics are calculated from the paired footprint samples within each
    spatial grid cell.

    Notes
    -----
    CC remains undefined when:
        - fewer than two paired samples are available;
        - MRMS has zero variance;
        - the evaluated product has zero variance.

    Relative bias remains undefined when the grid-cell mean MRMS
    precipitation is smaller than
    minimum_reference_mean_for_relative_bias.
    """

    ensure_columns(
        data,
        [
            latitude_column,
            longitude_column,
            reference_column,
            *product_columns,
        ],
    )

    required = [
        latitude_column,
        longitude_column,
        reference_column,
        *product_columns,
    ]

    working = (
        data[required]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(
            subset=required
        )
    )

    working = add_spatial_grid_indices(
        working,
        grid=grid,
        latitude_column=latitude_column,
        longitude_column=longitude_column,
    )

    dataset = xr.Dataset(
        coords={
            "latitude": np.asarray(
                grid["lat_centers"],
                dtype=float,
            ),
            "longitude": np.asarray(
                grid["lon_centers"],
                dtype=float,
            ),
        }
    )

    group_columns = [
        "lat_index",
        "lon_index",
    ]

    for product in product_columns:

        temporary = working[
            [
                *group_columns,
                reference_column,
                product,
            ]
        ].copy()

        temporary["_reference_squared"] = (
            temporary[reference_column] ** 2
        )

        temporary["_product_squared"] = (
            temporary[product] ** 2
        )

        temporary["_cross_product"] = (
            temporary[reference_column]
            * temporary[product]
        )

        temporary["_error"] = (
            temporary[product]
            - temporary[reference_column]
        )

        temporary["_squared_error"] = (
            temporary["_error"] ** 2
        )

        summary = temporary.groupby(
            group_columns,
            observed=True,
            sort=False,
        ).agg(
            sample_count=(
                reference_column,
                "count",
            ),
            sum_reference=(
                reference_column,
                "sum",
            ),
            sum_product=(
                product,
                "sum",
            ),
            sum_reference_squared=(
                "_reference_squared",
                "sum",
            ),
            sum_product_squared=(
                "_product_squared",
                "sum",
            ),
            sum_cross_product=(
                "_cross_product",
                "sum",
            ),
            sum_error=(
                "_error",
                "sum",
            ),
            sum_squared_error=(
                "_squared_error",
                "sum",
            ),
        )

        n = summary[
            "sample_count"
        ].astype(float)

        mean_reference = (
            summary["sum_reference"]
            / n
        )

        mean_product = (
            summary["sum_product"]
            / n
        )

        mean_bias = (
            mean_product
            - mean_reference
        )

        # Population covariance and variances.
        covariance = (
            summary["sum_cross_product"]
            / n
            - mean_reference
            * mean_product
        )

        variance_reference = (
            summary["sum_reference_squared"]
            / n
            - mean_reference ** 2
        )

        variance_product = (
            summary["sum_product_squared"]
            / n
            - mean_product ** 2
        )

        # Remove tiny negative values caused by floating-point cancellation.
        variance_reference = (
            variance_reference.clip(
                lower=0.0
            )
        )

        variance_product = (
            variance_product.clip(
                lower=0.0
            )
        )

        standard_deviation_product = np.sqrt(
            variance_product
        )

        standard_deviation_reference = np.sqrt(
            variance_reference
        )

        correlation_denominator = (
            standard_deviation_reference
            * standard_deviation_product
        )

        correlation = (
            covariance
            / correlation_denominator
        )

        # Prevent tiny floating-point excursions outside [-1, 1].
        correlation = correlation.clip(
            lower=-1.0,
            upper=1.0,
        )

        correlation_invalid = (
            (n < max(
                2,
                minimum_sample_count,
            ))
            | (variance_reference <= 0.0)
            | (variance_product <= 0.0)
        )

        correlation = correlation.mask(
            correlation_invalid
        )

        # Equivalent to:
        # 100 * sum(product - reference) / sum(reference)
        relative_bias = (
            100.0
            * mean_bias
            / mean_reference
        )

        relative_bias = relative_bias.mask(
            (n < minimum_sample_count)
            | (
                mean_reference
                < minimum_reference_mean_for_relative_bias
            )
        )

        rmse = np.sqrt(
            summary["sum_squared_error"]
            / n
        )

        rmse = rmse.mask(
            n < minimum_sample_count
        )

        mean_bias = mean_bias.mask(
            n < minimum_sample_count
        )

        mean_reference = mean_reference.mask(
            n < minimum_sample_count
        )

        mean_product = mean_product.mask(
            n < minimum_sample_count
        )

        fields = {
            "CC": correlation,
            "relative_bias": relative_bias,
            "mean_bias": mean_bias,
            "RMSE": rmse,
            "reference_mean": mean_reference,
            "product_mean": mean_product,
            "sample_count": n,
        }

        for metric_name, metric_series in fields.items():

            is_count = (
                metric_name
                == "sample_count"
            )

            array = _series_to_spatial_array(
                metric_series,
                grid=grid,
                fill_value=(
                    0.0
                    if is_count
                    else np.nan
                ),
            )

            dataset[
                f"{product}_{metric_name}"
            ] = (
                (
                    "latitude",
                    "longitude",
                ),
                array,
            )

    dataset.attrs.update(
        {
            "reference_product": (
                reference_column
            ),
            "minimum_sample_count": int(
                minimum_sample_count
            ),
            "minimum_reference_mean_for_relative_bias": float(
                minimum_reference_mean_for_relative_bias
            ),
            "grid_resolution_degrees": float(
                grid["resolution"]
            ),
            "relative_bias_definition": (
                "100 * (mean_product - mean_reference) "
                "/ mean_reference"
            ),
        }
    )

    return dataset


def calculate_spatial_categorical_metrics(
    data: pd.DataFrame,
    *,
    grid: Mapping[str, object],
    reference_column: str = "MRMS",
    product_columns: Sequence[str] = (
        "GPROF_V7",
        "GPROF_V8",
        "ERA5",
    ),
    wet_threshold: float = 0.1,
    latitude_column: str = "latitude",
    longitude_column: str = "longitude",
    minimum_event_count: int = 20,
) -> xr.Dataset:
    """
    Calculate spatial POD, FAR, and CSI using MRMS as event reference.
    """

    ensure_columns(
        data,
        [
            latitude_column,
            longitude_column,
            reference_column,
            *product_columns,
        ],
    )

    required = [
        latitude_column,
        longitude_column,
        reference_column,
        *product_columns,
    ]

    working = data[
        required
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna(
        subset=required
    )

    working = add_spatial_grid_indices(
        working,
        grid=grid,
        latitude_column=latitude_column,
        longitude_column=longitude_column,
    )

    dataset = xr.Dataset(
        coords={
            "latitude": np.asarray(
                grid["lat_centers"]
            ),
            "longitude": np.asarray(
                grid["lon_centers"]
            ),
        }
    )

    reference_wet = (
        working[reference_column]
        >= wet_threshold
    )

    group_columns = [
        "lat_index",
        "lon_index",
    ]

    for product in product_columns:
        product_wet = (
            working[product]
            >= wet_threshold
        )

        temporary = working[
            group_columns
        ].copy()

        temporary["hits"] = (
            reference_wet
            & product_wet
        ).astype(np.int8)

        temporary["misses"] = (
            reference_wet
            & ~product_wet
        ).astype(np.int8)

        temporary["false_alarms"] = (
            ~reference_wet
            & product_wet
        ).astype(np.int8)

        temporary["correct_negatives"] = (
            ~reference_wet
            & ~product_wet
        ).astype(np.int8)

        counts = temporary.groupby(
            group_columns,
            observed=True,
            sort=False,
        )[
            [
                "hits",
                "misses",
                "false_alarms",
                "correct_negatives",
            ]
        ].sum()

        observed_events = (
            counts["hits"]
            + counts["misses"]
        )

        forecast_events = (
            counts["hits"]
            + counts["false_alarms"]
        )

        union_events = (
            counts["hits"]
            + counts["misses"]
            + counts["false_alarms"]
        )

        # Division is allowed only where the corresponding denominator exists.

        pod = (
            counts["hits"]
            .div(
                observed_events.where(
                    observed_events > 0
                )
            )
                            )

        far = (
            counts["false_alarms"]
            .div(
                forecast_events.where(
                    forecast_events > 0
                )
            )
        )

        csi = (
            counts["hits"]
            .div(
                union_events.where(
                    union_events > 0
                )
            )
        )

        # Optional event-count threshold. With minimum_event_count=1,
        # only mathematically undefined cells remain masked.

        pod = pod.mask(
            observed_events
            < minimum_event_count
        )

        far = far.mask(
            forecast_events
            < minimum_event_count
        )

        csi = csi.mask(
            union_events
            < minimum_event_count
        )

        fields = {
            "POD": pod,
            "FAR": far,
            "CSI": csi,
            "hits": counts["hits"],
            "misses": counts["misses"],
            "false_alarms": counts[
                "false_alarms"
            ],
            "correct_negatives": counts[
                "correct_negatives"
            ],
            "observed_event_count": observed_events,
            "forecast_event_count": forecast_events,
            "union_event_count": union_events,
        }

        for metric_name, metric_series in fields.items():
            array = _series_to_spatial_array(
                metric_series,
                grid=grid,
                fill_value=(
                    0
                    if metric_name in {
                            "hits",
                            "misses",
                            "false_alarms",
                            "correct_negatives",
                            "observed_event_count",
                            "forecast_event_count",
                            "union_event_count",
                        }
                    else np.nan
                ),
            )

            dataset[
                f"{product}_{metric_name}"
            ] = (
                (
                    "latitude",
                    "longitude",
                ),
                array,
            )

    dataset.attrs.update(
        {
            "reference_product": (
                reference_column
            ),
            "wet_threshold": float(
                wet_threshold
            ),
            "minimum_event_count": int(
                minimum_event_count
            ),
            "grid_resolution_degrees": float(
                grid["resolution"]
            ),
        }
    )

    return dataset


def calculate_spatial_mean_differences(
    spatial_mean_dataset,
):
    """
    Calculate differences among spatial mean precipitation fields.

    Required input variables
    ------------------------
    GPROF_V7_mean
    GPROF_V8_mean
    MRMS_mean
    ERA5_mean

    Returned differences
    --------------------
    GPROF_V8_minus_MRMS
        Positive values indicate GPROF V8 exceeds MRMS.

    GPROF_V7_minus_MRMS
        Positive values indicate GPROF V7 exceeds MRMS.

    ERA5_minus_MRMS
        Positive values indicate ERA5 exceeds MRMS.

    GPROF_V7_minus_GPROF_V8
        Positive values indicate GPROF V7 exceeds GPROF V8.

    Sign convention
    ---------------
    Product minus reference.

    Therefore:
        negative = product underestimates MRMS
        positive = product overestimates MRMS
    """

    import xarray as xr

    required_variables = (
        "MRMS_mean",
        "GPROF_V8_mean",
        "GPROF_V7_mean",
        "ERA5_mean",
    )

    missing_variables = [
        variable
        for variable in required_variables
        if variable not in spatial_mean_dataset
    ]

    if missing_variables:
        raise KeyError(
            "Missing required spatial mean variables: "
            + ", ".join(missing_variables)
        )

    differences = xr.Dataset(
        data_vars={
            "GPROF_V8_minus_MRMS": (
                spatial_mean_dataset["GPROF_V8_mean"]
                - spatial_mean_dataset["MRMS_mean"]
            ),
            "GPROF_V7_minus_MRMS": (
                spatial_mean_dataset["GPROF_V7_mean"]
                - spatial_mean_dataset["MRMS_mean"]
            ),
            "ERA5_minus_MRMS": (
                spatial_mean_dataset["ERA5_mean"]
                - spatial_mean_dataset["MRMS_mean"]
            ),
            "GPROF_V7_minus_GPROF_V8": (
                spatial_mean_dataset["GPROF_V7_mean"]
                - spatial_mean_dataset["GPROF_V8_mean"]
            ),
        },
        coords={
            coordinate: spatial_mean_dataset.coords[
                coordinate
            ]
            for coordinate in spatial_mean_dataset.coords
        },
        attrs={
            "description": (
                "Differences between common-sample spatial "
                "mean precipitation fields."
            ),
            "difference_sign_convention": (
                "Product minus reference. Positive values indicate "
                "the first product exceeds the second."
            ),
            "units": "mm h-1",
        },
    )

    long_names = {
        "GPROF_V8_minus_MRMS": (
            "GPROF V8 minus MRMS mean precipitation"
        ),
        "GPROF_V7_minus_MRMS": (
            "GPROF V7 minus MRMS mean precipitation"
        ),
        "ERA5_minus_MRMS": (
            "ERA5 minus MRMS mean precipitation"
        ),
        "GPROF_V7_minus_GPROF_V8": (
            "GPROF V7 minus GPROF V8 mean precipitation"
        ),
    }

    for variable, long_name in long_names.items():
        differences[
            variable
        ].attrs.update(
            {
                "long_name": long_name,
                "units": "mm h-1",
            }
        )

    return differences
# =============================================================================
# Time aggregation
# =============================================================================

def add_time_coordinates(
    data: pd.DataFrame,
    *,
    date_column: str = "orbit_date",
) -> pd.DataFrame:
    """Add datetime, month, year, and meteorological-season columns."""
    ensure_columns(data, [date_column])
    result = data.copy()
    result["date"] = pd.to_datetime(result[date_column], errors="coerce")
    result["year"] = result["date"].dt.year.astype("Int16")
    result["month"] = result["date"].dt.month.astype("Int8")
    season_map = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
    result["season"] = result["month"].map(season_map)
    result["year_month"] = result["date"].dt.to_period("M").dt.to_timestamp()
    return result


def calculate_monthly_mean_timeseries(
    data: pd.DataFrame,
    *,
    product_columns: Sequence[str] = DEFAULT_PRODUCT_COLUMNS,
    date_column: str = "orbit_date",
) -> pd.DataFrame:
    """Calculate common-sample monthly mean precipitation for all products."""
    ensure_columns(data, [date_column, *product_columns])
    working = add_time_coordinates(data, date_column=date_column)
    required = ["year_month", *product_columns]
    working = working.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    monthly = working.groupby("year_month", observed=True)[list(product_columns)].mean().reset_index()
    monthly["sample_count"] = working.groupby("year_month", observed=True).size().to_numpy()
    return monthly


def calculate_seasonal_spatial_mean_fields(
    data: pd.DataFrame,
    *,
    grid: Mapping[str, object],
    product_columns: Sequence[str] = DEFAULT_PRODUCT_COLUMNS,
    date_column: str = "orbit_date",
    latitude_column: str = "latitude",
    longitude_column: str = "longitude",
    minimum_sample_count: int = 1,
) -> xr.Dataset:
    """Calculate spatial mean precipitation separately for DJF, MAM, JJA, and SON."""
    working = add_time_coordinates(data, date_column=date_column)
    seasons = ("DJF", "MAM", "JJA", "SON")
    datasets = []
    for season in seasons:
        subset = working.loc[working["season"] == season]
        ds = calculate_spatial_mean_fields(
            subset,
            grid=grid,
            product_columns=product_columns,
            latitude_column=latitude_column,
            longitude_column=longitude_column,
            minimum_sample_count=minimum_sample_count,
        )
        datasets.append(ds.expand_dims(season=[season]))
    return xr.concat(datasets, dim="season")


def calculate_spatial_precipitation_fraction(
    data: pd.DataFrame,
    *,
    grid: Mapping[str, object],
    product_columns: Sequence[str] = DEFAULT_PRODUCT_COLUMNS,
    wet_threshold: float = 0.1,
    latitude_column: str = "latitude",
    longitude_column: str = "longitude",
    minimum_sample_count: int = 1,
) -> xr.Dataset:
    """Calculate wet-footprint fraction (%) in each spatial grid cell."""
    ensure_columns(data, [latitude_column, longitude_column, *product_columns])
    required = [latitude_column, longitude_column, *product_columns]
    working = data[required].replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    working = add_spatial_grid_indices(
        working,
        grid=grid,
        latitude_column=latitude_column,
        longitude_column=longitude_column,
    )
    group_columns = ["lat_index", "lon_index"]
    dataset = xr.Dataset(coords={"latitude": np.asarray(grid["lat_centers"]), "longitude": np.asarray(grid["lon_centers"])})
    total = working.groupby(group_columns, observed=True, sort=False).size()
    for product in product_columns:
        wet = (working[product] > wet_threshold).astype(np.int8)
        wet_count = wet.groupby([working["lat_index"], working["lon_index"]], observed=True).sum()
        fraction = 100.0 * wet_count / total
        fraction = fraction.mask(total < minimum_sample_count)
        dataset[f"{product}_precipitation_fraction"] = (("latitude", "longitude"), _series_to_spatial_array(fraction, grid=grid, fill_value=np.nan))
        dataset[f"{product}_sample_count"] = (("latitude", "longitude"), _series_to_spatial_array(total.astype(float), grid=grid, fill_value=0.0))
    dataset.attrs.update({"wet_threshold_mmhr": float(wet_threshold), "minimum_sample_count": int(minimum_sample_count), "definition": "100 * count(precipitation > wet_threshold) / total common-valid count"})
    return dataset

def calculate_spatial_precipitation_fraction_differences(
    spatial_fraction_dataset,
):
    """
    Calculate spatial precipitation-fraction differences.

    Difference convention
    ---------------------
    Product minus reference.

    Therefore:
        positive = product has a higher precipitation fraction
        negative = product has a lower precipitation fraction

    Units
    -----
    Percentage points.
    """

    import xarray as xr

    required_variables = (
        "MRMS_precipitation_fraction",
        "GPROF_V8_precipitation_fraction",
        "GPROF_V7_precipitation_fraction",
        "ERA5_precipitation_fraction",
    )

    missing_variables = [
        variable
        for variable in required_variables
        if variable not in spatial_fraction_dataset
    ]

    if missing_variables:
        raise KeyError(
            "Missing required precipitation-fraction variables: "
            + ", ".join(missing_variables)
        )

    difference_dataset = xr.Dataset(
        data_vars={
            "GPROF_V8_minus_MRMS": (
                spatial_fraction_dataset[
                    "GPROF_V8_precipitation_fraction"
                ]
                - spatial_fraction_dataset[
                    "MRMS_precipitation_fraction"
                ]
            ),
            "GPROF_V7_minus_MRMS": (
                spatial_fraction_dataset[
                    "GPROF_V7_precipitation_fraction"
                ]
                - spatial_fraction_dataset[
                    "MRMS_precipitation_fraction"
                ]
            ),
            "ERA5_minus_MRMS": (
                spatial_fraction_dataset[
                    "ERA5_precipitation_fraction"
                ]
                - spatial_fraction_dataset[
                    "MRMS_precipitation_fraction"
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
        },
        coords={
            coordinate: spatial_fraction_dataset.coords[
                coordinate
            ]
            for coordinate in spatial_fraction_dataset.coords
        },
        attrs={
            "description": (
                "Spatial precipitation-fraction differences "
                "using product-minus-reference convention."
            ),
            "units": "percentage points",
            "difference_sign_convention": (
                "Positive values indicate the first product "
                "has a higher precipitation fraction."
            ),
        },
    )

    long_names = {
        "GPROF_V8_minus_MRMS": (
            "GPROF V8 minus MRMS precipitation fraction"
        ),
        "GPROF_V7_minus_MRMS": (
            "GPROF V7 minus MRMS precipitation fraction"
        ),
        "ERA5_minus_MRMS": (
            "ERA5 minus MRMS precipitation fraction"
        ),
        "GPROF_V7_minus_GPROF_V8": (
            "GPROF V7 minus GPROF V8 precipitation fraction"
        ),
    }

    for variable, long_name in long_names.items():
        difference_dataset[
            variable
        ].attrs.update(
            {
                "long_name": long_name,
                "units": "percentage points",
            }
        )

    return difference_dataset

from pathlib import Path

import numpy as np
import xarray as xr
import rioxarray  # noqa: F401
from rasterio.enums import Resampling


def prepare_conus_land_mask(
    target_grid,
    *,
    land_sea_mask_path: str | Path = (
        "/ra1/pubdat/AVHRR_CloudSat_proj/IMERG/"
        "ancillary_imerg_data/"
        "GPM_IMERG_LandSeaMask.2.nc4"
    ),
    source_variable: str = "landseamask",
    land_threshold: float = 25.0,
):
    """
    Prepare a binary land mask on the spatial-analysis grid.

    Parameters
    ----------
    target_grid
        A two-dimensional xarray DataArray on the final analysis
        latitude-longitude grid. For example, spatial_means["MRMS_mean"].

    land_sea_mask_path
        Path to the native IMERG land-sea mask.

    source_variable
        Name of the land-sea-mask variable.

    land_threshold
        Native mask values below this threshold are classified as land,
        following the existing project convention.

    Returns
    -------
    xarray.DataArray
        Binary mask on the target grid:

        1 = land
        0 = ocean/water
    """

    # --------------------------------------------------------------
    # Read native IMERG land-sea mask
    # --------------------------------------------------------------
    with xr.open_dataset(
        land_sea_mask_path
    ) as source_dataset:
        source_mask = (
            source_dataset[source_variable]
            .squeeze(drop=True)
            .load()
        )

    # Standardize coordinate names.
    rename_mapping = {}

    if "longitude" in source_mask.dims:
        rename_mapping["longitude"] = "lon"

    if "latitude" in source_mask.dims:
        rename_mapping["latitude"] = "lat"

    if rename_mapping:
        source_mask = source_mask.rename(
            rename_mapping
        )

    source_mask = source_mask.transpose(
        "lat",
        "lon",
    )

    # Ensure latitude increases south to north.
    if source_mask["lat"][0] > source_mask["lat"][-1]:
        source_mask = source_mask.sortby(
            "lat"
        )

    # Binary convention:
    #     1 = land
    #     0 = water/ocean
    source_binary_mask = xr.where(
        source_mask < land_threshold,
        1,
        0,
    ).astype("uint8")

    source_binary_mask.name = "land_mask"

    source_binary_mask.attrs.update(
        {
            "long_name": (
                "Binary IMERG land mask"
            ),
            "flag_values": np.array(
                [0, 1],
                dtype="uint8",
            ),
            "flag_meanings": "water land",
            "land_value": 1,
            "water_value": 0,
        }
    )

    # --------------------------------------------------------------
    # Prepare source grid for rioxarray
    # --------------------------------------------------------------
    source_binary_mask = (
        source_binary_mask
        .rio.set_spatial_dims(
            x_dim="lon",
            y_dim="lat",
            inplace=False,
        )
        .rio.write_crs(
            "EPSG:4326",
            inplace=False,
        )
    )

    # --------------------------------------------------------------
    # Prepare target grid
    # --------------------------------------------------------------
    target = target_grid.squeeze(
        drop=True
    ).copy()

    target_rename_mapping = {}

    if "longitude" in target.dims:
        target_rename_mapping[
            "longitude"
        ] = "lon"

    if "latitude" in target.dims:
        target_rename_mapping[
            "latitude"
        ] = "lat"

    if target_rename_mapping:
        target = target.rename(
            target_rename_mapping
        )

    target = target.transpose(
        "lat",
        "lon",
    )

    if target["lat"][0] > target["lat"][-1]:
        target = target.sortby(
            "lat"
        )

    target = (
        target
        .rio.set_spatial_dims(
            x_dim="lon",
            y_dim="lat",
            inplace=False,
        )
        .rio.write_crs(
            "EPSG:4326",
            inplace=False,
        )
    )

    # --------------------------------------------------------------
    # Resample categorical mask using mode
    # --------------------------------------------------------------
    land_mask = source_binary_mask.rio.reproject_match(
        target,
        resampling=Resampling.mode,
        nodata=0,
    )

    # Force exact target coordinates to avoid tiny floating-point
    # coordinate differences.
    land_mask = land_mask.assign_coords(
        lon=target["lon"],
        lat=target["lat"],
    )

    land_mask = xr.where(
        land_mask >= 0.5,
        1,
        0,
    ).astype("uint8")

    land_mask.name = "land_mask"

    land_mask.attrs.update(
        {
            "long_name": (
                "Binary land mask resampled to the analysis grid"
            ),
            "resampling_method": "mode",
            "flag_values": np.array(
                [0, 1],
                dtype="uint8",
            ),
            "flag_meanings": "water land",
            "land_value": 1,
            "water_value": 0,
        }
    )

    return land_mask

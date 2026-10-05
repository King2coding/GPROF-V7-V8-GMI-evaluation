"""Manuscript-oriented analysis helpers for GPROF V7/V8, MRMS, and Stage IV.

The functions in this module deliberately keep sample selection explicit.  A
function only requires the products needed for its calculation; MRMS and Stage
IV are never blended into a synthetic reference.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from gprof_v7_v8_analysis_functions_v1 import (
    add_spatial_grid_indices,
    calculate_categorical_metrics,
    calculate_continuous_metrics,
    create_spatial_grid,
    ensure_columns,
)


def select_conus_land_footprints(
    data: pd.DataFrame, *, chunk_size: int = 1_000_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select US land-footprint centers in the contiguous-US bounding domain.

    Uses Natural Earth 10m administrative country geometry (including US
    islands); borders/coasts have cartographic, not survey-level precision.
    The reusable full-archive cache is never modified. First use may download
    Cartopy's country geometry; failure raises rather than using a rectangle.
    """
    import cartopy.io.shapereader as shapereader
    import shapely

    ensure_columns(data, ["latitude", "longitude", "land_mask"])
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    boundary_path = shapereader.natural_earth(
        resolution="10m", category="cultural", name="admin_0_countries",
    )
    geometries = [
        record.geometry for record in shapereader.Reader(boundary_path).records()
        if record.attributes.get("ADM0_A3") == "USA"
    ]
    if len(geometries) != 1:
        raise ValueError("Expected one US country geometry; refusing rectangle fallback")
    geometry = geometries[0]
    shapely.prepare(geometry)
    keep = np.zeros(len(data), dtype=bool)
    for start in range(0, len(data), chunk_size):
        stop = min(start + chunk_size, len(data))
        chunk = data.iloc[start:stop]
        latitude = chunk["latitude"].to_numpy(dtype=float)
        longitude = chunk["longitude"].to_numpy(dtype=float)
        longitude = np.where(longitude > 180, longitude - 360, longitude)
        candidate = (
            np.isfinite(latitude) & np.isfinite(longitude)
            & (latitude >= 24) & (latitude < 50)
            & (longitude >= -125) & (longitude < -66)
            & chunk["land_mask"].eq(1).to_numpy()
        )
        local = np.zeros(stop - start, dtype=bool)
        local[candidate] = shapely.contains_xy(
            geometry, longitude[candidate], latitude[candidate],
        )
        keep[start:stop] = local
    counts = pd.DataFrame({
        "population": ["before_CONUS_mask", "CONUS_land_retained", "excluded"],
        "footprint_count": [len(data), int(keep.sum()), int((~keep).sum())],
    })
    selected = data.loc[keep].copy()
    selected.attrs.update(
        geographic_selection="US land centers within contiguous-US domain",
        boundary_source="Natural Earth 10m admin_0_countries USA",
    )
    return selected, counts


# =============================================================================
# CONSTANTS AND INPUT SCHEMA
# =============================================================================

NETCDF_TO_ANALYSIS_COLUMNS = {
    "surfacePrecipitation_V7": "GPROF_V7",
    "surfacePrecipitation_V8": "GPROF_V8",
    "MRMS_Pass2": "MRMS",
    "StageIV": "StageIV",
    "ERA5_precipitation": "ERA5",
    "MERRA2_T2M": "T2M",
    "MERRA2_T2MWET": "T2MWET",
}

FOOTPRINT_VARIABLES = (
    "latitude",
    "longitude",
    "latitude_V8",
    "longitude_V8",
    "v7_v8_footprint_separation_km",
    "v7_v8_pair_valid_flag",
    "surfacePrecipitation_V7",
    "surfacePrecipitation_V8",
    "MRMS_Pass2",
    "RAQI",
    "ERA5_precipitation",
    "MERRA2_T2M",
    "MERRA2_T2MWET",
    "AutoSnow",
    "StageIV",
    "StageIV_source_cell_count",
    "land_mask",
    "phase_regime",
    "mrms_valid_flag",
    "stageiv_valid_flag",
    "valid_reference_flag",
)

SCAN_VARIABLES = (
    "scan_time",
    "v7_v8_scan_time_difference_seconds",
    "MRMS_interval_start",
    "MRMS_interval_end",
    "StageIV_interval_start",
    "StageIV_interval_end",
    "ERA5_time",
    "MERRA2_time",
)

PRECIPITATION_COLUMNS = ("GPROF_V7", "GPROF_V8", "MRMS", "StageIV", "ERA5")
PHASE_NAMES = {0: "transition", 1: "snow", 2: "rain"}
SEASON_ORDER = ("DJF", "MAM", "JJA", "SON")
REFERENCE_VALID_FLAGS = {"MRMS": "mrms_valid_flag", "StageIV": "stageiv_valid_flag"}


@dataclass(frozen=True)
class MatchupLoadRecord:
    """One-file loading result used for transparent archive QC."""

    source_file: str
    source_path: str
    rows_total: int
    rows_retained: int
    status: str
    message: str = ""


# =============================================================================
# MATCHUP LOADING AND CACHE MANAGEMENT
# =============================================================================

def discover_matchup_files(matchup_directory: str | Path) -> list[Path]:
    """Return sorted completed orbit NetCDF files from *matchup_directory*."""

    directory = Path(matchup_directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Matchup directory does not exist: {directory}")
    return sorted(path for path in directory.glob("*.nc") if path.is_file())


def _orbit_metadata(path: Path) -> tuple[str | None, str | None]:
    match = re.search(r"_(\d{6})_(\d{8})\.nc$", path.name)
    return match.groups() if match else (None, None)


def _flatten_footprint_variable(dataset: xr.Dataset, name: str) -> np.ndarray:
    values = dataset[name].values
    expected = (dataset.sizes["scan"], dataset.sizes["pixel"])
    if values.shape != expected:
        raise ValueError(f"{name} has shape {values.shape}; expected {expected}")
    return values.reshape(-1)


def _broadcast_scan_variable(dataset: xr.Dataset, name: str) -> np.ndarray:
    values = dataset[name].values
    if values.shape != (dataset.sizes["scan"],):
        raise ValueError(f"{name} must have the scan dimension only")
    return np.repeat(values, dataset.sizes["pixel"])


def load_matchup_file(
    filepath: str | Path,
    *,
    land_only: bool = True,
    extent: tuple[float, float, float, float] | None = (-125.0, -66.0, 24.0, 50.0),
    retain_identifiers: bool = True,
    maximum_pair_separation_km: float | None = 1.0,
) -> tuple[pd.DataFrame, MatchupLoadRecord]:
    """Load one native scan x pixel matchup without an all-products-valid filter."""

    path = Path(filepath)
    try:
        # Some files contain intentionally missing reference-interval times.
        # Older xarray versions decode those fill values to NaT correctly but
        # emit one RuntimeWarning per affected time variable.  Suppress only
        # that known decoder warning; all other warnings remain visible.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in cast",
                category=RuntimeWarning,
                module=r"xarray\.coding\.times",
            )
            with xr.open_dataset(
                path,
                decode_cf=True,
                decode_timedelta=False,
                mask_and_scale=True,
            ) as dataset:
                if "scan" not in dataset.sizes or "pixel" not in dataset.sizes:
                    raise ValueError("Expected native 'scan' and 'pixel' dimensions")
                missing = [name for name in FOOTPRINT_VARIABLES if name not in dataset]
                if missing:
                    raise KeyError("Missing required variables: " + ", ".join(missing))

                frame_data = {
                    NETCDF_TO_ANALYSIS_COLUMNS.get(name, name): _flatten_footprint_variable(dataset, name)
                    for name in FOOTPRINT_VARIABLES
                }
                for name in SCAN_VARIABLES:
                    if name in dataset:
                        frame_data[name] = _broadcast_scan_variable(dataset, name)
                frame = pd.DataFrame(frame_data)
                rows_total = len(frame)

        numeric_columns = [
            column for column in frame.columns
            if column not in {"scan_time", *SCAN_VARIABLES}
        ]
        frame[numeric_columns] = frame[numeric_columns].replace([np.inf, -np.inf], np.nan)
        if maximum_pair_separation_km is not None:
            if maximum_pair_separation_km <= 0:
                raise ValueError("maximum_pair_separation_km must be positive or None")
            separation = pd.to_numeric(
                frame["v7_v8_footprint_separation_km"], errors="coerce"
            )
            frame = frame.loc[
                np.isfinite(separation) & separation.le(maximum_pair_separation_km)
            ].copy()
        if extent is not None:
            west, east, south, north = map(float, extent)
            longitude = pd.to_numeric(frame["longitude"], errors="coerce")
            longitude = longitude.where(longitude <= 180.0, longitude - 360.0)
            frame["longitude"] = longitude
            frame = frame.loc[
                longitude.ge(west)
                & longitude.lt(east)
                & frame["latitude"].ge(south)
                & frame["latitude"].lt(north)
            ].copy()
        if land_only:
            frame = frame.loc[frame["land_mask"] == 1].copy()

        if retain_identifiers:
            orbit, orbit_date = _orbit_metadata(path)
            frame["source_file"] = path.name
            frame["orbit"] = orbit
            frame["orbit_date"] = orbit_date

        for column in PRECIPITATION_COLUMNS + ("RAQI", "T2M", "T2MWET"):
            if column in frame:
                frame[column] = pd.to_numeric(frame[column], errors="coerce", downcast="float")
        for column in ("land_mask", "phase_regime", "v7_v8_pair_valid_flag", "mrms_valid_flag", "stageiv_valid_flag"):
            if column in frame:
                frame[column] = pd.to_numeric(frame[column], errors="coerce", downcast="unsigned")

        record = MatchupLoadRecord(path.name, str(path), rows_total, len(frame), "PASS")
        return frame.reset_index(drop=True), record
    except Exception as exc:
        raise RuntimeError(f"Failed to load {path}: {exc}") from exc


def load_matchup_archive(
    matchup_files: Iterable[str | Path],
    *,
    land_only: bool = True,
    extent: tuple[float, float, float, float] | None = (-125.0, -66.0, 24.0, 50.0),
    retain_identifiers: bool = True,
    maximum_pair_separation_km: float | None = 1.0,
    skip_failed_files: bool = False,
    show_progress: bool = True,
    progress_interval: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load orbit files sequentially; fail closed if any file is unreadable."""

    paths = [Path(path) for path in matchup_files]
    if not paths:
        raise ValueError("No matchup files were supplied")
    if progress_interval <= 0:
        raise ValueError("progress_interval must be positive")
    frames: list[pd.DataFrame] = []
    records: list[dict[str, object]] = []
    for number, path in enumerate(paths, start=1):
        try:
            frame, record = load_matchup_file(
                path,
                land_only=land_only,
                extent=extent,
                retain_identifiers=retain_identifiers,
                maximum_pair_separation_km=maximum_pair_separation_km,
            )
            frames.append(frame)
            records.append(asdict(record))
            if show_progress and (
                number % progress_interval == 0
                or number == len(paths)
            ):
                print(
                    f"[{number:>4}/{len(paths)}] files processed; "
                    f"latest={path.name}; retained={len(frame):,} rows"
                )
        except Exception as exc:
            records.append(asdict(MatchupLoadRecord(path.name, str(path), 0, 0, "FAIL", str(exc))))
            if not skip_failed_files:
                raise
            warnings.warn(str(exc), RuntimeWarning)
    if not frames:
        raise RuntimeError("No matchup files loaded successfully")
    return pd.concat(frames, ignore_index=True, copy=False), pd.DataFrame(records)


def save_footprint_cache(data: pd.DataFrame, filepath: str | Path) -> Path:
    """Save the potentially large footprint table as a pickle, as requested."""

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_pickle(path)
    return path


def load_footprint_cache(filepath: str | Path) -> pd.DataFrame:
    """Load a previously constructed footprint pickle."""

    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_pickle(path)


# =============================================================================
# SAMPLE SELECTION
# =============================================================================

def assign_phase_regime(
    data: pd.DataFrame,
    *,
    snow_twet_max_k: float = 273.15,
    rain_twet_min_k: float = 275.15,
    output_column: str = "analysis_phase",
) -> pd.DataFrame:
    """Apply the Sims–Liu/IMERG high-confidence Twet-only phase partition."""
    ensure_columns(data, ["T2MWET"])
    if snow_twet_max_k >= rain_twet_min_k:
        raise ValueError("snow_twet_max_k must be less than rain_twet_min_k")

    result = data.copy()
    twet = pd.to_numeric(result["T2MWET"], errors="coerce")
    phase = np.full(len(result), "transition", dtype=object)
    phase[twet.le(snow_twet_max_k).to_numpy()] = "snow"
    phase[twet.ge(rain_twet_min_k).to_numpy()] = "rain"
    phase[~np.isfinite(twet.to_numpy(dtype=float))] = "unknown"
    result[output_column] = pd.Categorical(
        phase,
        categories=["snow", "rain", "transition", "unknown"],
        ordered=False,
    )
    return result


def select_footprint_sample(
    data: pd.DataFrame,
    *,
    required_columns: Sequence[str] = (),
    phase: str | None = None,
    land_only: bool = True,
    phase_column: str = "analysis_phase",
    require_valid_flags: Mapping[str, int] | None = None,
    enforce_reference_valid_flags: bool = True,
) -> pd.DataFrame:
    """Select an analysis population, enforcing named-reference validity flags."""

    needed = list(required_columns)
    if land_only:
        needed.append("land_mask")
    if phase is not None:
        needed.append(phase_column)
    valid_flags = dict(require_valid_flags or {})
    if enforce_reference_valid_flags:
        for reference, flag in REFERENCE_VALID_FLAGS.items():
            if reference in required_columns:
                valid_flags[flag] = 1
    needed.extend(valid_flags)
    ensure_columns(data, needed)

    mask = pd.Series(True, index=data.index)
    if land_only:
        mask &= data["land_mask"].eq(1)
    if phase is not None:
        if phase not in {"rain", "snow", "transition", "unknown"}:
            raise ValueError(f"Unsupported phase: {phase}")
        mask &= data[phase_column].eq(phase)
    for column in required_columns:
        mask &= np.isfinite(pd.to_numeric(data[column], errors="coerce"))
    for column, value in valid_flags.items():
        mask &= data[column].eq(value)
    return data.loc[mask].copy()


def summarize_sample_counts(
    data: pd.DataFrame,
    *,
    total_footprints_seen: int | None = None,
) -> pd.Series:
    """Return the first-run inventory requested by the manuscript prompt."""

    ensure_columns(data, ["land_mask", "analysis_phase", "MRMS", "StageIV"])
    land = data["land_mask"].eq(1)
    rain = data["analysis_phase"].eq("rain")
    return pd.Series(
        {
            "total_native_footprints_seen": int(
                len(data) if total_footprints_seen is None else total_footprints_seen
            ),
            "footprints_in_cache": int(len(data)),
            "land_footprints": int(land.sum()),
            "high_confidence_rain": int((land & rain).sum()),
            "high_confidence_snow": int((land & data["analysis_phase"].eq("snow")).sum()),
            "transition_or_ambiguous": int((land & data["analysis_phase"].eq("transition")).sum()),
            "unknown_phase": int((land & data["analysis_phase"].eq("unknown")).sum()),
            "finite_MRMS": int((land & np.isfinite(data["MRMS"])).sum()),
            "finite_StageIV": int((land & np.isfinite(data["StageIV"])).sum()),
            "MRMS_StageIV_common_rain": int(
                (land & rain & np.isfinite(data["MRMS"]) & np.isfinite(data["StageIV"])).sum()
            ),
        },
        name="sample_count",
        dtype="int64",
    )


# =============================================================================
# RAIN/SNOW METRICS
# =============================================================================

def select_common_reference_sample(data: pd.DataFrame, *, phase: str) -> pd.DataFrame:
    """Return phase-specific land footprints finite for V7, V8, MRMS and Stage IV."""
    return select_footprint_sample(
        data,
        required_columns=["GPROF_V7", "GPROF_V8", "MRMS", "StageIV"],
        phase=phase,
    )

def calculate_reference_agreement(
    data: pd.DataFrame,
    *,
    precipitation_threshold: float,
    phase: str = "rain",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare Stage IV with MRMS on a common phase-specific sample."""
    sample = select_common_reference_sample(data, phase=phase)
    continuous = pd.DataFrame([calculate_continuous_metrics(sample, "MRMS", "StageIV")])
    categorical = pd.DataFrame([
        calculate_categorical_metrics(
            sample,
            reference_column="MRMS",
            product_column="StageIV",
            threshold=precipitation_threshold,
        )
    ])
    continuous.insert(0, "phase", phase)
    categorical.insert(0, "phase", phase)
    return sample, continuous, categorical


def calculate_reference_metric_tables(
    data: pd.DataFrame,
    *,
    reference_column: str,
    product_columns: Sequence[str],
    precipitation_threshold: float,
    phase: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate continuous and categorical metrics on a common-valid
    footprint sample for all evaluated products.

    All products are evaluated against the specified reference using
    exactly the same footprints.

    This ensures a fair comparison among GPROF V7, GPROF V8, ERA5,
    or any other products included in product_columns.
    """

    # ------------------------------------------------------------------
    # Build one common sample for the reference + ALL evaluated products
    # ------------------------------------------------------------------
    required_columns = [
        reference_column,
        *product_columns,
    ]

    sample = select_footprint_sample(
        data,
        required_columns=required_columns,
        phase=phase,
    )

    if sample.empty:
        raise ValueError(
            f"No common-valid {phase} footprints remain for "
            f"reference {reference_column!r} and products "
            f"{list(product_columns)!r}."
        )

    # ------------------------------------------------------------------
    # Calculate metrics for each product using the SAME common sample
    # ------------------------------------------------------------------
    continuous_records = []
    categorical_records = []

    for product in product_columns:

        continuous = calculate_continuous_metrics(
            sample,
            reference_column,
            product,
        )

        continuous.update(
            {
                "phase": phase,
                "reference_label": reference_column,
                "common_sample_count": len(sample),
            }
        )

        categorical = calculate_categorical_metrics(
            sample,
            reference_column=reference_column,
            product_column=product,
            threshold=precipitation_threshold,
        )

        categorical.update(
            {
                "phase": phase,
                "reference_label": reference_column,
                "common_sample_count": len(sample),
            }
        )

        continuous_records.append(
            continuous
        )

        categorical_records.append(
            categorical
        )

    return (
        pd.DataFrame(continuous_records),
        pd.DataFrame(categorical_records),
    )

def select_common_rain_reference_sample(data: pd.DataFrame) -> pd.DataFrame:
    return select_common_reference_sample(data, phase="rain")


def calculate_phase_assessment_metrics(
    data: pd.DataFrame,
    *,
    precipitation_threshold: float,
    product_columns: Sequence[str] = (
        "GPROF_V7",
        "GPROF_V8",
        "ERA5",
    ),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate rain/snow assessment metrics using common-valid samples.

    Reference design
    ----------------
    Rain:
        GPROF V7, GPROF V8, and ERA5 are evaluated separately against:
            1. MRMS
            2. Stage IV

    Snow:
        GPROF V7, GPROF V8, and ERA5 are evaluated separately against:
            1. MRMS
            2. Stage IV

    For each reference/phase combination, all evaluated products use the
    exact same common-valid footprint population.
    """

    continuous_tables = []
    categorical_tables = []

    assessment_cases = (
        ("rain", "MRMS"),
        ("rain", "StageIV"),
        ("snow", "MRMS"),
        ("snow", "StageIV"),
    )

    for phase, reference in assessment_cases:

        continuous, categorical = (
            calculate_reference_metric_tables(
                data,
                reference_column=reference,
                product_columns=product_columns,
                precipitation_threshold=precipitation_threshold,
                phase=phase,
            )
        )

        continuous_tables.append(
            continuous
        )

        categorical_tables.append(
            categorical
        )

    continuous_metrics = pd.concat(
        continuous_tables,
        ignore_index=True,
    )

    categorical_metrics = pd.concat(
        categorical_tables,
        ignore_index=True,
    )

    return (
        continuous_metrics,
        categorical_metrics,
    )

# =============================================================================
# DISTRIBUTIONS
# =============================================================================

def calculate_precipitation_distributions(
    data: pd.DataFrame,
    *,
    product_columns: Sequence[str],
    bin_edges: Sequence[float],
    phase: str,
    precipitation_threshold: float = 0.1,
) -> pd.DataFrame:
    """
    Calculate wet-occurrence and precipitation-volume distributions using
    one common-valid footprint sample for all supplied products.

    The distributions are conditional on precipitation >= the configured
    wet threshold.

    All products appearing in a given phase comparison therefore use the
    exact same footprint population.
    """

    edges = np.asarray(
        bin_edges,
        dtype=float,
    )

    if (
        edges.ndim != 1
        or len(edges) < 2
        or np.any(np.diff(edges) <= 0)
    ):
        raise ValueError(
            "bin_edges must be a strictly increasing 1-D sequence"
        )

    if edges[0] > precipitation_threshold:
        raise ValueError(
            "The first bin edge must not exceed the wet threshold"
        )

    # ------------------------------------------------------------------
    # One common-valid phase sample for ALL products
    # ------------------------------------------------------------------
    sample = select_footprint_sample(
        data,
        required_columns=list(product_columns),
        phase=phase,
    )

    if sample.empty:
        raise ValueError(
            f"No common-valid {phase} footprints remain for "
            f"{list(product_columns)!r}."
        )

    common_sample_count = len(sample)

    # ------------------------------------------------------------------
    # Product distributions
    # ------------------------------------------------------------------
    records: list[dict[str, object]] = []

    for product in product_columns:

        values = sample[
            product
        ].to_numpy(dtype=float)

        wet_values = values[
            values >= precipitation_threshold
        ]

        counts, _ = np.histogram(
            wet_values,
            bins=edges,
        )

        volume, _ = np.histogram(
            wet_values,
            bins=edges,
            weights=wet_values,
        )

        total_count = counts.sum()
        total_volume = volume.sum()

        for index, (left, right) in enumerate(
            zip(
                edges[:-1],
                edges[1:],
            )
        ):

            right_label = (
                "inf"
                if np.isinf(right)
                else f"{right:g}"
            )

            records.append(
                {
                    "phase": phase,
                    "product": product,
                    "bin_index": index,
                    "bin_left": left,
                    "bin_right": right,
                    "bin_label": (
                        f"{left:g}–{right_label}"
                    ),
                    "bin_count": int(
                        counts[index]
                    ),
                    "occurrence_percent": (
                        100.0
                        * counts[index]
                        / total_count
                        if total_count
                        else np.nan
                    ),
                    "volume_percent": (
                        100.0
                        * volume[index]
                        / total_volume
                        if total_volume
                        else np.nan
                    ),
                    "common_sample_count": (
                        common_sample_count
                    ),
                    "wet_sample_count": int(
                        len(wet_values)
                    ),
                }
            )

    return pd.DataFrame(
        records
    )


# =============================================================================
# SPATIAL AGGREGATION
# =============================================================================

def calculate_spatial_fraction_differences(
    spatial_dataset: xr.Dataset,
    *,
    pairs: Sequence[tuple[str, str]] = (
        ("GPROF_V8", "MRMS"),
        ("GPROF_V8", "StageIV"),
        ("GPROF_V7", "MRMS"),
        ("GPROF_V7", "StageIV"),
    ),
) -> xr.Dataset:
    """
    Calculate gridded precipitation-fraction differences.

    The precipitation fractions stored in spatial_dataset are fractions
    from 0 to 1. Differences are converted to percentage points.

    Positive values mean that the first product has a larger
    precipitation-occurrence fraction than the second product.
    """

    output = xr.Dataset(
        coords={
            "latitude": spatial_dataset["latitude"],
            "longitude": spatial_dataset["longitude"],
        }
    )

    for first, second in pairs:

        first_variable = (
            f"{first}_precipitation_fraction"
        )

        second_variable = (
            f"{second}_precipitation_fraction"
        )

        if first_variable not in spatial_dataset:
            raise KeyError(
                f"Missing spatial variable: "
                f"{first_variable}"
            )

        if second_variable not in spatial_dataset:
            raise KeyError(
                f"Missing spatial variable: "
                f"{second_variable}"
            )

        variable_name = (
            f"{first}_minus_{second}"
        )

        output[variable_name] = (
            100.0
            * (
                spatial_dataset[first_variable]
                - spatial_dataset[second_variable]
            )
        )

        output[variable_name].attrs.update(
            units="percentage points",
            sign_convention=(
                f"positive means {first} has a larger "
                f"precipitation fraction than {second}"
            ),
            calculation_method=(
                "100 * (first precipitation fraction "
                "- second precipitation fraction)"
            ),
        )

    output.attrs.update(
        difference_type="precipitation fraction",
        units="percentage points",
    )

    return output

def _empty_spatial_dataset(grid: Mapping[str, object]) -> xr.Dataset:
    return xr.Dataset(
        coords={
            "latitude": np.asarray(grid["lat_centers"], dtype=float),
            "longitude": np.asarray(grid["lon_centers"], dtype=float),
        }
    )


def _spatial_array(series: pd.Series, grid: Mapping[str, object], fill: float = np.nan) -> np.ndarray:
    array = np.full((int(grid["nlat"]), int(grid["nlon"])), fill, dtype=float)
    if len(series):
        lat_index = series.index.get_level_values("lat_index").to_numpy(dtype=int)
        lon_index = series.index.get_level_values("lon_index").to_numpy(dtype=int)
        array[lat_index, lon_index] = series.to_numpy(dtype=float)
    return array


def calculate_spatial_precipitation_characteristics(
    data: pd.DataFrame,
    *,
    grid: Mapping[str, object],
    product_columns: Sequence[str] = PRECIPITATION_COLUMNS,
    precipitation_threshold: float = 0.1,
    minimum_sample_count: int | None = None,
) -> xr.Dataset:
    """
    Calculate 0.25-degree mean precipitation, precipitation fraction,
    and sample counts using one common-valid land footprint population
    for all requested products.

    Using a common sample ensures that spatial differences among products
    are not caused by differences in footprint availability.
    """

    # ------------------------------------------------------------------
    # Required validity flags for reference products
    # ------------------------------------------------------------------
    reference_flags = [
        REFERENCE_VALID_FLAGS[product]
        for product in product_columns
        if product in REFERENCE_VALID_FLAGS
    ]

    ensure_columns(
        data,
        [
            "latitude",
            "longitude",
            "land_mask",
            *product_columns,
            *reference_flags,
        ],
    )

    # ------------------------------------------------------------------
    # Build ONE common-valid land sample
    # ------------------------------------------------------------------
    working_columns = [
        "latitude",
        "longitude",
        *product_columns,
        *reference_flags,
    ]

    working = data.loc[
        data["land_mask"].eq(1),
        working_columns,
    ].copy()

    # Require all relevant reference-valid flags.
    for validity_flag in reference_flags:
        working = working.loc[
            working[validity_flag].eq(1)
        ]

    # Require all requested precipitation products to be finite.
    working = (
        working
        .replace([np.inf, -np.inf], np.nan)
        .dropna(
            subset=[
                "latitude",
                "longitude",
                *product_columns,
            ]
        )
    )

    if working.empty:
        raise ValueError(
            "No common-valid land footprints remain for "
            f"{list(product_columns)!r}."
        )

    # ------------------------------------------------------------------
    # Assign common sample to 0.25-degree analysis grid
    # ------------------------------------------------------------------
    working = add_spatial_grid_indices(
        working,
        grid=grid,
    )

    dataset = _empty_spatial_dataset(
        grid
    )

    # ------------------------------------------------------------------
    # Common sample count
    #
    # This is identical for every product by construction.
    # ------------------------------------------------------------------
    common_grouped = working.groupby(
        [
            "lat_index",
            "lon_index",
        ],
        observed=True,
        sort=False,
    )

    common_count = common_grouped.size()

    dataset["common_sample_count"] = (
        ("latitude", "longitude"),
        _spatial_array(
            common_count,
            grid,
            fill=0,
        ).astype(np.int64),
    )

    # ------------------------------------------------------------------
    # Product statistics
    # ------------------------------------------------------------------
    for product in product_columns:

        grouped = working.groupby(
            [
                "lat_index",
                "lon_index",
            ],
            observed=True,
            sort=False,
        )[product]

        mean = grouped.mean()

        wet_count = grouped.apply(
            lambda values: int(
                (
                    values
                    >= precipitation_threshold
                ).sum()
            )
        )

        fraction = wet_count.div(
            common_count.where(
                common_count > 0
            )
        )

        # --------------------------------------------------------------
        # Optional spatial sample threshold
        # --------------------------------------------------------------
        if minimum_sample_count is not None:

            insufficient = (
                common_count
                < minimum_sample_count
            )

            mean = mean.mask(
                insufficient
            )

            fraction = fraction.mask(
                insufficient
            )

        # --------------------------------------------------------------
        # Store outputs
        # --------------------------------------------------------------
        dataset[
            f"{product}_mean"
        ] = (
            ("latitude", "longitude"),
            _spatial_array(
                mean,
                grid,
            ),
        )

        dataset[
            f"{product}_precipitation_fraction"
        ] = (
            ("latitude", "longitude"),
            _spatial_array(
                fraction,
                grid,
            ),
        )

        dataset[
            f"{product}_sample_count"
        ] = (
            ("latitude", "longitude"),
            _spatial_array(
                common_count,
                grid,
                fill=0,
            ).astype(np.int64),
        )

        dataset[
            f"{product}_wet_count"
        ] = (
            ("latitude", "longitude"),
            _spatial_array(
                wet_count,
                grid,
                fill=0,
            ).astype(np.int64),
        )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    dataset.attrs.update(
        grid_resolution_degrees=float(
            grid["resolution"]
        ),
        precipitation_threshold_mm_h=float(
            precipitation_threshold
        ),
        fraction_definition=(
            "wet observations / common-valid observations"
        ),
        sample_definition=(
            "common-valid land footprints across all requested products"
        ),
        minimum_sample_count=(
            "None"
            if minimum_sample_count is None
            else int(minimum_sample_count)
        ),
        population="land-only footprint samples",
    )

    return dataset


def calculate_spatial_sampling_coverage(
    data: pd.DataFrame,
    *,
    grid: Mapping[str, object],
    paired_products: Sequence[str] = ("GPROF_V7", "GPROF_V8"),
) -> xr.Dataset:
    """Count common-valid paired GMI footprints and unique orbits per grid cell."""

    ensure_columns(
        data,
        ["latitude", "longitude", "land_mask", "orbit", *paired_products],
    )
    working = data.loc[
        data["land_mask"].eq(1),
        ["latitude", "longitude", "orbit", *paired_products],
    ].replace([np.inf, -np.inf], np.nan)
    working = working.dropna(
        subset=["latitude", "longitude", "orbit", *paired_products]
    )
    if working.empty:
        raise ValueError("No common-valid paired GMI footprints remain for coverage mapping.")

    working = add_spatial_grid_indices(working, grid=grid)
    grouped = working.groupby(
        ["lat_index", "lon_index"], observed=True, sort=False
    )
    footprint_count = grouped.size()
    orbit_count = grouped["orbit"].nunique()

    dataset = _empty_spatial_dataset(grid)
    dataset["paired_footprint_count"] = (
        ("latitude", "longitude"),
        _spatial_array(footprint_count, grid, fill=0).astype(np.int64),
    )
    dataset["unique_orbit_count"] = (
        ("latitude", "longitude"),
        _spatial_array(orbit_count, grid, fill=0).astype(np.int64),
    )
    dataset.attrs.update(
        grid_resolution_degrees=float(grid["resolution"]),
        population="common-valid paired GPROF V7 and V8 land footprints",
        total_paired_footprints=int(len(working)),
        total_unique_orbits=int(working["orbit"].nunique()),
    )
    return dataset


def calculate_spatial_mean_differences(
    data: pd.DataFrame,
    *,
    grid: Mapping[str, object],
    pairs: Sequence[tuple[str, str]] = (
        ("GPROF_V8", "GPROF_V7"),
        ("GPROF_V7", "StageIV"),
        ("GPROF_V8", "StageIV"),
    ),
    minimum_sample_count: int | None = None,
) -> xr.Dataset:
    """Calculate paired-sample cell means of first product minus second product."""

    dataset = _empty_spatial_dataset(grid)
    for first, second in pairs:
        sample = select_footprint_sample(
            data, required_columns=["latitude", "longitude", first, second]
        )
        working = add_spatial_grid_indices(sample[["latitude", "longitude", first, second]], grid=grid)
        working["difference"] = working[first] - working[second]
        grouped = working.groupby(["lat_index", "lon_index"], observed=True, sort=False)["difference"]
        count = grouped.count()
        mean = grouped.mean()
        if minimum_sample_count is not None:
            mean = mean.mask(count < minimum_sample_count)
        name = f"{first}_minus_{second}"
        dataset[name] = (("latitude", "longitude"), _spatial_array(mean, grid))
        dataset[f"{name}_sample_count"] = (
            ("latitude", "longitude"), _spatial_array(count, grid, fill=0).astype(np.int64)
        )
        dataset[name].attrs["sign_convention"] = f"positive means {first} exceeds {second}"
    dataset.attrs["difference_method"] = "mean of paired footprint differences in each grid cell"
    return dataset


# =============================================================================
# SEASONAL ANALYSIS
# =============================================================================

def add_time_coordinates(data: pd.DataFrame, time_column: str = "scan_time") -> pd.DataFrame:
    """Add UTC month and meteorological season from the native scan time."""

    ensure_columns(data, [time_column])
    result = data.copy()
    result["datetime_utc"] = pd.to_datetime(result[time_column], errors="coerce", utc=True)
    result["month"] = result["datetime_utc"].dt.month.astype("Int8")
    season_map = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
    result["season"] = pd.Categorical(result["month"].map(season_map), categories=SEASON_ORDER, ordered=True)
    return result


def calculate_seasonal_spatial_means(
    data: pd.DataFrame,
    *,
    grid: Mapping[str, object],
    product_columns: Sequence[str] = PRECIPITATION_COLUMNS,
    minimum_sample_count: int | None = None,
) -> xr.Dataset:
    """Calculate independent-valid land-only product means for DJF/MAM/JJA/SON."""

    timed = add_time_coordinates(data)
    seasonal = []
    for season in SEASON_ORDER:
        dataset = calculate_spatial_precipitation_characteristics(
            timed.loc[timed["season"].eq(season)],
            grid=grid,
            product_columns=product_columns,
            minimum_sample_count=minimum_sample_count,
        )
        seasonal.append(dataset.expand_dims(season=[season]))
    return xr.concat(seasonal, dim="season")


# =============================================================================
# TEMPERATURE-BIN ANALYSIS
# =============================================================================

def calculate_temperature_binned_means(
    data: pd.DataFrame,
    *,
    temperature_bin_edges: Sequence[float],
    product_columns: Sequence[str] = PRECIPITATION_COLUMNS,
) -> pd.DataFrame:
    """Calculate footprint-level mean precipitation and product-specific N by T2M bin."""

    reference_flags = [
        REFERENCE_VALID_FLAGS[product]
        for product in product_columns
        if product in REFERENCE_VALID_FLAGS
    ]
    ensure_columns(data, ["T2M", "land_mask", *product_columns, *reference_flags])
    edges = np.asarray(temperature_bin_edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("temperature_bin_edges must be strictly increasing")
    sample = select_footprint_sample(data, required_columns=["T2M"])
    sample["temperature_bin"] = pd.cut(sample["T2M"], bins=edges, right=False, include_lowest=True)
    records = []
    for interval, group in sample.groupby("temperature_bin", observed=False):
        record: dict[str, object] = {
            "temperature_bin": str(interval),
            "temperature_midpoint": float(interval.mid),
            "temperature_bin_left": float(interval.left),
            "temperature_bin_right": float(interval.right),
            "T2M_sample_count": int(len(group)),
        }
        for product in product_columns:
            values = pd.to_numeric(group[product], errors="coerce")
            validity_flag = REFERENCE_VALID_FLAGS.get(product)
            if validity_flag:
                values = values.where(group[validity_flag].eq(1))
            valid = values[np.isfinite(values)]
            record[f"{product}_mean"] = float(valid.mean()) if len(valid) else np.nan
            record[f"{product}_sample_count"] = int(len(valid))
        records.append(record)
    return pd.DataFrame(records)


# =============================================================================
# METRIC AS A FUNCTION OF TEMPERATURE
def assign_autosnow_surface_regime(
    data: pd.DataFrame,
    *,
    output_column: str = "surface_regime",
) -> pd.DataFrame:
    """Stratify retained land footprints using GMASI AutoSnow classes."""
    ensure_columns(data, ["AutoSnow"])
    result = data.copy()
    autosnow = pd.to_numeric(result["AutoSnow"], errors="coerce")
    labels = np.full(len(result), "unknown", dtype=object)
    labels[autosnow.eq(1).to_numpy()] = "snow_free_land"
    labels[autosnow.eq(2).to_numpy()] = "snow_covered_land"
    result[output_column] = pd.Categorical(
        labels,
        categories=[
            "snow_free_land",
            "snow_covered_land",
            "unknown",
        ],
    )
    return result

def calculate_twet_binned_v8_v7_metrics(
    data: pd.DataFrame,
    *,
    twet_bin_edges: Sequence[float],
    reference_column: str,
    precipitation_threshold: float = 0.1,
    minimum_bin_count: int = 1000,
    minimum_reference_event_count: int = 100,
) -> pd.DataFrame:
    """Compute V7/V8 RB, FB and CSI and V8-vs-V7 changes by Twet/surface."""
    required = [
        "T2MWET",
        "AutoSnow",
        "land_mask",
        "GPROF_V7",
        "GPROF_V8",
        "ERA5",
        reference_column,
    ]
    ensure_columns(data, required)
    edges = np.asarray(twet_bin_edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("twet_bin_edges must be strictly increasing")

    sample = assign_autosnow_surface_regime(data)
    sample = select_footprint_sample(
        sample,
        required_columns=[
            "T2MWET",
            "GPROF_V7",
            "GPROF_V8",
            "ERA5",
            reference_column,
        ],
    )
    sample = sample.loc[sample["surface_regime"].ne("unknown")].copy()
    sample["twet_bin"] = pd.cut(
        sample["T2MWET"], bins=edges, right=False, include_lowest=True,
    )

    records = []
    for (surface, interval), group in sample.groupby(
        ["surface_regime", "twet_bin"],
        observed=False,
    ):
        n = len(group)

        # --------------------------------------------------------------
        # Count reference precipitation events in this specific bin.
        # This is the relevant denominator for FB and CSI.
        # --------------------------------------------------------------
        reference_values = pd.to_numeric(
            group[reference_column],
            errors="coerce",
        ).to_numpy(dtype=float)

        reference_event_count = int(
            np.sum(
                reference_values
                >= precipitation_threshold
            )
        )

        reference_event_fraction = (
            reference_event_count / n
            if n > 0
            else np.nan
        )

        row = {
            "reference_label": reference_column,
            "surface_regime": str(surface),
            "twet_bin": str(interval),
            "twet_bin_left": float(interval.left),
            "twet_bin_right": float(interval.right),
            "twet_midpoint": float(interval.mid),
            "N": int(n),
            "reference_event_count": reference_event_count,
            "reference_event_fraction": reference_event_fraction,
        }
        # --------------------------------------------------------------
        # Total-N and event-count screening
        # --------------------------------------------------------------
        enough_total_samples = (
            n >= minimum_bin_count
        )

        enough_reference_events = (
            reference_event_count
            >= minimum_reference_event_count
        )

        if not (
            enough_total_samples
            and enough_reference_events
        ):
            for product in (
                "GPROF_V7",
                "GPROF_V8",
                "ERA5",
            ):
                for metric in (
                    "correlation",
                    "relative_bias_percent",
                    "frequency_bias",
                    "CSI",
                ):
                    row[f"{product}_{metric}"] = np.nan

            row["relative_bias_difference_pp"] = np.nan
            row["frequency_bias_difference"] = np.nan
            row["CSI_difference"] = np.nan

            row["relative_bias_error_improvement_pp"] = np.nan
            row["frequency_bias_error_improvement"] = np.nan
            row["CSI_improvement"] = np.nan

            row["metrics_retained"] = False
            row["screening_reason"] = (
                f"N={n}; "
                f"reference_events={reference_event_count}"
            )

            records.append(row)
            continue

        row["metrics_retained"] = True
        row["screening_reason"] = "retained"

        metrics = {}

        for product in (
            "GPROF_V7",
            "GPROF_V8",
            "ERA5",
        ):
            continuous = calculate_continuous_metrics(
                group,
                reference_column,
                product,
            )

            categorical = calculate_categorical_metrics(
                group,
                reference_column=reference_column,
                product_column=product,
                threshold=precipitation_threshold,
            )

            metrics[product] = {
                "correlation": float(
                    continuous["correlation"]
                ),
                "relative_bias_percent": float(
                    continuous["relative_bias_percent"]
                ),
                "frequency_bias": float(
                    categorical["frequency_bias"]
                ),
                "CSI": float(
                    categorical["CSI"]
                ),
            }

        # ------------------------------------------------------------------
        # Store the original V7, V8, and ERA5 metric values
        # ------------------------------------------------------------------
        for product in (
            "GPROF_V7",
            "GPROF_V8",
            "ERA5",
        ):
            for metric in (
                "correlation",
                "relative_bias_percent",
                "frequency_bias",
                "CSI",
            ):
                row[f"{product}_{metric}"] = (
                    metrics[product][metric]
                )


        # ------------------------------------------------------------------
        # Extract the metrics for clearer calculations
        # ------------------------------------------------------------------
        rb7 = metrics["GPROF_V7"]["relative_bias_percent"]
        rb8 = metrics["GPROF_V8"]["relative_bias_percent"]

        fb7 = metrics["GPROF_V7"]["frequency_bias"]
        fb8 = metrics["GPROF_V8"]["frequency_bias"]

        csi7 = metrics["GPROF_V7"]["CSI"]
        csi8 = metrics["GPROF_V8"]["CSI"]


        # ------------------------------------------------------------------
        # Raw V8-minus-V7 metric differences
        #
        # Relative-bias difference is in percentage points.
        # Frequency-bias and CSI differences are dimensionless.
        # ------------------------------------------------------------------
        row["relative_bias_difference_pp"] = (
            rb8 - rb7
            if np.isfinite(rb7) and np.isfinite(rb8)
            else np.nan
        )

        row["frequency_bias_difference"] = (
            fb8 - fb7
            if np.isfinite(fb7) and np.isfinite(fb8)
            else np.nan
        )

        row["CSI_difference"] = (
            csi8 - csi7
            if np.isfinite(csi7) and np.isfinite(csi8)
            else np.nan
        )


        # ------------------------------------------------------------------
        # Positive-is-improvement V8-versus-V7 quantities
        #
        # Ideal relative bias = 0
        # Ideal frequency bias = 1
        # Higher CSI is better
        #
        # Positive value: V8 is better
        # Negative value: V7 is better
        # Zero: no change in performance
        # ------------------------------------------------------------------
        row["relative_bias_error_improvement_pp"] = (
            abs(rb7) - abs(rb8)
            if np.isfinite(rb7) and np.isfinite(rb8)
            else np.nan
        )

        row["frequency_bias_error_improvement"] = (
            abs(fb7 - 1.0) - abs(fb8 - 1.0)
            if np.isfinite(fb7) and np.isfinite(fb8)
            else np.nan
        )

        row["CSI_improvement"] = (
            csi8 - csi7
            if np.isfinite(csi7) and np.isfinite(csi8)
            else np.nan
        )

        records.append(row)

    return pd.DataFrame(records)

# =============================================================================

# =============================================================================
# REFERENCE ROBUSTNESS
# =============================================================================

def calculate_reference_dependence(
    data: pd.DataFrame,
    *,
    precipitation_threshold: float,
    negligible_tolerances: Mapping[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Quantify V8 improvement with a positive-is-improvement sign convention.

    For CC/POD/CSI, improvement = V8 - V7.  For RMSE, MAE, absolute relative
    bias, and FAR, improvement = V7 error - V8 error.  Thus every positive
    value means V8 improved, regardless of the metric.
    """

    sample = select_common_rain_reference_sample(data)
    metric_rows = []
    improvement_rows = []
    for reference in ("MRMS", "StageIV"):
        version_metrics: dict[str, dict[str, float | int]] = {}
        for product in ("GPROF_V7", "GPROF_V8"):
            continuous = calculate_continuous_metrics(sample, reference, product)
            categorical = calculate_categorical_metrics(
                sample,
                reference_column=reference,
                product_column=product,
                threshold=precipitation_threshold,
            )
            merged = {**continuous, **categorical, "reference_label": reference}
            version_metrics[product] = merged
            metric_rows.append(merged)
        v7, v8 = version_metrics["GPROF_V7"], version_metrics["GPROF_V8"]
        changes = {
            "CC": v8["correlation"] - v7["correlation"],
            "RMSE": v7["RMSE"] - v8["RMSE"],
            "MAE": v7["MAE"] - v8["MAE"],
            "absolute_relative_bias": abs(v7["relative_bias_percent"]) - abs(v8["relative_bias_percent"]),
            "POD": v8["POD"] - v7["POD"],
            "FAR": v7["FAR"] - v8["FAR"],
            "CSI": v8["CSI"] - v7["CSI"],
        }
        for metric, improvement in changes.items():
            improvement_rows.append(
                {
                    "reference": reference,
                    "metric": metric,
                    "v8_improvement": float(improvement),
                    "sign_convention": "positive = V8 improved; negative = V7 better",
                }
            )

    improvements = pd.DataFrame(improvement_rows)
    summary_rows = []
    for metric, group in improvements.groupby("metric", sort=False):
        values = group.set_index("reference")["v8_improvement"]
        tolerance = float((negligible_tolerances or {}).get(metric, 0.0))
        if abs(values["MRMS"]) <= tolerance and abs(values["StageIV"]) <= tolerance:
            classification = "negligible"
        elif values["MRMS"] > tolerance and values["StageIV"] > tolerance:
            classification = "both references indicate V8 improvement"
        elif values["MRMS"] < -tolerance and values["StageIV"] < -tolerance:
            classification = "both references indicate V7 better"
        else:
            classification = "references disagree"
        summary_rows.append(
            {
                "metric": metric,
                "MRMS_v8_improvement": values["MRMS"],
                "StageIV_v8_improvement": values["StageIV"],
                "negligible_tolerance": tolerance,
                "classification": classification,
            }
        )
    return pd.DataFrame(metric_rows), improvements, pd.DataFrame(summary_rows)

def create_reference_consistency_summary_table(
    continuous_metrics: pd.DataFrame,
    categorical_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create the compact manuscript-oriented MRMS–Stage IV consistency table.

    The table retains the primary continuous and categorical metrics used
    elsewhere in the GPROF V7/V8 assessment:

        Continuous:
            CC
            RMSE
            Relative bias

        Categorical:
            POD
            FAR
            CSI

    MRMS is treated as the reference and Stage IV as the comparison product.
    """

    if continuous_metrics.empty:
        raise ValueError(
            "continuous_metrics is empty."
        )

    if categorical_metrics.empty:
        raise ValueError(
            "categorical_metrics is empty."
        )

    continuous_row = continuous_metrics.iloc[0]
    categorical_row = categorical_metrics.iloc[0]

    table = pd.DataFrame(
        {
            "Metric": [
                "CC",
                "RMSE",
                "Relative bias",
                "POD",
                "FAR",
                "CSI",
            ],

            "Value": [
                float(continuous_row["correlation"]),
                float(continuous_row["RMSE"]),
                float(continuous_row["relative_bias_percent"]),
                float(categorical_row["POD"]),
                float(categorical_row["FAR"]),
                float(categorical_row["CSI"]),
            ],

            "Units": [
                "–",
                "mm h$^{-1}$",
                "%",
                "–",
                "–",
                "–",
            ],
        }
    )

    return table


def calculate_reference_performance_tables(
    data: pd.DataFrame,
    *,
    precipitation_threshold: float,
    phase: str = "rain",
    products: Sequence[str] = (
        "GPROF_V7",
        "GPROF_V8",
        "ERA5",
    ),
    references: Sequence[str] = (
        "MRMS",
        "StageIV",
    ),
) -> dict[str, pd.DataFrame]:
    """
    Build reference-specific performance tables for a common phase sample.

    For each reference (MRMS and Stage IV), report:
        - original GPROF V7 metric
        - original GPROF V8 metric
        - original ERA5 metric
        - relative percentage change of V8 versus V7
        - relative percentage change of V8 versus ERA5

    Relative percentage change is computed uniformly as

        100 × (V8 − comparison) / comparison

    for every metric except relative bias. RB changes are signed differences
    in percentage points: RB_V8 − RB_comparison. Product RB values retain
    their signs. The RB row explicitly identifies the different delta units.

    Positive values indicate that the metric increased in V8,
    whereas negative values indicate that the metric decreased.

    Whether an increase or decrease represents better performance
    depends on the metric itself (e.g., CC/POD/CSI increase is
    desirable, whereas RMSE/MAE/FAR decrease is desirable).
    """

    # ------------------------------------------------------------------
    # Common high-confidence phase sample used for reference robustness.
    # ------------------------------------------------------------------
    sample = select_common_reference_sample(data, phase=phase).copy()

    required_columns = [
        *references,
        *products,
    ]

    ensure_columns(
        sample,
        required_columns,
    )

    # ------------------------------------------------------------------
    # Force all products/references onto the same finite sample.
    # This makes the V7/V8/ERA5 comparisons directly comparable.
    # ------------------------------------------------------------------
    common_valid = np.ones(
        len(sample),
        dtype=bool,
    )

    for column in required_columns:
        values = pd.to_numeric(
            sample[column],
            errors="coerce",
        ).to_numpy(dtype=float)

        common_valid &= np.isfinite(values)

    sample = sample.loc[
        common_valid
    ].copy()

    if sample.empty:
        raise ValueError(
            f"No common-valid {phase} footprints remain for "
            "MRMS, Stage IV, GPROF V7, GPROF V8, and ERA5."
        )

    # ------------------------------------------------------------------
    # Relative-change helper (not a uniform improvement convention).
    # ------------------------------------------------------------------
    def relative_change(
    new_value: float,
    reference_value: float,
    ) -> float:
        """
        Relative percentage change.

            100 * (new - reference) / reference

        Positive values indicate an increase.
        Negative values indicate a decrease.

        Interpretation depends on the metric itself.
        """

        if (
            not np.isfinite(new_value)
            or not np.isfinite(reference_value)
            or np.isclose(reference_value, 0.0)
        ):
            return np.nan

        return (
            100.0
            * (new_value - reference_value)
            / reference_value
        )

    # ------------------------------------------------------------------
    # Metric definitions.
    # ------------------------------------------------------------------
    metric_configuration = {
        "CC": "correlation",
        "RMSE": "RMSE",
        "MAE": "MAE",
        "Relative bias": "relative_bias_percent",
        "POD": "POD",
        "FAR": "FAR",
        "CSI": "CSI",
    }

    # ------------------------------------------------------------------
    # Calculate one table per reference.
    # ------------------------------------------------------------------
    output_tables: dict[str, pd.DataFrame] = {}

    for reference in references:

        product_metrics: dict[str, dict[str, float | int]] = {}

        for product in products:

            continuous = calculate_continuous_metrics(
                sample,
                reference,
                product,
            )

            categorical = calculate_categorical_metrics(
                sample,
                reference_column=reference,
                product_column=product,
                threshold=precipitation_threshold,
            )

            product_metrics[product] = {
                **continuous,
                **categorical,
            }

        rows = []

        for metric_label, source_metric in metric_configuration.items():

            v7_value = float(
                product_metrics["GPROF_V7"][source_metric]
            )

            v8_value = float(
                product_metrics["GPROF_V8"][source_metric]
            )

            era5_value = float(
                product_metrics["ERA5"][source_metric]
            )

            # RB is a signed percentage: report its difference in percentage
            # points, not a relative change that becomes unstable near zero.
            if source_metric == "relative_bias_percent":
                v8_vs_v7 = v8_value - v7_value
                v8_vs_era5 = v8_value - era5_value
            else:
                v8_vs_v7 = relative_change(v8_value, v7_value)
                v8_vs_era5 = relative_change(v8_value, era5_value)

            rows.append(
                {
                    "Metric": metric_label,
                    "Change units": "percentage points" if source_metric == "relative_bias_percent" else "%",
                    "GPROF V7": v7_value,
                    "GPROF V8": v8_value,
                    "ERA5": era5_value,
                    "ΔV8 versus V7": v8_vs_v7,
                    "ΔV8 versus ERA5": v8_vs_era5,
                }
            )

        table = pd.DataFrame(rows)

        # Useful metadata for QC / reporting.
        table.attrs["reference"] = reference
        table.attrs["phase"] = phase
        table.attrs["common_sample_count"] = int(len(sample))
        table.attrs["relative_bias_change_basis"] = "signed RB difference in percentage points"
        table.attrs["precipitation_threshold_mm_h"] = float(
            precipitation_threshold
        )

        output_tables[reference] = table

    return output_tables

__all__ = [
    "PRECIPITATION_COLUMNS",
    "SEASON_ORDER",
    "add_time_coordinates",
    "assign_phase_regime",
    "calculate_phase_assessment_metrics",
    "calculate_precipitation_distributions",
    "calculate_reference_agreement",
    "calculate_reference_dependence",
    "calculate_seasonal_spatial_means",
    "calculate_spatial_mean_differences",
    "calculate_spatial_precipitation_characteristics",
    "calculate_spatial_sampling_coverage",
    "calculate_temperature_binned_means",
    "create_spatial_grid",
    "discover_matchup_files",
    "load_footprint_cache",
    "load_matchup_archive",
    "save_footprint_cache",
    "select_common_rain_reference_sample",
    "select_footprint_sample",
    "create_reference_consistency_summary_table",
    "calculate_reference_performance_tables",
    "summarize_sample_counts",
]

#!/usr/bin/env python3
"""Strictly validate GMI matchup NetCDF files and export diagnostic quick looks."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle
import numpy as np
import xarray as xr

import my_functions_gprof_v7_v8_mrms_raqi_era5_merra2_autosnow as mf

REQUIRED_2D = [
    "latitude", "longitude", "surfacePrecipitation_V7", "surfacePrecipitation_V8",
    "MRMS_Pass2", "RAQI", "ERA5_precipitation", "MERRA2_T2M", "AutoSnow",
    "mrms_domain_flag", "mrms_valid_flag", "raqi_valid_flag", "autosnow_valid_flag", "valid_reference_flag",
]
REQUIRED_1D = [
    "scan_time", "MRMS_valid_time", "RAQI_time", "ERA5_time", "MERRA2_time",
    "MRMS_time_difference_minutes", "RAQI_time_difference_minutes",
    "ERA5_time_difference_minutes", "MERRA2_time_difference_minutes",
]
REQUIRED_VAR_ATTRS = [
    "long_name", "units", "source_dataset", "source_variable",
    "spatial_harmonization_method", "temporal_matching_method", "missing_value_handling",
]
REQUIRED_GLOBAL_ATTRS = [
    "source_gprof_v7_file", "source_gprof_v8_file", "source_mrms_files", "source_raqi_files",
    "source_era5_files", "source_merra2_files", "source_autosnow_files", "common_grid_source",
    "v7_v8_pairing_method", "geometry_qc_status", "geometry_qc_summary", "edge_handling",
    "temporal_conventions", "spatial_harmonization", "software_script", "code_version",
    "creation_time_utc", "orbit_crossed_midnight",
]


def finite_range(values):
    values = np.asarray(values); valid = np.isfinite(values)
    if not np.any(valid):
        return None, None, 1.0
    return float(np.nanmin(values)), float(np.nanmax(values)), float(1.0 - valid.mean())


def _equal_with_nan(a, b, atol=1e-5):
    return bool(np.allclose(np.asarray(a), np.asarray(b), atol=atol, equal_nan=True))


def _datetime_seconds(values):
    values = np.asarray(values).astype("datetime64[ns]")
    out = values.astype(np.int64).astype(np.float64) / 1e9
    out[np.isnat(values)] = np.nan
    return out


def _source_paths(ds, attr):
    return [Path(value) for value in ds.attrs.get(attr, "").split("; ") if value]


def _available_netcdf_times(paths, coordinate_names):
    values = []
    for path in paths:
        with xr.open_dataset(path, decode_times=True, decode_timedelta=False) as source:
            name = next((candidate for candidate in coordinate_names if candidate in source.coords), None)
            if name is None:
                raise KeyError(f"none of {coordinate_names} found in {path}")
            values.extend(_datetime_seconds(source[name].values).tolist())
    return np.asarray(values, dtype=float)


def validate_file(path):
    failures, warnings = [], []
    with xr.open_dataset(path, decode_times=True, decode_timedelta=False) as ds:
        nscan, npixel = ds.sizes.get("scan"), ds.sizes.get("pixel")
        if nscan is None or npixel is None:
            failures.append("missing scan/pixel dimensions")
            return {"file": str(path), "failures": failures, "warnings": warnings}
        for name in REQUIRED_2D + REQUIRED_1D:
            if name not in ds:
                failures.append(f"missing variable {name}")
        if failures:
            return {"file": str(path), "failures": failures, "warnings": warnings}
        for name in REQUIRED_2D:
            if ds[name].shape != (nscan, npixel):
                failures.append(f"{name} shape {ds[name].shape} != {(nscan, npixel)}")
        for name in REQUIRED_1D:
            if ds[name].shape != (nscan,):
                failures.append(f"{name} shape {ds[name].shape} != {(nscan,)}")
        for name in [*REQUIRED_2D, *REQUIRED_1D, "scan", "pixel"]:
            missing = [attr for attr in REQUIRED_VAR_ATTRS
                       if attr not in ds[name].attrs and attr not in ds[name].encoding]
            if missing:
                failures.append(f"{name} missing attributes: {', '.join(missing)}")
        missing_global = [name for name in REQUIRED_GLOBAL_ATTRS if name not in ds.attrs]
        if missing_global:
            failures.append(f"missing global attributes: {', '.join(missing_global)}")
        if ds.attrs.get("geometry_qc_status") not in ("PASS_NORMAL", "PASS_MINOR_EDGE_WARNING"):
            failures.append(f"geometry QC did not pass: {ds.attrs.get('geometry_qc_status')}")

        scan_seconds = _datetime_seconds(ds.scan_time.values)
        time_map = {
            "MRMS": "MRMS_valid_time", "RAQI": "RAQI_time",
            "ERA5": "ERA5_time", "MERRA2": "MERRA2_time",
        }
        temporal = {}
        for prefix, time_name in time_map.items():
            selected = _datetime_seconds(ds[time_name].values)
            expected = mf.time_difference_minutes(selected, scan_seconds)
            actual = ds[f"{prefix}_time_difference_minutes"].values
            if not _equal_with_nan(expected, actual, atol=2e-4):
                failures.append(f"{prefix} time-difference diagnostic is inconsistent")
            finite_actual = actual[np.isfinite(actual)]
            temporal[prefix] = {
                "selected_times": [str(value) for value in np.unique(ds[time_name].values)],
                "difference_min": float(np.min(finite_actual)) if finite_actual.size else None,
                "difference_max": float(np.max(finite_actual)) if finite_actual.size else None,
            }

        source_time_checks = {}
        source_specs = {
            "RAQI": ("source_raqi_files", ("time",)),
            "ERA5": ("source_era5_files", ("valid_time", "time")),
            "MERRA2": ("source_merra2_files", ("time",)),
        }
        for prefix, (source_attr, coordinate_names) in source_specs.items():
            stored = _datetime_seconds(ds[time_map[prefix]].values)
            stored = np.unique(stored[np.isfinite(stored)])
            available = _available_netcdf_times(_source_paths(ds, source_attr), coordinate_names)
            matched = all(np.any(np.isclose(value, available, atol=.5)) for value in stored)
            source_time_checks[prefix] = bool(matched)
            if not matched:
                failures.append(f"{prefix} stored time is not an actual source coordinate")
        mrms_stamps = []
        for source in _source_paths(ds, "source_mrms_files"):
            match = mf.MRMS_RE.fullmatch(source.name)
            if match:
                value = dt.datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
                mrms_stamps.append((value - mf.EPOCH).total_seconds())
        stored_mrms = _datetime_seconds(ds.MRMS_valid_time.values)
        stored_mrms = np.unique(stored_mrms[np.isfinite(stored_mrms)])
        mrms_matched = all(np.any(np.isclose(value, mrms_stamps, atol=.5)) for value in stored_mrms)
        source_time_checks["MRMS"] = bool(mrms_matched)
        if not mrms_matched:
            failures.append("MRMS stored time does not match a selected source file validity stamp")
        mrms_time = _datetime_seconds(ds.MRMS_valid_time.values)
        raqi_time = _datetime_seconds(ds.RAQI_time.values)
        both_times = np.isfinite(mrms_time) & np.isfinite(raqi_time)
        if not np.allclose(mrms_time[both_times], raqi_time[both_times]):
            failures.append("RAQI_time and MRMS_valid_time differ where both fields loaded")

        field_time_map = {
            "MRMS_Pass2": "MRMS_valid_time", "RAQI": "RAQI_time",
            "ERA5_precipitation": "ERA5_time", "MERRA2_T2M": "MERRA2_time",
        }
        for field_name, time_name in field_time_map.items():
            field_available = np.any(np.isfinite(ds[field_name].values), axis=1)
            time_available = np.isfinite(_datetime_seconds(ds[time_name].values))
            if np.any(field_available & ~time_available):
                failures.append(f"{field_name} has sampled data but {time_name} is missing")

        mrms = ds.MRMS_Pass2.values; raqi = ds.RAQI.values; snow = ds.AutoSnow.values
        expected_mrms = np.isfinite(mrms).astype(np.uint8)
        expected_rq = np.isfinite(raqi).astype(np.uint8)
        expected_snow = np.isfinite(snow).astype(np.uint8)
        expected_ref = (np.isfinite(mrms) & np.isfinite(raqi)).astype(np.uint8)
        for name, expected in [("mrms_valid_flag", expected_mrms),
                               ("raqi_valid_flag", expected_rq),
                               ("autosnow_valid_flag", expected_snow),
                               ("valid_reference_flag", expected_ref)]:
            if not np.array_equal(ds[name].values, expected):
                failures.append(f"{name} is inconsistent with its definition")
            if not set(np.unique(ds[name].values)).issubset({0, 1}):
                failures.append(f"{name} contains values outside 0/1")
        if not set(np.unique(ds.mrms_domain_flag.values)).issubset({0, 1}):
            failures.append("mrms_domain_flag contains values outside 0/1")
        bounds = tuple(float(ds.attrs[f"mrms_native_center_{side}"])
                       for side in ("south", "north", "west", "east"))
        expected_domain, *_ = mf.quality_flags(
            mrms, raqi, snow, ds.latitude.values, ds.longitude.values, bounds
        )
        if not np.array_equal(ds.mrms_domain_flag.values, expected_domain):
            failures.append("mrms_domain_flag is inconsistent with native center-coordinate bounds")
        required_orientation = {"mrms_iScansNegatively": 0, "mrms_jScansPositively": 0,
                                "mrms_jPointsAreConsecutive": 0,
                                "mrms_alternativeRowScanning": 0}
        for attr, expected in required_orientation.items():
            if int(ds.attrs.get(attr, -1)) != expected:
                failures.append(f"{attr} is not the verified operational value {expected}")

        domain = ds.mrms_domain_flag.values.astype(bool)
        finite_mrms = np.isfinite(mrms)
        outside_finite = (~domain) & finite_mrms
        inside_nonfinite = domain & ~finite_mrms
        boundary_qc = {
            "count_domain_0_and_finite_MRMS_Pass2": int(np.count_nonzero(outside_finite)),
            "count_domain_1_and_nonfinite_MRMS_Pass2": int(np.count_nonzero(inside_nonfinite)),
            "outside_domain_finite_zero_count": int(np.count_nonzero(outside_finite & (mrms == 0))),
            "outside_domain_finite_positive_count": int(np.count_nonzero(outside_finite & (mrms > 0))),
        }

        ranges = {}
        for name in REQUIRED_2D[:9]:
            lo, hi, missing = finite_range(ds[name].values)
            ranges[name] = {"min": lo, "max": hi, "missing_fraction": missing}
        for name in ("surfacePrecipitation_V7", "surfacePrecipitation_V8", "MRMS_Pass2", "ERA5_precipitation"):
            if ranges[name]["min"] is not None and ranges[name]["min"] < 0:
                failures.append(f"{name} contains negative values")
        if ranges["RAQI"]["min"] is not None and not (ranges["RAQI"]["min"] >= 0 and ranges["RAQI"]["max"] <= 1):
            failures.append("RAQI outside 0..1")
        if ranges["MERRA2_T2M"]["min"] is not None and not (150 <= ranges["MERRA2_T2M"]["min"] <= ranges["MERRA2_T2M"]["max"] <= 350):
            failures.append("MERRA2_T2M outside plausible 150..350 K")
        snow_values = np.unique(snow[np.isfinite(snow)])
        if not set(snow_values.tolist()).issubset({0, 1, 2, 3}):
            failures.append(f"invalid AutoSnow classes: {snow_values.tolist()}")
        if ranges["MRMS_Pass2"]["missing_fraction"] > 0.8:
            warnings.append("Large MRMS missing fraction is expected for a global orbit and CONUS-only reference")

        result = {
            "file": str(path), "status": "PASS" if not failures else "FAIL",
            "dimensions": {"scan": nscan, "pixel": npixel},
            "file_size_mib": round(os.path.getsize(path) / 2**20, 3),
            "geometry_qc": ds.attrs.get("geometry_qc_status"),
            "crossed_midnight": ds.attrs.get("orbit_crossed_midnight"),
            "ranges": ranges, "temporal": temporal,
            "actual_source_time_checks": source_time_checks,
            "mrms_boundary_qc": boundary_qc,
            "quality_flag_counts": {name: int(ds[name].values.sum()) for name in
                                    ("mrms_domain_flag", "mrms_valid_flag", "raqi_valid_flag", "autosnow_valid_flag", "valid_reference_flag")},
            "autosnow_classes": snow_values.tolist(), "failures": failures, "warnings": warnings,
        }
    return result


def _robust_max(*arrays, percentile=99.5):
    values = np.concatenate([np.asarray(a)[np.isfinite(a)] for a in arrays])
    return max(float(np.percentile(values, percentile)), 0.1) if values.size else 1.0


def make_field_figure(path, output_dir):
    with xr.open_dataset(path, decode_times=True, decode_timedelta=False) as ds:
        fields = [
            ("surfacePrecipitation_V7", "GPROF V7", "Blues"),
            ("surfacePrecipitation_V8", "GPROF V8", "Blues"),
            ("MRMS_Pass2", "MRMS Pass2", "Blues"),
            ("RAQI", "RAQI", "cividis"),
            ("ERA5_precipitation", "ERA5", "Blues"),
            ("MERRA2_T2M", "MERRA-2 T2M", "coolwarm"),
            ("AutoSnow", "AutoSnow class", None),
            ("valid_reference_flag", "Valid MRMS + RAQI", "Greys"),
        ]
        v7max = _robust_max(ds.surfacePrecipitation_V7.values, ds.surfacePrecipitation_V8.values)
        fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
        cat_cmap = ListedColormap(["#2563a6", "#b7a56a", "#f4f7fb", "#dd8a22"])
        for ax, (name, title, cmap) in zip(axes.flat, fields):
            data = ds[name].values
            kwargs = {"aspect": "auto", "origin": "lower", "interpolation": "nearest"}
            if name.startswith("surfacePrecipitation"):
                kwargs.update(cmap=cmap, vmin=0, vmax=v7max)
            elif name == "RAQI": kwargs.update(cmap=cmap, vmin=0, vmax=1)
            elif name == "AutoSnow": kwargs.update(cmap=cat_cmap, norm=BoundaryNorm([-0.5, .5, 1.5, 2.5, 3.5], 4))
            elif name == "valid_reference_flag": kwargs.update(cmap=cmap, vmin=0, vmax=1)
            else: kwargs.update(cmap=cmap)
            image = ax.imshow(data.T, **kwargs)
            ax.set_title(title, fontsize=11, color="#202936")
            ax.set_xlabel("Scan"); ax.set_ylabel("Pixel")
            cb = fig.colorbar(image, ax=ax, shrink=.82,
                              ticks=[0, 1, 2, 3] if name == "AutoSnow" else None)
            if name == "AutoSnow":
                cb.ax.set_yticklabels(["water", "land", "snow", "ice"])
            cb.ax.tick_params(labelsize=8)
        fig.suptitle(
            f"GMI collocation field quick look — {path.stem}\n"
            "GPROF V7/V8 color scales capped at the joint orbit 99.5th percentile",
            fontsize=13, color="#202936"
        )
        out = output_dir / f"{path.stem}_fields.png"
        fig.savefig(out, dpi=150, facecolor="white"); plt.close(fig)
    return out


def make_time_figure(path, output_dir):
    with xr.open_dataset(path, decode_times=True, decode_timedelta=False) as ds:
        fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
        styles = [
            ("MRMS_time_difference_minutes", "MRMS", "#2563a6", "-"),
            ("RAQI_time_difference_minutes", "RAQI", "#dd8a22", "--"),
            ("ERA5_time_difference_minutes", "ERA5", "#735c0f", ":"),
            ("MERRA2_time_difference_minutes", "MERRA-2", "#c6537a", "-."),
        ]
        scan = ds.scan.values
        for name, label, color, linestyle in styles:
            ax.plot(scan, ds[name].values, label=label, color=color, linestyle=linestyle, linewidth=1.25)
        ax.axhline(0, color="#303846", linewidth=.8)
        ax.set_title(f"Ancillary selected time minus GMI scan time — {path.stem}", color="#202936")
        ax.set_xlabel("Native GMI V7 scan index"); ax.set_ylabel("Time difference (minutes)")
        ax.grid(axis="y", color="#d8dde5", linewidth=.6); ax.legend(ncol=4, frameon=False, loc="upper center")
        out = output_dir / f"{path.stem}_time_offsets.png"
        fig.savefig(out, dpi=150, facecolor="white"); plt.close(fig)
    return out


def _lon180(value):
    return ((float(value) + 180.0) % 360.0) - 180.0


def make_mrms_boundary_figure(path, output_dir):
    """Plot native, domain-test, and post-harmonization MRMS boundaries."""
    with xr.open_dataset(path, decode_times=True, decode_timedelta=False) as ds:
        lat = ds.latitude.values
        lon = ((ds.longitude.values.astype(float) + 180.0) % 360.0) - 180.0
        mrms = ds.MRMS_Pass2.values
        domain = ds.mrms_domain_flag.values.astype(bool)
        finite = np.isfinite(mrms)

        cs = float(ds.attrs["mrms_native_center_south"])
        cn = float(ds.attrs["mrms_native_center_north"])
        cw = _lon180(ds.attrs["mrms_native_center_west"])
        ce = _lon180(ds.attrs["mrms_native_center_east"])
        es = float(ds.attrs["mrms_native_edge_south"])
        en = float(ds.attrs["mrms_native_edge_north"])
        ew = _lon180(ds.attrs["mrms_native_edge_west"])
        ee = _lon180(ds.attrs["mrms_native_edge_east"])

        distance = np.minimum.reduce([
            np.abs(lat - cs), np.abs(lat - cn), np.abs(lon - cw), np.abs(lon - ce)
        ])
        near = (lat >= es - 0.75) & (lat <= en + 0.75) & (lon >= ew - 0.75) & (lon <= ee + 0.75)
        near &= distance <= 0.75
        outside_finite = near & ~domain & finite

        fig, ax = plt.subplots(figsize=(13, 7), constrained_layout=True)
        flat_near = np.flatnonzero(near.ravel())[::4]
        ax.scatter(lon.ravel()[flat_near], lat.ravel()[flat_near], s=4, color="#9aa3af",
                   alpha=.45, label="GMI footprint centers near boundary", rasterized=True)
        finite_near = np.flatnonzero((near & finite).ravel())[::2]
        ax.scatter(lon.ravel()[finite_near], lat.ravel()[finite_near], s=7, color="#dd8a22",
                   alpha=.65, label="finite MRMS_Pass2 after ERA5 harmonization", rasterized=True)
        outside = np.flatnonzero(outside_finite.ravel())
        ax.scatter(lon.ravel()[outside], lat.ravel()[outside], s=17, facecolors="none",
                   edgecolors="#c3342b", linewidths=.8,
                   label="finite MRMS_Pass2 outside domain flag", rasterized=True)

        ax.add_patch(Rectangle((ew, es), ee - ew, en - es, fill=False, linewidth=2,
                               edgecolor="#202936", label="native MRMS grid outer boundary"))
        ax.add_patch(Rectangle((cw, cs), ce - cw, cn - cs, fill=False, linewidth=1.8,
                               linestyle="--", edgecolor="#2563a6",
                               label="mrms_domain_flag center-coordinate boundary"))
        required_harmonized = [f"mrms_harmonized_finite_center_{side}"
                               for side in ("south", "north", "west", "east")]
        if all(name in ds.attrs for name in required_harmonized):
            hs = float(ds.attrs[required_harmonized[0]])
            hn = float(ds.attrs[required_harmonized[1]])
            hw = _lon180(ds.attrs[required_harmonized[2]])
            he = _lon180(ds.attrs[required_harmonized[3]])
            ax.add_patch(Rectangle((hw, hs), he - hw, hn - hs, fill=False, linewidth=2,
                                   linestyle="-.", edgecolor="#8a4f00",
                                   label="finite ERA5-cell-center envelope"))
        ax.set_xlim(ew - 1.0, ee + 1.0); ax.set_ylim(es - 1.0, en + 1.0)
        ax.set_xlabel("Longitude (degrees east)"); ax.set_ylabel("Latitude (degrees north)")
        ax.set_title(f"MRMS boundary QC — {path.stem}", color="#202936")
        ax.grid(color="#d8dde5", linewidth=.5); ax.legend(loc="best", frameon=True, fontsize=9)
        out = output_dir / f"{path.stem}_mrms_boundary.png"
        fig.savefig(out, dpi=170, facecolor="white"); plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(mf.DEFAULT_OUTPUT_ROOT / "matchups" / "GMI"))
    parser.add_argument("--output-dir", default=str(mf.DEFAULT_OUTPUT_ROOT / "qc"))
    parser.add_argument("--pattern", default="GMI_GPROF_V7_V8_*.nc")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    paths = sorted(Path(args.input_dir).glob(args.pattern))
    if not paths:
        raise SystemExit(f"No files matched {Path(args.input_dir) / args.pattern}")
    output_dir = Path(args.output_dir); figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_figures: figure_dir.mkdir(parents=True, exist_ok=True)
    results = []
    orientation_qc = None
    for path in paths:
        result = validate_file(path)
        if orientation_qc is None and result.get("status") == "PASS":
            with xr.open_dataset(path, decode_times=False, decode_timedelta=False) as ds:
                source = ds.attrs.get("source_mrms_files", "").split("; ")[0]
            if source:
                orientation_qc = mf.inspect_mrms_grib_orientation(source)
                if not orientation_qc["passed"]:
                    result["status"] = "FAIL"
                    result["failures"].append("independent MRMS GRIB coordinate-orientation check failed")
        if not args.no_figures and result.get("status") == "PASS":
            result["figures"] = [str(make_field_figure(path, figure_dir)),
                                 str(make_time_figure(path, figure_dir)),
                                 str(make_mrms_boundary_figure(path, figure_dir))]
        results.append(result); print(json.dumps(result, default=mf.json_ready))
    report = {"created_utc": dt.datetime.now(dt.timezone.utc), "files": results,
              "mrms_grib_orientation_qc": orientation_qc,
              "passed": sum(r.get("status") == "PASS" for r in results),
              "failed": sum(r.get("status") != "PASS" for r in results)}
    mf.write_json(output_dir / "gmi_validation_summary.json", report)
    if report["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

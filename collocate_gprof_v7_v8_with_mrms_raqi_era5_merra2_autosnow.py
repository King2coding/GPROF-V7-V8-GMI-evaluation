#!/usr/bin/env python3
"""Generalized native-footprint GPROF V7/V8 collocation driver."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import xarray as xr

import my_functions_gprof_v7_v8_mrms_raqi_era5_merra2_autosnow as mf

SCRIPT = Path(__file__).name
CODE_VERSION = "2.1.0"


def output_path(config, pair, output_root):
    return Path(output_root) / "matchups" / config["config_name"] / (
        f"{config['config_name']}_GPROF_V7_V8_MRMS_RAQI_ERA5_MERRA2_AutoSnow_"
        f"{pair['orbit_id']}.nc"
    )


def _nearest_merra2_time(value):
    base = dt.datetime.combine(value.date(), dt.time(0, 30))
    if value < base:
        previous = base - dt.timedelta(hours=1)
        return base if (base - value) < (value - previous) else previous
    hours = int((value - base).total_seconds() // 3600)
    lower = base + dt.timedelta(hours=hours); upper = lower + dt.timedelta(hours=1)
    return upper if (upper - value) < (value - lower) else lower


def _variable_attrs(long_name, units, source, spatial, temporal, source_variable):
    return {
        "long_name": long_name, "units": units, "source_dataset": source,
        "source_variable": source_variable, "spatial_harmonization_method": spatial,
        "temporal_matching_method": temporal,
        "missing_value_handling": "Invalid/source-fill values are represented by NetCDF fill values (decoded as NaN).",
    }


def process_pair(pair, config, roots, output_root, overwrite=False, geometry_tolerance=5e-3):
    final = output_path(config, pair, output_root)
    partial = Path(str(final) + ".partial")
    if final.exists() and not overwrite:
        if partial.exists():
            partial.unlink()
        return {"status": "skipped_existing", "sensor": config["config_name"],
                "orbit_id": pair["orbit_id"], "output": str(final)}
    if partial.exists():
        partial.unlink()

    v7 = mf.read_gprof(pair["v7_file"], config)
    v8 = mf.read_gprof(pair["v8_file"], config)
    qc = mf.geometry_qc(v7, v8, geometry_tolerance)
    if not qc["passed"]:
        return {"status": "failed_geometry", "sensor": config["config_name"],
                "orbit_id": pair["orbit_id"], "geometry_qc": qc}

    # V7 is the sole output geometry and time reference.
    lat, lon, scan_dt, scan_seconds = v7["lat"], v7["lon"], v7["scan_datetimes"], v7["scan_seconds"]
    first_era, era_lat, era_lon, first_era_path, _ = mf.read_era5_hour(
        roots["era5"], mf.nearest_hour(scan_dt[0])
    )
    shape = lat.shape
    fields = {name: np.full(shape, np.nan, np.float32) for name in
              ("MRMS_Pass2", "RAQI", "ERA5_precipitation", "MERRA2_T2M")}
    autosnow = np.full(shape, mf.AUTOSNOW_FILL, np.uint8)
    mrms_times = np.full(len(scan_dt), np.nan); raqi_times = np.full(len(scan_dt), np.nan)
    era_times = np.full(len(scan_dt), np.nan); merra_times = np.full(len(scan_dt), np.nan)
    sources = {name: set() for name in ("mrms", "raqi", "era5", "merra2", "autosnow")}
    gaps = []
    mrms_bounds = (1.0, 0.0, 0.0, 0.0)  # empty until a source grid is read
    mrms_meta = None
    mrms_harmonized_bounds = None

    # The same IMERG-style nearest valid hour T is used for MRMS and RAQI.
    selected_hours = np.array([mf.nearest_hour(value) for value in scan_dt], dtype=object)
    for hour in sorted(set(selected_hours)):
        rows = np.where(selected_hours == hour)[0]
        try:
            era, elat, elon, era_path, actual_time = mf.read_era5_hour(roots["era5"], hour)
            sampled = mf.sample_era_grid(era, elat, elon, lat[rows], lon[rows])
            fields["ERA5_precipitation"][rows] = sampled
            era_times[rows] = actual_time
            sources["era5"].add(era_path)
        except Exception as exc:
            gaps.append(f"ERA5 {hour.isoformat()}: {exc}")
        mrms_path = mf.mrms_file(roots["mrms"], hour)
        if not mrms_path:
            gaps.append(f"MRMS hour unavailable {hour.isoformat()}")
        else:
            try:
                mrms, mlat, mlon, current_meta = mf.read_mrms(mrms_path)
                current_bounds = (float(np.nanmin(mlat)), float(np.nanmax(mlat)),
                                  float(np.nanmin(mlon)), float(np.nanmax(mlon)))
                mrms_era = mf.resample_to_era5(mrms, mlat, mlon, era_lat, era_lon)
                sampled = mf.sample_era_grid(mrms_era, era_lat, era_lon, lat[rows], lon[rows])
                fields["MRMS_Pass2"][rows] = sampled
                mrms_times[rows] = float(current_meta["valid_time_seconds"])
                mrms_bounds = current_bounds
                mrms_meta = current_meta
                finite_y, finite_x = np.where(np.isfinite(mrms_era))
                if finite_y.size:
                    bounds = (float(np.min(era_lat[finite_y])), float(np.max(era_lat[finite_y])),
                              float(np.min(era_lon[finite_x])), float(np.max(era_lon[finite_x])))
                    if mrms_harmonized_bounds is None:
                        mrms_harmonized_bounds = bounds
                    else:
                        mrms_harmonized_bounds = (
                            min(mrms_harmonized_bounds[0], bounds[0]),
                            max(mrms_harmonized_bounds[1], bounds[1]),
                            min(mrms_harmonized_bounds[2], bounds[2]),
                            max(mrms_harmonized_bounds[3], bounds[3]),
                        )
                sources["mrms"].add(mrms_path)
            except Exception as exc:
                gaps.append(f"MRMS {hour.isoformat()}: {exc}")

        try:
            raqi, rlat, rlon, raqi_path, actual_time = mf.read_raqi(roots["raqi"], hour)
            raqi_era = mf.resample_to_era5(raqi, rlat, rlon, era_lat, era_lon)
            sampled = mf.sample_era_grid(raqi_era, era_lat, era_lon, lat[rows], lon[rows])
            fields["RAQI"][rows] = sampled
            raqi_times[rows] = actual_time
            sources["raqi"].add(raqi_path)
        except Exception as exc:
            gaps.append(f"RAQI {hour.isoformat()}: {exc}")

    selected_merra = np.array([_nearest_merra2_time(value) for value in scan_dt], dtype=object)
    for hour in sorted(set(selected_merra)):
        rows = np.where(selected_merra == hour)[0]
        try:
            data, slat, slon, path, _, actual_time = mf.read_merra2_t2m(roots["merra2"], hour)
            data_era = mf.resample_to_era5(data, slat, slon, era_lat, era_lon)
            sampled = mf.sample_era_grid(data_era, era_lat, era_lon, lat[rows], lon[rows])
            fields["MERRA2_T2M"][rows] = sampled
            merra_times[rows] = actual_time
            sources["merra2"].add(path)
        except Exception as exc:
            gaps.append(f"MERRA2 {hour.isoformat()}: {exc}")

    scan_dates = np.array([value.date() for value in scan_dt], dtype=object)
    for date in sorted(set(scan_dates)):
        rows = np.where(scan_dates == date)[0]
        try:
            data, slat, slon, path = mf.read_autosnow(roots["autosnow"], date)
            data_era = mf.resample_to_era5(data, slat, slon, era_lat, era_lon, categorical=True)
            autosnow[rows] = mf.sample_era_grid(data_era, era_lat, era_lon, lat[rows], lon[rows])
            sources["autosnow"].add(path)
        except Exception as exc:
            gaps.append(f"AutoSnow {date.isoformat()}: {exc}")

    crossed_midnight = len(set(scan_dates)) > 1
    continuous = "Arithmetic mean of valid source-grid pixels contributing to each native ERA5 cell; nearest ERA5 cell sampled at V7 footprint center."
    mrms_domain, mrms_valid, raqi_valid, autosnow_valid, valid_reference = mf.quality_flags(
        fields["MRMS_Pass2"], fields["RAQI"], autosnow, lat, lon, mrms_bounds
    )
    mrms_dt = mf.time_difference_minutes(mrms_times, scan_seconds)
    raqi_dt = mf.time_difference_minutes(raqi_times, scan_seconds)
    era_dt = mf.time_difference_minutes(era_times, scan_seconds)
    merra_dt = mf.time_difference_minutes(merra_times, scan_seconds)
    native_space = "native GMI V7 scan and footprint geometry"
    native_time = "native GMI V7 scan time; no temporal interpolation"
    index_missing = "No missing values; indices enumerate the complete native V7 dimension."
    time_missing = "Missing ancillary selections are represented by NetCDF fill values (decoded as NaN)."
    flag_missing = "No missing values; 1 means true/valid and 0 means false/invalid."
    ds = xr.Dataset(
        data_vars={
            "latitude": (("scan", "pixel"), lat, {**_variable_attrs(
                "GMI V7 footprint-center latitude", "degrees_north", "GPROF V7", native_space,
                native_time, config["latitude_variable"]), "standard_name": "latitude"}),
            "longitude": (("scan", "pixel"), lon, {**_variable_attrs(
                "GMI V7 footprint-center longitude", "degrees_east", "GPROF V7", native_space,
                native_time, config["longitude_variable"]), "standard_name": "longitude"}),
            "scan_time": (("scan",), scan_seconds, {**_variable_attrs(
                "GMI V7 scan time", mf.TIME_UNITS, "GPROF V7", native_space, native_time,
                config["scan_time_variable"]), "standard_name": "time", "calendar": "standard"}),
            "surfacePrecipitation_V7": (("scan", "pixel"), v7["precip"], _variable_attrs(
                f"{config['sensor']} GPROF V7 surface precipitation", v7["precip_units"], "GPROF V7", "native V7 geometry", "native scan time", config["precipitation_variable"])),
            "surfacePrecipitation_V8": (("scan", "pixel"), v8["precip"], _variable_attrs(
                f"{config['sensor']} GPROF V8 surface precipitation on compatible V7 geometry", v8["precip_units"], "GPROF V8", "geometry verified against V7; V7 coordinates retained", "paired scan time within 1 second", config["precipitation_variable"])),
            "MRMS_Pass2": (("scan", "pixel"), fields["MRMS_Pass2"], _variable_attrs(
                "MRMS MultiSensor QPE 01H Pass2 one-hour precipitation accumulation", "mm", "MRMS MultiSensor_QPE_01H_Pass2", continuous, "nearest hourly valid time to each scan; half-hour ties choose earlier", "GRIB2 field 1")),
            "RAQI": (("scan", "pixel"), fields["RAQI"], _variable_attrs(
                "Reconstructed hourly radar accumulation quality index", "1", "Reconstructed RAQI", continuous, "identical MRMS valid hour T", "RAQI")),
            "ERA5_precipitation": (("scan", "pixel"), fields["ERA5_precipitation"], _variable_attrs(
                "ERA5 hourly total precipitation", "mm", "ERA5 hourly", "native ERA5 value; nearest ERA5 cell sampled at V7 footprint center", "nearest hourly valid time; half-hour ties choose earlier", "tp")),
            "MERRA2_T2M": (("scan", "pixel"), fields["MERRA2_T2M"], _variable_attrs(
                "MERRA-2 2-meter air temperature", "K", "MERRA-2", continuous, "nearest available hourly time (fields centered at :30); ties choose earlier", "T2M")),
            "AutoSnow": (("scan", "pixel"), autosnow, {
                "long_name": "GMASI AutoSnow majority class", "units": "1", "source_dataset": "GMASI AutoSnow v003",
                "source_variable": "autosnow_class", "flag_values": np.array([0, 1, 2, 3], dtype=np.uint8),
                "flag_meanings": "clear_water snow_free_land snow_covered_land ice_covered_water",
                "spatial_harmonization_method": "Mode (majority class) of valid source pixels contributing to each ERA5 cell; nearest ERA5 cell sampled at V7 footprint center.",
                "temporal_matching_method": "daily map selected from each scan's actual UTC date", "missing_value": int(mf.AUTOSNOW_FILL),
                "missing_value_handling": "Values outside 0..3 and missing daily files are 255."}),
            "MRMS_valid_time": (("scan",), mrms_times, _variable_attrs(
                "Selected MRMS valid time", mf.TIME_UNITS, "MRMS MultiSensor_QPE_01H_Pass2",
                continuous, "nearest hourly valid time; exact half-hour ties choose earlier", "GRIB2 validity time")),
            "RAQI_time": (("scan",), raqi_times, _variable_attrs(
                "Selected RAQI interval-ending time", mf.TIME_UNITS, "Reconstructed RAQI",
                continuous, "identical selected valid hour as MRMS", "time")),
            "ERA5_time": (("scan",), era_times, _variable_attrs(
                "Selected ERA5 valid time", mf.TIME_UNITS, "ERA5 hourly", "native ERA5 grid",
                "nearest hourly valid time; exact half-hour ties choose earlier", "valid_time")),
            "MERRA2_time": (("scan",), merra_times, _variable_attrs(
                "Selected MERRA-2 field time", mf.TIME_UNITS, "MERRA-2", continuous,
                "nearest available hourly field; exact ties choose earlier", "time")),
            "MRMS_time_difference_minutes": (("scan",), mrms_dt, _variable_attrs(
                "MRMS selected time minus GMI scan time", "minutes", "derived collocation diagnostic",
                "not applicable", "selected MRMS time minus native GMI scan time", "MRMS_valid_time - scan_time")),
            "RAQI_time_difference_minutes": (("scan",), raqi_dt, _variable_attrs(
                "RAQI selected time minus GMI scan time", "minutes", "derived collocation diagnostic",
                "not applicable", "selected RAQI time minus native GMI scan time", "RAQI_time - scan_time")),
            "ERA5_time_difference_minutes": (("scan",), era_dt, _variable_attrs(
                "ERA5 selected time minus GMI scan time", "minutes", "derived collocation diagnostic",
                "not applicable", "selected ERA5 time minus native GMI scan time", "ERA5_time - scan_time")),
            "MERRA2_time_difference_minutes": (("scan",), merra_dt, _variable_attrs(
                "MERRA-2 selected time minus GMI scan time", "minutes", "derived collocation diagnostic",
                "not applicable", "selected MERRA-2 time minus native GMI scan time", "MERRA2_time - scan_time")),
            "mrms_domain_flag": (("scan", "pixel"), mrms_domain, {**_variable_attrs(
                "Footprint center lies inside the native MRMS source-grid center-coordinate bounding box", "1", "MRMS grid geometry",
                "native-source-grid center-coordinate bounding-box test only; not a collocated-data validity test",
                "not applicable", "native MRMS first/last grid-point coordinates"),
                "flag_values": np.array([0, 1], dtype=np.uint8), "flag_meanings": "outside_domain inside_domain",
                "missing_value_handling": flag_missing}),
            "mrms_valid_flag": (("scan", "pixel"), mrms_valid, {**_variable_attrs(
                "MRMS_Pass2 value is finite after ERA5-grid harmonization and footprint sampling", "1",
                "MRMS MultiSensor_QPE_01H_Pass2", continuous,
                "actual successfully loaded MRMS valid time", "isfinite(MRMS_Pass2)"),
                "flag_values": np.array([0, 1], dtype=np.uint8), "flag_meanings": "invalid valid",
                "missing_value_handling": flag_missing}),
            "raqi_valid_flag": (("scan", "pixel"), raqi_valid, {**_variable_attrs(
                "RAQI value is finite after spatial harmonization and footprint sampling", "1", "Reconstructed RAQI",
                continuous, "identical selected valid hour as MRMS", "RAQI"),
                "flag_values": np.array([0, 1], dtype=np.uint8), "flag_meanings": "invalid valid",
                "missing_value_handling": flag_missing}),
            "autosnow_valid_flag": (("scan", "pixel"), autosnow_valid, {**_variable_attrs(
                "AutoSnow class is valid", "1", "GMASI AutoSnow v003",
                "majority mode on ERA5 grid then footprint-center sampling", "actual UTC scan date", "autosnow_class"),
                "flag_values": np.array([0, 1], dtype=np.uint8), "flag_meanings": "invalid valid",
                "missing_value_handling": flag_missing}),
            "valid_reference_flag": (("scan", "pixel"), valid_reference, {**_variable_attrs(
                "Both MRMS and RAQI references are finite", "1", "derived collocation diagnostic",
                continuous, "common MRMS and RAQI valid hour", "isfinite(MRMS_Pass2) AND isfinite(RAQI)"),
                "flag_values": np.array([0, 1], dtype=np.uint8), "flag_meanings": "invalid valid",
                "missing_value_handling": flag_missing}),
        },
        coords={
            "scan": ("scan", np.arange(shape[0], dtype=np.uint32), {"long_name": "native GMI V7 scan index",
                "units": "1", "source_dataset": "GPROF V7", "source_variable": "generated scan index",
                "spatial_harmonization_method": native_space, "temporal_matching_method": native_time,
                "missing_value_handling": index_missing}),
            "pixel": ("pixel", np.arange(shape[1], dtype=np.uint16), {"long_name": "native GMI V7 footprint index within scan",
                "units": "1", "source_dataset": "GPROF V7", "source_variable": "generated pixel index",
                "spatial_harmonization_method": native_space, "temporal_matching_method": native_time,
                "missing_value_handling": index_missing}),
        },
        attrs={
            "title": "GMI GPROF V7/V8 collocation with MRMS, RAQI, ERA5, MERRA-2, and AutoSnow",
            "sensor_configuration": config["config_name"], "sensor": config["sensor"], "platform": config["platform"],
            "source_gprof_v7_file": pair["v7_file"], "source_gprof_v8_file": pair["v8_file"],
            "source_mrms_files": "; ".join(sorted(sources["mrms"])), "source_raqi_files": "; ".join(sorted(sources["raqi"])),
            "source_era5_files": "; ".join(sorted(sources["era5"] or {first_era_path})),
            "source_merra2_files": "; ".join(sorted(sources["merra2"])), "source_autosnow_files": "; ".join(sorted(sources["autosnow"])),
            "common_grid_source": first_era_path, "v7_v8_pairing_method": "filename UTC date/start/end plus orbit number",
            "collocation_method": "Ancillary grids harmonized in memory to actual ERA5 coordinates, then nearest ERA5 cell sampled at native V7 footprint centers.",
            "mrms_raqi_time_matching": "IMERG-inherited nearest available valid time to each scan; identical T for MRMS and RAQI; exact half-hour ties choose earlier.",
            "edge_handling": "Each actual ScanTime selects its ancillary hour/date; previous/current/next dates are opened on demand; scans are neither duplicated nor clipped.",
            "orbit_crossed_midnight": str(crossed_midnight).lower(), "geometry_qc_status": qc["classification"],
            "geometry_qc_summary": qc["note"], "ancillary_gaps": "; ".join(gaps),
            "temporal_conventions": "MRMS and ERA5 use nearest hour with exact half-hour ties earlier; RAQI uses identical MRMS hour; MERRA-2 uses nearest available hourly field; AutoSnow uses actual UTC scan date.",
            "spatial_harmonization": "MRMS, RAQI, and MERRA-2 arithmetic-mean resampling to native ERA5; AutoSnow majority mode; ERA5 retained native; nearest ERA5 center sampled to V7 footprints.",
            "creation_time_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "software_script": SCRIPT, "code_version": CODE_VERSION,
        },
    )
    if mrms_meta is not None:
        south, north, west, east = mrms_bounds
        dx = (east - west) / (int(mrms_meta["Ni"]) - 1)
        dy = (north - south) / (int(mrms_meta["Nj"]) - 1)
        ds.attrs.update({
            "mrms_domain_flag_definition": (
                "Native MRMS source-grid center-coordinate bounding-box test only; "
                "mrms_valid_flag is the authoritative collocated-value validity test."
            ),
            "mrms_native_center_south": south, "mrms_native_center_north": north,
            "mrms_native_center_west": west, "mrms_native_center_east": east,
            "mrms_native_edge_south": south - dy / 2.0,
            "mrms_native_edge_north": north + dy / 2.0,
            "mrms_native_edge_west": west - dx / 2.0,
            "mrms_native_edge_east": east + dx / 2.0,
            "mrms_iScansNegatively": int(mrms_meta["iScansNegatively"]),
            "mrms_jScansPositively": int(mrms_meta["jScansPositively"]),
            "mrms_jPointsAreConsecutive": int(mrms_meta["jPointsAreConsecutive"]),
            "mrms_alternativeRowScanning": int(mrms_meta["alternativeRowScanning"]),
            "mrms_grib_orientation": (
                "values.reshape(Nj, Ni): rows scan north-to-south; columns scan west-to-east; "
                "adjacent points are along i; rows do not alternate."
            ),
        })
    if mrms_harmonized_bounds is not None:
        south, north, west, east = mrms_harmonized_bounds
        ds.attrs.update({
            "mrms_harmonized_finite_center_south": south,
            "mrms_harmonized_finite_center_north": north,
            "mrms_harmonized_finite_center_west": west,
            "mrms_harmonized_finite_center_east": east,
        })
    mf.atomic_write_netcdf(ds, final)
    return {"status": "ok_with_ancillary_gaps" if gaps else "ok", "sensor": config["config_name"], "orbit_id": pair["orbit_id"],
            "output": str(final), "geometry_qc": qc, "scan_start": scan_dt[0], "scan_end": scan_dt[-1],
            "crossed_midnight": crossed_midnight,
            "mrms_raqi_hours": sorted(set(selected_hours)), "era5_hours": sorted(set(selected_hours)),
            "merra2_hours": sorted(set(selected_merra)), "gaps": gaps}


def parse_dates(text):
    if not text:
        return None
    return {dt.datetime.strptime(item.strip(), "%Y-%m-%d").date() for item in text.split(",")}


def inclusive_date_range(start, end):
    if start > end:
        raise ValueError("start date must be on or before end date")
    return {start + dt.timedelta(days=offset) for offset in range((end - start).days + 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor", choices=["GMI"], default="GMI",
                        help="Phase-1 production is intentionally restricted to GMI")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--dates", help="comma-separated YYYY-MM-DD; default is the full common V7/V8 archive")
    parser.add_argument("--start-date", type=dt.date.fromisoformat,
                        help="inclusive YYYY-MM-DD discovery boundary; requires --end-date")
    parser.add_argument("--end-date", type=dt.date.fromisoformat,
                        help="inclusive YYYY-MM-DD discovery boundary; requires --start-date")
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--orbit-id", action="append")
    parser.add_argument("--output-root", default=str(mf.DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--discovery-only", action="store_true")
    parser.add_argument("--geometry-tolerance-deg", type=float, default=5e-3)
    for key, default in mf.ANCILLARY_ROOTS.items():
        parser.add_argument(f"--{key.replace('merra2','merra2').replace('_','-')}-dir", default=default)
    args = parser.parse_args()
    roots = {key: getattr(args, f"{key}_dir") for key in mf.ANCILLARY_ROOTS}
    if args.dates and (args.start_date or args.end_date):
        parser.error("--dates cannot be combined with --start-date/--end-date")
    if bool(args.start_date) != bool(args.end_date):
        parser.error("--start-date and --end-date must be supplied together")
    try:
        requested = (inclusive_date_range(args.start_date, args.end_date)
                     if args.start_date else parse_dates(args.dates))
    except ValueError as exc:
        parser.error(str(exc))
    allowed = requested
    sensors = list(mf.SENSOR_CONFIGS) if args.sensor == "all" else [args.sensor]
    all_jobs, summaries = [], {}
    for name in sensors:
        config = mf.sensor_config(name); config["config_name"] = name
        discovery = mf.discover_pairs(config, allowed)
        pairs = discovery.pop("pairs")
        if args.orbit_id:
            pairs = [pair for pair in pairs if pair["orbit_id"] in args.orbit_id]
        if args.max_files is not None:
            pairs = pairs[:args.max_files]
        existing_outputs = sum(output_path(config, pair, args.output_root).exists() for pair in pairs)
        summaries[name] = {**discovery, "selected_pairs": len(pairs), "v7_directory": config["v7_root"],
                           "v8_directory": config["v8_root"], "common_dates": sorted({p["date"] for p in pairs}),
                           "earliest_common_acquisition_date": min((p["date"] for p in pairs), default=None),
                           "latest_common_acquisition_date": max((p["date"] for p in pairs), default=None),
                           "existing_output_count": existing_outputs,
                           "remaining_to_process": len(pairs) - existing_outputs}
        all_jobs.extend((pair, config) for pair in pairs)
        if args.discovery_only:
            concise = {key: summaries[name][key] for key in (
                "v7_count", "v8_count", "matched_count", "earliest_common_acquisition_date",
                "latest_common_acquisition_date", "existing_output_count", "remaining_to_process",
            )}
            print(json.dumps({name: concise}, default=mf.json_ready))
        else:
            print(json.dumps({name: summaries[name]}, default=mf.json_ready))
    out_root = Path(args.output_root)
    if args.discovery_only:
        mf.write_json(out_root / "qc" / "discovery_summary.json", summaries)
        return
    results = []
    kwargs = dict(roots=roots, output_root=args.output_root, overwrite=args.overwrite,
                  geometry_tolerance=args.geometry_tolerance_deg)
    if args.workers == 1:
        for pair, config in all_jobs:
            try: results.append(process_pair(pair, config, **kwargs))
            except Exception as exc: results.append({"status": "failed", "sensor": config["config_name"],
                                                     "orbit_id": pair["orbit_id"], "error": str(exc), "traceback": traceback.format_exc()})
            print(json.dumps(results[-1], default=mf.json_ready))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_pair, pair, config, **kwargs): (pair, config) for pair, config in all_jobs}
            for future in as_completed(futures):
                pair, config = futures[future]
                try: result = future.result()
                except Exception as exc: result = {"status": "failed", "sensor": config["config_name"],
                                                   "orbit_id": pair["orbit_id"], "error": str(exc), "traceback": traceback.format_exc()}
                results.append(result); print(json.dumps(result, default=mf.json_ready))
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    mf.write_json(out_root / "logs" / f"collocation_run_{stamp}.json", {"discovery": summaries, "results": results})


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Collocate GMI GPROF V7/V8 with interval-aware precipitation products.

The primary output remains one native-GMI-footprint NetCDF file per orbit.
All implementation and outputs are isolated from the legacy production flow.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import xarray as xr

import my_functions_GPROF_V7_V8_MRMS_RAQI_ERA5_MERRA2_AutoSnow_StageIV_20260824 as mf

SCRIPT = Path(__file__).name
CODE_VERSION = "20260907.1_ERA5_interval_containment"


def output_path(config, pair, output_root):
    return Path(output_root) / "data" / "matchups" / config["config_name"] / (
        f"{config['config_name']}_{mf.DATASET_TAG}_{pair['orbit_id']}.nc"
    )


def _attrs(long_name, units, source, spatial, temporal, source_variable):
    return {
        "long_name": long_name,
        "units": units,
        "source_dataset": source,
        "source_variable": source_variable,
        "spatial_harmonization_method": spatial,
        "temporal_matching_method": temporal,
        "missing_value_handling": "Unavailable or invalid continuous values decode as NaN.",
    }


def _seconds(value):
    return (value - mf.EPOCH).total_seconds()


def process_pair(pair, config, roots, land_mask_path, output_root, overwrite=False,
                 geometry_tolerance=5e-3):
    final = output_path(config, pair, output_root)
    partial = Path(str(final) + ".partial")
    if final.exists() and not overwrite:
        return {"status": "skipped_existing", "orbit_id": pair["orbit_id"], "output": str(final)}
    if partial.exists():
        partial.unlink()

    v7 = mf.read_gprof(pair["v7_file"], config)
    v8 = mf.read_gprof(pair["v8_file"], config)
    geometry = mf.geometry_qc(v7, v8, geometry_tolerance)
    if not geometry["passed"]:
        return {"status": "failed_geometry", "orbit_id": pair["orbit_id"], "geometry_qc": geometry}

    lat = v7["lat"]
    lon = v7["lon"]
    scan_dt = v7["scan_datetimes"]
    scan_seconds = v7["scan_seconds"]
    shape = lat.shape
    first_era, era_lat, era_lon, first_era_path, _ = mf.read_era5_hour(
        roots["era5"], mf.nearest_hour(scan_dt[0])
    )
    del first_era

    continuous_names = (
        "MRMS_Pass2", "RAQI", "ERA5_precipitation", "MERRA2_T2M",
        "MERRA2_T2MWET", "StageIV",
    )
    fields = {name: np.full(shape, np.nan, np.float32) for name in continuous_names}
    autosnow = np.full(shape, mf.AUTOSNOW_FILL, np.uint8)
    land_mask = mf.sample_imerg_land_mask(land_mask_path, lat, lon)

    time_names = (
        "mrms_end", "mrms_start", "raqi_end", "era", "merra",
        "stage_end", "stage_start",
    )
    times = {name: np.full(len(scan_dt), np.nan, np.float64) for name in time_names}
    stage_status = np.full(len(scan_dt), np.nan, np.float32)
    stage_source_count = np.zeros(shape, np.uint32)
    sources = {name: set() for name in roots}
    gaps = []
    requested_stage_hours = set()
    found_stage_hours = set()
    invalid_stage_hours = set()
    mrms_bounds = (1.0, 0.0, 0.0, 0.0)

    # ERA5 total precipitation is the accumulation over the hour ending at its
    # validity time. Select the unique interval satisfying start < scan <= end,
    # consistent with the MRMS and Stage IV temporal convention.
    selected_era = np.array(
        [mf.accumulation_end_hour(value) for value in scan_dt], dtype=object
    )
    for hour in sorted(set(selected_era)):
        rows = np.flatnonzero(selected_era == hour)
        try:
            data, slat, slon, path, actual = mf.read_era5_hour(roots["era5"], hour)
            fields["ERA5_precipitation"][rows] = mf.sample_era_grid(
                data, slat, slon, lat[rows], lon[rows]
            )
            times["era"][rows] = actual
            sources["era5"].add(path)
        except Exception as exc:
            gaps.append(f"ERA5 {hour.isoformat()}: {exc}")

    # MRMS 01H and its RAQI use the unique interval satisfying start < scan <= end.
    selected_accum = np.array([mf.accumulation_end_hour(value) for value in scan_dt], dtype=object)
    for end in sorted(set(selected_accum)):
        rows = np.flatnonzero(selected_accum == end)
        start = end - dt.timedelta(hours=1)
        path = mf.mrms_file(roots["mrms"], end)
        if path is None:
            gaps.append(f"MRMS accumulation unavailable ({start.isoformat()}, {end.isoformat()}]")
        else:
            try:
                data, slat, slon, meta = mf.read_mrms(path)
                mrms_bounds = (
                    float(np.nanmin(slat)), float(np.nanmax(slat)),
                    float(np.nanmin(slon)), float(np.nanmax(slon)),
                )
                on_grid = mf.resample_to_era5(data, slat, slon, era_lat, era_lon)
                fields["MRMS_Pass2"][rows] = mf.sample_era_grid(
                    on_grid, era_lat, era_lon, lat[rows], lon[rows]
                )
                actual_end = float(meta["valid_time_seconds"])
                times["mrms_end"][rows] = actual_end
                times["mrms_start"][rows] = actual_end - 3600.0
                sources["mrms"].add(path)
            except Exception as exc:
                gaps.append(f"MRMS ({start.isoformat()}, {end.isoformat()}]: {exc}")
        try:
            data, slat, slon, path, actual = mf.read_raqi(roots["raqi"], end)
            on_grid = mf.resample_to_era5(data, slat, slon, era_lat, era_lon)
            fields["RAQI"][rows] = mf.sample_era_grid(
                on_grid, era_lat, era_lon, lat[rows], lon[rows]
            )
            times["raqi_end"][rows] = actual
            sources["raqi"].add(path)
        except Exception as exc:
            gaps.append(f"RAQI ({start.isoformat()}, {end.isoformat()}]: {exc}")

    # MERRA-2 tavg1 fields are centered at :30; both temperatures use the same
    # time and native-grid nearest-neighbor footprint mapping.
    selected_merra = np.array([mf.nearest_merra2_time(value) for value in scan_dt], dtype=object)
    for hour in sorted(set(selected_merra)):
        rows = np.flatnonzero(selected_merra == hour)
        try:
            t2m, wet, slat, slon, path, _, _, actual = mf.read_merra2_temperatures(
                roots["merra2"], hour
            )
            fields["MERRA2_T2M"][rows] = mf.sample_regular_nearest(
                t2m, slat, slon, lat[rows], lon[rows]
            )
            fields["MERRA2_T2MWET"][rows] = mf.sample_regular_nearest(
                wet, slat, slon, lat[rows], lon[rows]
            )
            times["merra"][rows] = actual
            sources["merra2"].add(path)
        except Exception as exc:
            gaps.append(f"MERRA2 {hour.isoformat()}: {exc}")

    scan_dates = np.array([value.date() for value in scan_dt], dtype=object)
    for date in sorted(set(scan_dates)):
        rows = np.flatnonzero(scan_dates == date)
        try:
            data, slat, slon, path = mf.read_autosnow(roots["autosnow"], date)
            on_grid = mf.resample_to_era5(
                data, slat, slon, era_lat, era_lon, categorical=True
            )
            autosnow[rows] = mf.sample_era_grid(
                on_grid, era_lat, era_lon, lat[rows], lon[rows]
            )
            sources["autosnow"].add(path)
        except Exception as exc:
            gaps.append(f"AutoSnow {date.isoformat()}: {exc}")

    # Stage IV annual files are lazy-opened, and only requested hourly slices load.
    readers = {}
    try:
        for end in sorted(set(selected_accum)):
            rows = np.flatnonzero(selected_accum == end)
            requested_stage_hours.add(end)
            try:
                if end.year not in readers:
                    path = mf.stageiv_file(roots["stageiv"], end.year)
                    if path is None:
                        raise FileNotFoundError(f"missing Stage IV year {end.year}")
                    readers[end.year] = mf.StageIVAnnualReader(path, era_lat, era_lon)
                    sources["stageiv"].add(path)
                grid, actual_end, actual_start, status, counts = readers[end.year].read_hour(end)
                fields["StageIV"][rows] = mf.sample_era_grid(
                    grid, era_lat, era_lon, lat[rows], lon[rows]
                )
                stage_source_count[rows] = mf.sample_era_grid(
                    counts, era_lat, era_lon, lat[rows], lon[rows]
                )
                times["stage_end"][rows] = actual_end
                times["stage_start"][rows] = actual_start
                stage_status[rows] = status
                found_stage_hours.add(end)
            except Exception as exc:
                invalid_stage_hours.add(end)
                gaps.append(f"StageIV ending {end.isoformat()}: {exc}")
    finally:
        for reader in readers.values():
            reader.close()

    phase = mf.phase_regime(fields["MERRA2_T2M"], fields["MERRA2_T2MWET"])
    mrms_domain, mrms_valid, raqi_valid, autosnow_valid, valid_reference = mf.quality_flags(
        fields["MRMS_Pass2"], fields["RAQI"], autosnow, lat, lon, mrms_bounds
    )
    stage_valid = np.isfinite(fields["StageIV"]).astype(np.uint8)
    stage_interval_valid = (
        np.isfinite(times["stage_start"]) & np.isfinite(times["stage_end"])
        & (times["stage_start"] < scan_seconds) & (scan_seconds <= times["stage_end"])
    ).astype(np.uint8)

    native = "native GMI V7 scan/footprint geometry"
    mean025 = "Mean of valid high-resolution source-cell centers assigned to each native ERA5 0.25-degree cell; nearest cell sampled at footprint center."
    nearest_native = "Nearest native MERRA-2 grid-cell center sampled at footprint center; no spatial smoothing."
    interval = "unique hourly accumulation interval with start < GMI ScanTime <= end"
    vars = {
        "latitude": (("scan", "pixel"), lat, _attrs("GMI footprint latitude", "degrees_north", "GPROF V7", native, "native scan time", "S1/Latitude")),
        "longitude": (("scan", "pixel"), lon, _attrs("GMI footprint longitude", "degrees_east", "GPROF V7", native, "native scan time", "S1/Longitude")),
        "scan_time": (("scan",), scan_seconds, _attrs("GMI scan time", mf.TIME_UNITS, "GPROF V7", native, "native", "S1/ScanTime")),
        "surfacePrecipitation_V7": (("scan", "pixel"), v7["precip"], _attrs("GPROF V7 surface precipitation", v7["precip_units"], "GPROF V7", native, "native", "S1/surfacePrecipitation")),
        "surfacePrecipitation_V8": (("scan", "pixel"), v8["precip"], _attrs("GPROF V8 surface precipitation", v8["precip_units"], "GPROF V8", "V7/V8 geometry verified", "paired within one second", "S1/surfacePrecipitation")),
        "MRMS_Pass2": (("scan", "pixel"), fields["MRMS_Pass2"], _attrs("MRMS 01H Pass2 accumulation", "mm", "MRMS", mean025, interval, "MultiSensor_QPE_01H_Pass2")),
        "RAQI": (("scan", "pixel"), fields["RAQI"], _attrs("Reconstructed radar accumulation quality index", "1", "RAQI", mean025, "same interval end as MRMS", "RAQI")),
        "ERA5_precipitation": (("scan", "pixel"), fields["ERA5_precipitation"], _attrs("ERA5 hourly total precipitation", "mm", "ERA5", "nearest native ERA5 cell", interval, "tp")),
        "MERRA2_T2M": (("scan", "pixel"), fields["MERRA2_T2M"], _attrs("MERRA-2 2-m air temperature", "K", "MERRA-2", nearest_native, "nearest tavg1 center (:30); ties earlier", "T2M")),
        "MERRA2_T2MWET": (("scan", "pixel"), fields["MERRA2_T2MWET"], _attrs("MERRA-2 2-m wet-bulb temperature", "K", "MERRA-2", nearest_native, "same time as MERRA2_T2M", "T2MWET")),
        "AutoSnow": (("scan", "pixel"), autosnow, {**_attrs("GMASI AutoSnow class", "1", "AutoSnow", "mode on 0.25-degree grid", "actual UTC scan date", "autosnow_class"), "missing_value": int(mf.AUTOSNOW_FILL), "flag_values": np.array([0, 1, 2, 3], np.uint8), "flag_meanings": "clear_water snow_free_land snow_covered_land ice_covered_water"}),
        "StageIV": (("scan", "pixel"), fields["StageIV"], _attrs("NOAA Stage IV preceding-hour precipitation accumulation", "mm", "IEM-packaged NOAA Stage IV", mean025, interval + "; explicit time_bnds verified", "p01m")),
        "StageIV_source_cell_count": (("scan", "pixel"), stage_source_count, _attrs("Stage IV source centers contributing to sampled 0.25-degree cell", "1", "derived", mean025, interval, "count")),
        "land_mask": (("scan", "pixel"), land_mask, {**_attrs("Static IMERG binary land mask", "1", "IMERG static land-sea mask", "nearest native mask cell", "static", "landseamask < 25"), "flag_values": np.array([0, 1], np.uint8), "flag_meanings": "water land"}),
        "phase_regime": (("scan", "pixel"), phase, {**_attrs("Provisional temperature phase regime", "1", "derived from raw MERRA-2 fields", "not applicable", "same MERRA-2 field", "T2M and T2MWET thresholds"), "flag_values": np.array([0, 1, 2], np.uint8), "flag_meanings": "transition_or_ambiguous high_confidence_snow high_confidence_rain"}),
    }
    time_specs = {
        "MRMS_interval_start": "mrms_start", "MRMS_interval_end": "mrms_end",
        "RAQI_interval_end": "raqi_end", "ERA5_time": "era", "MERRA2_time": "merra",
        "StageIV_interval_start": "stage_start", "StageIV_interval_end": "stage_end",
    }
    for name, key in time_specs.items():
        vars[name] = (("scan",), times[key], _attrs(name.replace("_", " "), mf.TIME_UNITS, "source/derived timing", "not applicable", interval if "interval" in name else "dataset-specific", "time coordinate or bounds"))
    vars.update({
        "StageIV_status": (("scan",), stage_status, _attrs("Stage IV source status", "1", "Stage IV", "not applicable", interval, "p01m_status")),
        "MRMS_time_difference_minutes": (("scan",), mf.time_difference_minutes(times["mrms_end"], scan_seconds), _attrs("MRMS interval end minus scan time", "minutes", "derived", "not applicable", interval, "MRMS_interval_end - scan_time")),
        "ERA5_time_difference_minutes": (("scan",), mf.time_difference_minutes(times["era"], scan_seconds), _attrs("ERA5 interval end minus scan time", "minutes", "derived", "not applicable", interval, "ERA5_time - scan_time")),
        "MERRA2_time_difference_minutes": (("scan",), mf.time_difference_minutes(times["merra"], scan_seconds), _attrs("MERRA-2 center minus scan time", "minutes", "derived", "not applicable", "nearest tavg1 center", "MERRA2_time - scan_time")),
        "StageIV_time_difference_minutes": (("scan",), mf.time_difference_minutes(times["stage_end"], scan_seconds), _attrs("Stage IV interval end minus scan time", "minutes", "derived", "not applicable", interval, "StageIV_interval_end - scan_time")),
        "mrms_domain_flag": (("scan", "pixel"), mrms_domain),
        "mrms_valid_flag": (("scan", "pixel"), mrms_valid),
        "raqi_valid_flag": (("scan", "pixel"), raqi_valid),
        "autosnow_valid_flag": (("scan", "pixel"), autosnow_valid),
        "valid_reference_flag": (("scan", "pixel"), valid_reference),
        "stageiv_valid_flag": (("scan", "pixel"), stage_valid),
        "stageiv_interval_valid_flag": (("scan",), stage_interval_valid),
    })
    ds = xr.Dataset(
        data_vars=vars,
        coords={"scan": np.arange(shape[0], dtype=np.uint32), "pixel": np.arange(shape[1], dtype=np.uint16)},
        attrs={
            "title": "Footprint-level GMI GPROF V7/V8, MRMS, RAQI, ERA5, MERRA-2, AutoSnow, and Stage IV collocation",
            "experiment_id": mf.EXPERIMENT_ID,
            "source_gprof_v7_file": pair["v7_file"], "source_gprof_v8_file": pair["v8_file"],
            **{f"source_{name}_files": "; ".join(sorted(paths)) for name, paths in sources.items()},
            "hourly_accumulation_boundary_convention": "(start, end]",
            "mrms_raqi_time_matching": interval,
            "stageiv_time_matching": interval + "; source time_bnds are authoritative and explicitly checked",
            "temporal_conventions": "MRMS/RAQI, Stage IV, and ERA5 precipitation use (start,end] containment; MERRA-2 retains nearest :30-centered tavg1 field; AutoSnow uses UTC date.",
            "spatial_harmonization": "MRMS and Stage IV mean to 0.25-degree ERA5 cells; RAQI mean; MERRA-2 nearest native cell; AutoSnow mode; all values then sampled at native GMI footprint centers.",
            "primary_evaluation_sample": "land_mask == 1",
            "phase_definition": "snow: T2M < 275.15 K and T2MWET <= 273.15 K; rain: T2MWET >= 275.15 K; otherwise transition/ambiguous",
            "ancillary_gaps": "; ".join(gaps), "geometry_qc_summary": geometry["note"],
            "creation_time_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "software_script": SCRIPT, "code_version": CODE_VERSION,
        },
    )
    mf.atomic_write_netcdf(ds, final)
    finite_stage = int(np.isfinite(fields["StageIV"]).sum())
    return {
        "status": "ok_with_ancillary_gaps" if gaps else "ok", "orbit_id": pair["orbit_id"],
        "output": str(final), "scan_start": scan_dt[0], "scan_end": scan_dt[-1],
        "footprints": int(np.prod(shape)), "stageiv_matched_footprints": finite_stage,
        "stageiv_match_fraction": finite_stage / int(np.prod(shape)),
        "stageiv_requested_hours": len(requested_stage_hours),
        "stageiv_found_hours": len(found_stage_hours),
        "stageiv_invalid_or_missing_hours": len(invalid_stage_hours),
        "gaps": gaps, "geometry_qc": geometry,
    }


def inclusive_date_range(start, end):
    if start > end:
        raise ValueError("start date must be on or before end date")
    return {start + dt.timedelta(days=i) for i in range((end - start).days + 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor", choices=["GMI"], default="GMI")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--start-date", type=dt.date.fromisoformat)
    parser.add_argument("--end-date", type=dt.date.fromisoformat)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--orbit-id", action="append")
    parser.add_argument("--output-root", default=str(mf.DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--land-mask-file", default=str(mf.DEFAULT_LAND_MASK))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--discovery-only", action="store_true")
    parser.add_argument("--geometry-tolerance-deg", type=float, default=5e-3)
    for key, default in mf.ANCILLARY_ROOTS.items():
        parser.add_argument(f"--{key}-dir", default=default)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if bool(args.start_date) != bool(args.end_date):
        parser.error("--start-date and --end-date must be supplied together")
    roots = {key: getattr(args, f"{key}_dir") for key in mf.ANCILLARY_ROOTS}
    available_years = mf.readable_stageiv_years(roots["stageiv"])
    if not available_years:
        parser.error(f"no readable annual Stage IV files in {roots['stageiv']}")
    start = args.start_date or dt.date(min(available_years), 1, 1)
    end = args.end_date or dt.date(max(available_years), 12, 31)
    try:
        allowed = inclusive_date_range(start, end)
    except ValueError as exc:
        parser.error(str(exc))
    missing_years = sorted(set(range(start.year, end.year + 1)) - set(available_years))
    if missing_years:
        parser.error(
            f"requested range {start}..{end} includes unreadable/unavailable Stage IV year(s) "
            f"{missing_years}; readable years are {available_years}"
        )
    config = mf.sensor_config(args.sensor)
    config["config_name"] = args.sensor
    discovery = mf.discover_pairs(config, allowed)
    pairs = discovery.pop("pairs")
    if args.orbit_id:
        pairs = [pair for pair in pairs if pair["orbit_id"] in args.orbit_id]
    if args.max_files is not None:
        pairs = pairs[:args.max_files]
    if not pairs:
        parser.error(f"no common GPROF V7/V8 pairs found for {start}..{end}")
    summary = {
        **discovery, "requested_start_date": start, "requested_end_date": end,
        "readable_stageiv_years": available_years, "selected_pairs": len(pairs),
        "existing_outputs": sum(output_path(config, p, args.output_root).exists() for p in pairs),
    }
    print(json.dumps(summary, default=mf.json_ready))
    out_root = Path(args.output_root)
    for directory in ("data", "tables", "figures", "logs", "qc"):
        (out_root / directory).mkdir(parents=True, exist_ok=True)
    if args.discovery_only:
        mf.write_json(out_root / "qc" / f"discovery_{start:%Y%m%d}_{end:%Y%m%d}.json", summary)
        return
    kwargs = {
        "roots": roots, "land_mask_path": args.land_mask_file,
        "output_root": args.output_root, "overwrite": args.overwrite,
        "geometry_tolerance": args.geometry_tolerance_deg,
    }
    results = []
    if args.workers == 1:
        for pair in pairs:
            try:
                result = process_pair(pair, config, **kwargs)
            except Exception as exc:
                result = {"status": "failed", "orbit_id": pair["orbit_id"], "error": str(exc), "traceback": traceback.format_exc()}
            results.append(result)
            print(json.dumps(result, default=mf.json_ready))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_pair, p, config, **kwargs): p for p in pairs}
            for future in as_completed(futures):
                pair = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"status": "failed", "orbit_id": pair["orbit_id"], "error": str(exc), "traceback": traceback.format_exc()}
                results.append(result)
                print(json.dumps(result, default=mf.json_ready))
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = {"summary": summary, "results": results}
    mf.write_json(out_root / "logs" / f"collocation_{start:%Y%m%d}_{end:%Y%m%d}_{stamp}.json", run)
    qc_rows = [
        {
            "orbit_id": r.get("orbit_id"), "status": r.get("status"),
            "footprints": r.get("footprints"),
            "stageiv_matched_footprints": r.get("stageiv_matched_footprints"),
            "stageiv_match_fraction": r.get("stageiv_match_fraction"),
            "stageiv_requested_hours": r.get("stageiv_requested_hours"),
            "stageiv_found_hours": r.get("stageiv_found_hours"),
            "stageiv_invalid_or_missing_hours": r.get("stageiv_invalid_or_missing_hours"),
        } for r in results
    ]
    import csv
    qc_path = out_root / "qc" / f"qc_{mf.DATASET_TAG}_{start:%Y%m%d}_{end:%Y%m%d}.csv"
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    with qc_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(qc_rows[0]))
        writer.writeheader(); writer.writerows(qc_rows)


if __name__ == "__main__":
    main()

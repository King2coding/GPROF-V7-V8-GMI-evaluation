#!/usr/bin/env python3
"""Small read-only diagnostic for V7/V8 geometry and retrieval validity."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


def read_scan_keys(group: h5py.Group) -> list[tuple[int, ...]]:
    names = ("Year", "Month", "DayOfMonth", "Hour", "Minute", "Second")
    values = [np.asarray(group[name][:], dtype=np.int64) for name in names]
    if "MilliSecond" in group:
        values.append(np.asarray(group["MilliSecond"][:], dtype=np.int64))
    else:
        values.append(np.zeros_like(values[0]))
    return list(zip(*values))


def read_product(path: str) -> dict[str, object]:
    with h5py.File(path, "r") as handle:
        group = handle["S1"]
        latitude = np.asarray(group["Latitude"][:], dtype=np.float64)
        longitude = np.asarray(group["Longitude"][:], dtype=np.float64)
        variable = group["surfacePrecipitation"]
        precipitation = np.asarray(variable[:], dtype=np.float64)
        fill = float(np.asarray(variable.attrs.get("_FillValue", -9999.9)).reshape(-1)[0])
        precipitation[(precipitation == fill) | (precipitation < 0) | ~np.isfinite(precipitation)] = np.nan
        scans = read_scan_keys(group["ScanTime"])
    return {
        "latitude": latitude,
        "longitude": longitude,
        "precipitation": precipitation,
        "scans": scans,
    }


def parse_file(path: Path) -> dict[str, object] | None:
    parts = path.name.split(".")
    if len(parts) < 6:
        return None
    match = re.fullmatch(r"(\d{8})-S(\d{6})-E(\d{6})", parts[4])
    if match is None:
        return None
    ymd, start_hms, end_hms = match.groups()
    return {
        "date": dt.datetime.strptime(ymd, "%Y%m%d").date(),
        "key": f"{ymd}-S{start_hms}-E{end_hms}.{parts[5]}",
        "orbit_id": f"{parts[5]}_{ymd}",
        "path": str(path),
    }


def discover_pairs(v7_root: Path, v8_root: Path) -> list[dict[str, object]]:
    def index(root: Path, pattern: str) -> dict[str, dict[str, object]]:
        result = {}
        for path in root.glob(pattern):
            info = parse_file(path)
            if info is not None:
                result[str(info["key"])] = info
        return result

    v7 = index(v7_root, "*.HDF5")
    v8 = index(v8_root, "*.nc")
    return [
        {
            "orbit_id": v7[key]["orbit_id"],
            "v7_file": v7[key]["path"],
            "v8_file": v8[key]["path"],
        }
        for key in sorted(set(v7) & set(v8))
    ]


def align_scans(v7_keys, v8_keys) -> tuple[np.ndarray, np.ndarray]:
    v8_lookup = {key: index for index, key in enumerate(v8_keys)}
    matched = [(index, v8_lookup[key]) for index, key in enumerate(v7_keys) if key in v8_lookup]
    if not matched:
        return np.array([], dtype=int), np.array([], dtype=int)
    return tuple(np.asarray(values, dtype=int) for values in zip(*matched))


def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.deg2rad(lat1)
    lat2 = np.deg2rad(lat2)
    dlat = lat2 - lat1
    dlon = np.deg2rad(((lon2 - lon1 + 180.0) % 360.0) - 180.0)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def choose_pairs(all_pairs, results, per_group):
    by_orbit = {pair["orbit_id"]: pair for pair in all_pairs}
    classes = {}
    for result in results:
        geometry = result.get("geometry_qc") or {}
        classes[result.get("orbit_id")] = geometry.get("classification", result.get("status"))
    groups = {}
    for orbit_id, classification in classes.items():
        pair = by_orbit.get(orbit_id)
        if pair is not None:
            groups.setdefault(classification, []).append(pair)
    selected = []
    for classification, pairs in sorted(groups.items()):
        if classification in {"PASS_NORMAL", "PASS_MINOR_EDGE_WARNING", "FAIL_TRUE_GEOMETRY_MISMATCH", "FAIL_SHAPE_OR_TIME_MISMATCH"}:
            step = max(1, len(pairs) // per_group)
            selected.extend((classification, pair) for pair in pairs[::step][:per_group])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v7-root", type=Path, required=True)
    parser.add_argument("--v8-root", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-group", type=int, default=20)
    args = parser.parse_args()

    pairs = discover_pairs(args.v7_root, args.v8_root)
    log = json.loads(args.log.read_text())
    selected = choose_pairs(pairs, log["results"], args.per_group)

    category_counts = Counter()
    class_counts = Counter()
    shape_counts = Counter()
    separation_values = []
    by_pixel = {}
    records = []

    for classification, pair in selected:
        v7 = read_product(pair["v7_file"])
        v8 = read_product(pair["v8_file"])
        i7, i8 = align_scans(v7["scans"], v8["scans"])
        pixels = min(v7["latitude"].shape[1], v8["latitude"].shape[1])
        shape_counts[(str(v7["latitude"].shape), str(v8["latitude"].shape))] += 1
        if len(i7) == 0 or pixels == 0:
            continue

        p7 = v7["precipitation"][i7, :pixels]
        p8 = v8["precipitation"][i8, :pixels]
        lat7 = v7["latitude"][i7, :pixels]
        lon7 = v7["longitude"][i7, :pixels]
        lat8 = v8["latitude"][i8, :pixels]
        lon8 = v8["longitude"][i8, :pixels]
        geo = np.isfinite(lat7) & np.isfinite(lon7) & np.isfinite(lat8) & np.isfinite(lon8)
        conus_box = geo & (lat7 >= 24.0) & (lat7 < 50.0) & (lon7 >= -125.0) & (lon7 < -66.0)
        f7 = np.isfinite(p7)
        f8 = np.isfinite(p8)
        labels = {
            "both_finite": f7 & f8,
            "v7_only": f7 & ~f8,
            "v8_only": ~f7 & f8,
            "both_missing": ~f7 & ~f8,
        }
        for name, mask in labels.items():
            category_counts[(classification, "global", name)] += int((geo & mask).sum())
            category_counts[(classification, "conus_box", name)] += int((conus_box & mask).sum())

        distance = haversine_km(lat7[geo], lon7[geo], lat8[geo], lon8[geo])
        if distance.size:
            separation_values.extend(distance[:: max(1, distance.size // 10000)].tolist())

        for pixel in range(pixels):
            pixel_geo = geo[:, pixel]
            if not pixel_geo.any():
                continue
            slot = by_pixel.setdefault(pixel, Counter())
            for name, mask in labels.items():
                slot[name] += int((pixel_geo & mask[:, pixel]).sum())

        class_counts[classification] += 1
        records.append({
            "orbit_id": pair["orbit_id"],
            "classification": classification,
            "v7_shape": list(v7["latitude"].shape),
            "v8_shape": list(v8["latitude"].shape),
            "aligned_scans": int(len(i7)),
            "v7_unmatched_scans": int(len(v7["scans"]) - len(i7)),
            "v8_unmatched_scans": int(len(v8["scans"]) - len(i8)),
        })

    separation = np.asarray(separation_values, dtype=float)
    output = {
        "sampled_granules": int(len(records)),
        "sampled_by_class": dict(class_counts),
        "shape_pairs": {f"{a} | {b}": count for (a, b), count in shape_counts.items()},
        "validity_counts": {
            f"{classification}|{domain}|{category}": count
            for (classification, domain, category), count in sorted(category_counts.items())
        },
        "separation_km_quantiles": (
            dict(zip(
                ["min", "p25", "median", "p75", "p90", "p95", "p99", "max"],
                np.quantile(separation, [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]).tolist(),
            )) if separation.size else {}
        ),
        "cross_track_counts": {str(pixel): dict(counts) for pixel, counts in sorted(by_pixel.items())},
        "granules": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps({key: output[key] for key in ("sampled_granules", "sampled_by_class", "shape_pairs", "separation_km_quantiles")}, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()

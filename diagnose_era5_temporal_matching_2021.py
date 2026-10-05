#!/usr/bin/env python3
"""Compare legacy nearest-hour and corrected containing-hour ERA5 matching."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


THRESHOLD = 0.1
EXTENT = (-125.0, -66.0, 24.0, 50.0)
PRODUCTS = ("GPROF_V7", "GPROF_V8", "ERA5")
REFERENCES = ("MRMS", "StageIV")
PHASES = ("rain", "snow")
SCENARIOS = ("legacy", "corrected")


class Stats:
    def __init__(self):
        self.n = self.hits = self.misses = self.false_alarms = 0
        self.sum_x = self.sum_y = self.sum_x2 = self.sum_y2 = self.sum_xy = 0.0
        self.sum_sqerr = 0.0

    def add(self, product, reference):
        x = np.asarray(product, np.float64)
        y = np.asarray(reference, np.float64)
        self.n += x.size
        self.sum_x += x.sum(); self.sum_y += y.sum()
        self.sum_x2 += np.dot(x, x); self.sum_y2 += np.dot(y, y)
        self.sum_xy += np.dot(x, y); self.sum_sqerr += np.dot(x - y, x - y)
        wet_x, wet_y = x >= THRESHOLD, y >= THRESHOLD
        self.hits += int(np.count_nonzero(wet_x & wet_y))
        self.misses += int(np.count_nonzero(~wet_x & wet_y))
        self.false_alarms += int(np.count_nonzero(wet_x & ~wet_y))

    def metrics(self):
        n = self.n
        den_cc = ((n*self.sum_x2-self.sum_x**2)*(n*self.sum_y2-self.sum_y**2))**0.5
        cc = (n*self.sum_xy-self.sum_x*self.sum_y)/den_cc if den_cc > 0 else np.nan
        pod_den = self.hits + self.misses
        far_den = self.hits + self.false_alarms
        csi_den = self.hits + self.misses + self.false_alarms
        return {
            "N": n,
            "POD": self.hits/pod_den if pod_den else np.nan,
            "FAR": self.false_alarms/far_den if far_den else np.nan,
            "CSI": self.hits/csi_den if csi_den else np.nan,
            "frequency_bias": far_den/pod_den if pod_den else np.nan,
            "CC": cc,
            "RMSE": (self.sum_sqerr/n)**0.5 if n else np.nan,
            "relative_bias_percent": 100*(self.sum_x-self.sum_y)/self.sum_y if self.sum_y else np.nan,
        }

    def raw(self):
        return dict(vars(self))


def nearest_indices(values, grid):
    values, grid = np.asarray(values), np.asarray(grid)
    order = np.argsort(grid); sorted_grid = grid[order]
    pos = np.searchsorted(sorted_grid, values)
    p0 = np.clip(pos-1, 0, len(grid)-1); p1 = np.clip(pos, 0, len(grid)-1)
    return order[np.where(np.abs(sorted_grid[p1]-values) < np.abs(sorted_grid[p0]-values), p1, p0)]


def corrected_end_hours(times):
    hours = times.astype("datetime64[h]")
    exact = times.astype("datetime64[s]") == hours.astype("datetime64[s]")
    return hours + (~exact).astype("timedelta64[h]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--era5", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-files", type=int)
    ap.add_argument("--start-index", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.output); (out/"tables").mkdir(parents=True, exist_ok=True); (out/"figures").mkdir(exist_ok=True)
    files = sorted(Path(args.archive).glob("*2021*.nc"))
    files = files[args.start_index:]
    if args.max_files: files = files[:args.max_files]
    stats = defaultdict(Stats)
    changed = total = shifted_scans = total_scans = 0
    absdiff_sum = sqdiff_sum = 0.0

    with xr.open_dataset(args.era5, decode_cf=True, decode_timedelta=False) as era:
        tname = "valid_time" if "valid_time" in era.coords else "time"
        era_lat = era["latitude"].values; era_lon = era["longitude"].values
        lat_sel = np.flatnonzero((era_lat >= EXTENT[2]-0.5) & (era_lat <= EXTENT[3]+0.5))
        west360, east360 = EXTENT[0] % 360.0, EXTENT[1] % 360.0
        lon_sel = np.flatnonzero((era_lon >= west360-0.5) & (era_lon <= east360+0.5))
        lat_slice = slice(lat_sel.min(), lat_sel.max()+1); lon_slice = slice(lon_sel.min(), lon_sel.max()+1)
        era_lat_sub = era_lat[lat_slice]; era_lon_sub = era_lon[lon_slice]
        units = era["tp"].attrs.get("units", "m"); scale = 1000.0 if units == "m" else 1.0
        cache = {}
        for fi, path in enumerate(files, 1):
            with xr.open_dataset(path, decode_cf=True, decode_timedelta=False, mask_and_scale=True) as ds:
                lat = ds["latitude"].values; lon = ds["longitude"].values
                scan_time = ds["scan_time"].values.astype("datetime64[s]")
                end_hours = corrected_end_hours(scan_time)
                old_hours = ds["ERA5_time"].values.astype("datetime64[s]").astype("datetime64[h]")
                shifted_scans += int(np.count_nonzero(old_hours != end_hours)); total_scans += len(scan_time)
                corrected = np.full(lat.shape, np.nan, np.float32)
                spatial = np.isfinite(lat) & np.isfinite(lon) & (lat >= EXTENT[2]) & (lat < EXTENT[3]) & (lon >= EXTENT[0]) & (lon < EXTENT[1])
                if not np.any(spatial):
                    if fi % 100 == 0:
                        print(f"processed {fi}/{len(files)}", flush=True)
                    continue
                for hour in np.unique(end_hours):
                    rows = np.flatnonzero(end_hours == hour)
                    if hour not in cache:
                        field = era["tp"].sel({tname: hour}, method="nearest").isel(
                            latitude=lat_slice, longitude=lon_slice
                        ).values.astype(np.float32)*scale
                        field[field < 0] = 0
                        cache.clear(); cache[hour] = field
                    for row in rows:
                        mask = spatial[row]
                        if not np.any(mask): continue
                        yy = nearest_indices(lat[row, mask], era_lat_sub)
                        xx = nearest_indices(np.mod(lon[row, mask], 360.0), era_lon_sub)
                        corrected[row, mask] = cache[hour][yy, xx]

                arrays = {
                    "GPROF_V7": ds["surfacePrecipitation_V7"].values,
                    "GPROF_V8": ds["surfacePrecipitation_V8"].values,
                    "ERA5_legacy": ds["ERA5_precipitation"].values,
                    "ERA5_corrected": corrected,
                    "MRMS": ds["MRMS_Pass2"].values,
                    "StageIV": ds["StageIV"].values,
                }
                base = spatial & (ds["land_mask"].values == 1)
                twet = ds["MERRA2_T2MWET"].values
                phase_masks = {"rain": twet >= 275.15, "snow": twet <= 273.15}
                ref_flags = {"MRMS": ds["mrms_valid_flag"].values == 1, "StageIV": ds["stageiv_valid_flag"].values == 1}
                both = base & np.isfinite(arrays["ERA5_legacy"]) & np.isfinite(corrected)
                d = corrected[both].astype(np.float64)-arrays["ERA5_legacy"][both].astype(np.float64)
                total += d.size; changed += int(np.count_nonzero(d != 0)); absdiff_sum += np.abs(d).sum(); sqdiff_sum += np.dot(d,d)
                for phase in PHASES:
                    for ref in REFERENCES:
                        for scenario in SCENARIOS:
                            era_key = f"ERA5_{scenario}"
                            common = base & phase_masks[phase] & ref_flags[ref]
                            for key in (ref, "GPROF_V7", "GPROF_V8", era_key): common &= np.isfinite(arrays[key])
                            for product in PRODUCTS:
                                key = era_key if product == "ERA5" else product
                                stats[(scenario, phase, ref, product)].add(arrays[key][common], arrays[ref][common])
            if fi % 100 == 0: print(f"processed {fi}/{len(files)}", flush=True)

    rows = []
    for key, value in sorted(stats.items()):
        scenario, phase, reference, product = key
        rows.append({"scenario":scenario,"phase":phase,"reference":reference,"product":product,**value.metrics()})
    fields = list(rows[0])
    with open(out/"tables"/"metrics_old_vs_corrected.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    summary = {
        "files": len(files), "scans": total_scans,
        "shifted_scans": shifted_scans,
        "shifted_scan_percent": 100*shifted_scans/total_scans if total_scans else np.nan,
        "common_era5_footprints": total,
        "changed_era5_footprints": changed,
        "absolute_change_sum": absdiff_sum,
        "squared_change_sum": sqdiff_sum,
        "changed_value_percent": 100*changed/total if total else np.nan,
        "mean_absolute_change_mm_h": absdiff_sum/total if total else np.nan,
        "rms_change_mm_h": (sqdiff_sum/total)**0.5 if total else np.nan,
    }
    (out/"tables"/"change_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    raw_stats = {"|".join(key): value.raw() for key, value in stats.items()}
    (out/"tables"/"raw_sufficient_statistics.json").write_text(json.dumps(raw_stats,indent=2)+"\n")

    era_rows = [r for r in rows if r["product"]=="ERA5"]
    metrics = ["POD","FAR","CSI","frequency_bias","CC","RMSE","relative_bias_percent"]
    labels=[]; values=[]
    for phase in PHASES:
        for ref in REFERENCES:
            labels.append(f"{phase}\n{ref}")
            a=next(r for r in era_rows if r["scenario"]=="legacy" and r["phase"]==phase and r["reference"]==ref)
            b=next(r for r in era_rows if r["scenario"]=="corrected" and r["phase"]==phase and r["reference"]==ref)
            values.append([b[m]-a[m] for m in metrics])
    fig,axes=plt.subplots(2,4,figsize=(12,6)); axes=axes.ravel()
    for i,m in enumerate(metrics):
        axes[i].bar(labels,[v[i] for v in values],color="#356a9a"); axes[i].axhline(0,color="black",lw=.8); axes[i].set_title(f"Δ {m}"); axes[i].tick_params(axis="x",labelsize=8)
    axes[-1].axis("off"); fig.suptitle("Corrected minus legacy ERA5 temporal matching, 2021"); fig.tight_layout()
    fig.savefig(out/"figures"/"era5_metric_change_summary.png",dpi=180); plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__ == "__main__": main()

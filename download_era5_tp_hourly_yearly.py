#!/usr/bin/env python3
"""
Download ERA5 hourly total precipitation data year-by-year.

Dataset:
    ERA5 single levels: total_precipitation

Output:
    One NetCDF file per year saved to:
    /scratch/kkumah/ERA5_tp_hourly

Recommended script name:
    download_era5_tp_hourly_yearly.py
"""

from pathlib import Path
import cdsapi


# ---------------------------------------------------------
# User settings
# ---------------------------------------------------------
DATASET = "reanalysis-era5-single-levels"

OUT_DIR = Path("/scratch/kkumah/ERA5_tp_hourly")
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [str(y) for y in range(2014, 2026)]

MONTHS = [f"{m:02d}" for m in range(1, 13)]

DAYS = [f"{d:02d}" for d in range(1, 32)]

TIMES = [f"{h:02d}:00" for h in range(24)]


# ---------------------------------------------------------
# Main download routine
# ---------------------------------------------------------
def build_request(year: str) -> dict:
    """Build a CDS API request for one year of ERA5 hourly total precipitation."""
    return {
        "product_type": ["reanalysis"],
        "variable": ["total_precipitation"],
        "year": [year],
        "month": MONTHS,
        "day": DAYS,
        "time": TIMES,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def main():
    client = cdsapi.Client()

    print("=" * 70)
    print("ERA5 hourly total precipitation download")
    print(f"Output directory: {OUT_DIR}")
    print(f"Years requested: {YEARS[0]}–{YEARS[-1]}")
    print("=" * 70)

    completed_years = []

    for year in YEARS:
        out_file = OUT_DIR / f"ERA5_tp_hourly_{year}.nc"

        print("\n" + "-" * 70)
        print(f"Starting ERA5 total precipitation download for {year}")
        print(f"Target file: {out_file}")

        if out_file.exists() and out_file.stat().st_size > 0:
            print(f"Year {year} already exists. Skipping download.")
            completed_years.append(year)
            print(f"Completed years so far: {', '.join(completed_years)}")
            continue

        request = build_request(year)

        try:
            client.retrieve(DATASET, request).download(str(out_file))

            if out_file.exists() and out_file.stat().st_size > 0:
                completed_years.append(year)
                size_gb = out_file.stat().st_size / (1024**3)

                print(f"SUCCESS: ERA5 total precipitation for {year} downloaded.")
                print(f"Saved to: {out_file}")
                print(f"File size: {size_gb:.2f} GB")
                print(f"Completed years so far: {', '.join(completed_years)}")
            else:
                print(f"WARNING: Download for {year} finished, but output file is missing or empty.")

        except Exception as e:
            print(f"ERROR: Download failed for {year}")
            print(f"Reason: {e}")
            print("Continuing to the next year...")

    print("\n" + "=" * 70)
    print("ERA5 download routine finished.")
    print(f"Successfully completed years: {', '.join(completed_years) if completed_years else 'None'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
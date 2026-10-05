# Footprint-aligned V7/V8 rerun

All large matchup files and caches remain on Rain scratch. The prior 20260907
matchups and evaluation code are unchanged.

## 1. One-time environment check

```bash
conda activate kkumah_conda_env
python -c "import eccodes, h5py, netCDF4, numpy, rasterio, scipy, xarray; print('environment OK')"
```

At the time this workflow was prepared, only `eccodes` was missing from
`kkumah_conda_env`. The existing working `cml_env` contains the conda-forge
packages `eccodes` and `python-eccodes`. If the check still fails, install the
Python binding into the requested environment before running:

```bash
conda install -n kkumah_conda_env -c conda-forge python-eccodes
```

## 2. Short validation run

```bash
cd /home/kkumah/Projects/Comparing_V7_V8_GPROF-GMI_data/Codes
./run_GPROF_V7_V8_footprint_aligned_20260925.sh \
  --start-date 2022-08-06 \
  --end-date 2022-08-06 \
  --orbit-id 047939_20220806 \
  --workers 1
```

## 3. Full 2021–2024 matchup rerun

```bash
cd /home/kkumah/Projects/Comparing_V7_V8_GPROF-GMI_data/Codes
WORKERS=2 ./run_GPROF_V7_V8_footprint_aligned_20260925.sh \
  --start-date 2021-01-01 \
  --end-date 2024-12-31
```

The default server output is:

```text
/scratch/kkumah/GPROF_V7_V8_MRMS_RAQI_ERA5_MERRA2_AutoSnow_StageIV_matchups_footprint_aligned_20260925
```

## 4. Evaluation rerun

Open and run this file cell by cell after collocation completes:

```text
/home/kkumah/Projects/Comparing_V7_V8_GPROF-GMI_data/Codes/compare_gprof_v7_v8_mrms_stageiv_footprint_aligned_fullarchive_results_20260925.py
```

The evaluation defaults to a 1 km V7/V8 footprint-center separation limit.
Change `MAXIMUM_PAIR_SEPARATION_KM` to `2.5` or `5.0` for sensitivity tests.
Each setting creates a distinct cache on Rain scratch, so no collocation rerun
is required and no large cache is written to the local computer.

Evaluation plots, tables, and intermediate dataframes are also written beneath
the same scratch output root in `results/fullarchive_CONUS_land_20260925`.
Only the small source-code files live under the home project directory.

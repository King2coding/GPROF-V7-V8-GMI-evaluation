#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/home/kkumah/.conda/envs/kkumah_conda_env/bin/python}"
WORKERS="${WORKERS:-2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/kkumah/GPROF_V7_V8_MRMS_RAQI_ERA5_MERRA2_AutoSnow_StageIV_matchups_footprint_aligned_20260925}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: expected kkumah_conda_env Python at ${PYTHON}" >&2
  exit 2
fi

if ! "${PYTHON}" -c 'import eccodes, h5py, netCDF4, numpy, rasterio, scipy, xarray' 2>/dev/null; then
  echo "ERROR: kkumah_conda_env is missing a required package (currently eccodes)." >&2
  echo "No processing was started. Add eccodes to kkumah_conda_env, then rerun this command." >&2
  exit 3
fi

cd "${SCRIPT_DIR}"
exec "${PYTHON}" \
  collocate_GPROF_V7_V8_MRMS_RAQI_ERA5_MERRA2_AutoSnow_StageIV_footprint_aligned_20260925.py \
  --workers "${WORKERS}" \
  --output-root "${OUTPUT_ROOT}" \
  "$@"

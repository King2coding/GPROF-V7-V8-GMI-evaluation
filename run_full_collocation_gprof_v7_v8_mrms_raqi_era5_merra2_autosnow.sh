#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/home/kkumah/.conda/envs/cml_env/bin/python}"
WORKERS="${WORKERS:-2}"
SENSOR="${SENSOR:-GMI}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/kkumah/GPROF_V7_V8_MRMS_RAQI_ERA5_MERRA2_AutoSnow_matchups}"

cd "${SCRIPT_DIR}"
exec "${PYTHON}" collocate_gprof_v7_v8_with_mrms_raqi_era5_merra2_autosnow.py \
  --sensor "${SENSOR}" \
  --workers "${WORKERS}" \
  --output-root "${OUTPUT_ROOT}" \
  "$@"

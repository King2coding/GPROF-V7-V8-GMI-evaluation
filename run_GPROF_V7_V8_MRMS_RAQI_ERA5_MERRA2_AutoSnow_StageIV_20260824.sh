#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/home/kkumah/.conda/envs/cml_env/bin/python}"
WORKERS="${WORKERS:-2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/../analysis_outputs_StageIV_20260824}"

cd "${SCRIPT_DIR}"
exec "${PYTHON}" \
  collocate_GPROF_V7_V8_MRMS_RAQI_ERA5_MERRA2_AutoSnow_StageIV_20260824.py \
  --workers "${WORKERS}" \
  --output-root "${OUTPUT_ROOT}" \
  "$@"

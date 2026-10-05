#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/kkumah/GPROF_V7_V8_MRMS_RAQI_ERA5_MERRA2_AutoSnow_matchups}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${OUTPUT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/gmi_study_period_20210101_20241231_${STAMP}.log"
STATUS_FILE="${LOG_FILE}.exit_status"

mkdir -p "${LOG_DIR}"
cd "${SCRIPT_DIR}"

set +e
SENSOR=GMI OUTPUT_ROOT="${OUTPUT_ROOT}" \
  ./run_full_collocation_gprof_v7_v8_mrms_raqi_era5_merra2_autosnow.sh "$@" \
  --start-date 2021-01-01 \
  --end-date 2024-12-31 \
  2>&1 | tee "${LOG_FILE}"
status=${PIPESTATUS[0]}
set -e

printf 'Production command exit status: %d\n' "${status}" | tee -a "${LOG_FILE}"
printf '%d\n' "${status}" | tee "${STATUS_FILE}" >/dev/null
exit "${status}"

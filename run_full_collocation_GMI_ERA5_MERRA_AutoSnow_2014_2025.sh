#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/home/kkumah/Projects/Comparing_V7_V8_GPROF-GMI_data"
LOG_DIR="${PROJECT_ROOT}/misce"
OUTPUT_DIR="/scratch/kkumah/Collocated_GMI_ERA5_MERRA_Autosnow"
RUN_ID="$(date -u +%Y%m%d_%H%M%S)"
RUN_LOG="${LOG_DIR}/collocation_run_2014_2025_${RUN_ID}.log"
WORKERS="${WORKERS:-4}"
PYTHON="${PYTHON:-/home/kkumah/.conda/envs/cml_env/bin/python}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

cd "${SCRIPT_DIR}"

{
  echo "Starting GMI V7/V8 ERA5 MERRA2 AutoSnow collocation"
  echo "UTC start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "Run ID: ${RUN_ID}"
  echo "Workers: ${WORKERS}"
  echo "Python: ${PYTHON}"
  echo "Output directory: ${OUTPUT_DIR}"
  echo "Log directory: ${LOG_DIR}"

  PYTHONUNBUFFERED=1 "${PYTHON}" collocate_GMI_footprints_with_ERA5_MERRA2_AutoSnow.py \
    --year-start 2014 \
    --year-end 2025 \
    --workers "${WORKERS}" \
    --output-dir "${OUTPUT_DIR}" \
    --log-dir "${LOG_DIR}" \
    --run-id "${RUN_ID}" \
    --gmi-v7-dir /scratch/kkumah/GPM_GMI/V7 \
    --gmi-v8-dir /scratch/kkumah/GPM_GMI/V8 \
    --era5-dir /scratch/kkumah/ERA5_tp_hourly \
    --merra2-dir /ra1/pubdat/AVHRR_CloudSat_proj/MERRA2/merra2_archive_19800101_20251231 \
    --autosnow-dir /scratch/kkumah/Autosnow_2014_to_2024_nc

  echo "UTC end: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee "${RUN_LOG}"

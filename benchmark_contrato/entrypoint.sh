#!/usr/bin/env bash
set -euo pipefail

SCENARIO="${1:-official_reproduction}"

mkdir -p /app/benchmark_contrato/results/raw_logs
mkdir -p /app/benchmark_contrato/results/optional_predictions

export PYTHONPATH="/app:${PYTHONPATH:-}"

python /app/benchmark_contrato/export_results.py --stage metadata --scenario "${SCENARIO}"
python /app/benchmark_contrato/export_results.py --stage environment --scenario "${SCENARIO}"
python /app/benchmark_contrato/run_inference.py --scenario "${SCENARIO}"

if [[ "${SCENARIO}" == "standardized_efficiency" ]]; then
  python /app/benchmark_contrato/profile.py --scenario "${SCENARIO}"
else
  python /app/benchmark_contrato/profile.py --scenario "${SCENARIO}" || true
fi

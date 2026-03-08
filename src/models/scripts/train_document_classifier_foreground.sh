#!/usr/bin/env bash

set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "market-foundry" ]]; then
  echo "Expected active conda environment: market-foundry" >&2
  echo "Current environment: ${CONDA_DEFAULT_ENV:-<none>}" >&2
  echo "Run: conda activate market-foundry" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1

python src/models/train_document_classifier.py "$@"
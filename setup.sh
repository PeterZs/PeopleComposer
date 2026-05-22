#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-xflux}"
ENV_FILE="${2:-environment.yml}"

if ! command -v conda >/dev/null 2>&1; then
  echo "error: conda not found in PATH" >&2
  exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "error: env file not found: $ENV_FILE" >&2
  exit 1
fi

has_env() {
  conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "$ENV_NAME"
}

export CONDA_ALWAYS_YES=1

if has_env; then
  echo "[1/2] Updating conda env '$ENV_NAME' from '$ENV_FILE'..."
  conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
else
  echo "[1/2] Creating conda env from '$ENV_FILE'..."
  conda env create -f "$ENV_FILE"
fi

echo "[2/2] Installing OpenMMLab packages via mim into '$ENV_NAME'..."
conda run -n "$ENV_NAME" python -m pip install -U pip openmim
conda run -n "$ENV_NAME" mim install mmengine
conda run -n "$ENV_NAME" mim install "mmcv>=2.0.1"
conda run -n "$ENV_NAME" mim install "mmdet>=3.1.0"

echo "Done. To use it in your shell:"
echo "  conda activate $ENV_NAME"

# bash setup.sh
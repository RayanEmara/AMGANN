#!/usr/bin/env bash
# Native host training + local Linux PETSc/BoomerAMG solves. No x86 emulation.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
run_name="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ ! "$run_name" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
    echo 'Run name must contain only letters, digits, underscores and hyphens.' >&2
    exit 1
fi
output="data/navier-stokes/$run_name"
if [[ -e "$output" ]]; then
    echo "Output already exists: $output. Choose a new run name." >&2
    exit 1
fi
docker info >/dev/null
ns_python="$repo_root/.venv/bin/python"
if [[ ! -x "$ns_python" ]]; then
    python3 -m venv "$repo_root/.venv"
fi
if ! "$ns_python" -c 'import numpy, scipy, skfem, pyamg, torch, pytest' >/dev/null 2>&1; then
    "$ns_python" -m pip install -r code/navier-stokes/requirements.txt
fi
export PYTHONPATH="$repo_root/code/navier-stokes"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
docker build -t amg-ann-ns -f environment/Dockerfile.navier-stokes environment
docker run --rm -v "$repo_root:/workspace" amg-ann-ns collect \
    --backend petsc --output "$output/dataset" --repeats 3 --steps 5
"$ns_python" -m ns_amg train --dataset "$output/dataset" \
    --output "$output/model.npz" --epochs 150
docker run --rm -v "$repo_root:/workspace" amg-ann-ns compare \
    --backend petsc --model "$output/model.npz" --output "$output/comparison" \
    --cells 12 --dt 0.002 --steps 5 --viscosity 0.05 --repeats 5
docker run --rm -v "$repo_root:/workspace" amg-ann-ns convergence \
    --backend petsc --output "$output/convergence"
printf '\nResults: %s/%s\n' "$repo_root" "$output"

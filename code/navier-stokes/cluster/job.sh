#!/usr/bin/env bash
# Called by PBS from the repo root.
set -euo pipefail
profile=${1:?expected smoke or full}
case "$profile" in
    smoke) steps=2; epochs=3; trials=1; repeats=1; cells=12; dt=0.002 ;;
    full) steps=24; epochs=150; trials=3; repeats=5; cells=96; dt=0.0003 ;;
    *) exit 2 ;;
esac
: "${PBS_JOBID:?Submit the PBS script from the repository root}"
project=/work/$(id -un)/amg-ann
sif=${NS_SIF:-$project/ns-l4.sif}
[[ -r "$sif" ]] || { echo "Missing image: $sif" >&2; exit 1; }
src="$PBS_O_WORKDIR/code/navier-stokes"
[[ -f "$src/ns_amg/__main__.py" ]] || { echo 'Submit from the repository root.' >&2; exit 1; }
umask 077
mkdir -p "$project/runs"
dest=$(mktemp -d "$project/runs/$profile-$PBS_JOBID.XXXXXX")
scratch=$(mktemp -d "/scratch_local/ns-$(id -un)-$PBS_JOBID.XXXXXX")
echo "Compute node: $(hostname); outputs: $dest"
# Preserve partial results for diagnosis if a stage fails.
trap 'cp -a "$scratch/." "$dest/" || exit 1' EXIT
cd "$scratch"
nvidia-smi --query-gpu=name,driver_version,uuid --format=csv > gpu.csv
# Explicit binds also work on sites that do not mount /work or scratch by default.
runner=(apptainer exec --nv --bind "$src:/ns:ro,$scratch:/run-ns" --pwd /run-ns
        --env PYTHONPATH=/ns --env NS_TEST_BACKEND=petsc-cuda "$sif" /opt/ns/bin/python)
if [[ $profile == smoke ]]; then
    "${runner[@]}" -m pytest -p no:cacheprovider -q /ns/tests/test_solver.py
fi
"${runner[@]}" -m ns_amg collect --backend petsc-cuda --profile "$profile" --steps "$steps" --repeats "$trials" --output dataset
"${runner[@]}" -m ns_amg train --dataset dataset --epochs "$epochs" --output model.npz
"${runner[@]}" -m ns_amg compare --backend petsc-cuda --model model.npz --output comparison \
    --cells "$cells" --dt "$dt" --steps "$steps" --repeats "$repeats"
if [[ $profile == full ]]; then
    "${runner[@]}" -m ns_amg convergence --backend petsc-cuda --output convergence
fi

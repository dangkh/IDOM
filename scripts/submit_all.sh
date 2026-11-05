#!/bin/bash
# Submit the step-1 grid.
#   bash scripts/submit_all.sh                       # 4 jobs: one per config, its seeds share one GPU
#   PER_SEED=1 bash scripts/submit_all.sh            # 12 jobs: one per (config, seed), each with its own GPU
#   EPOCHS=41 SEEDS=1997 bash scripts/submit_all.sh  # quick test
#   CONFIGS="idom_idtext" PER_SEED=1 bash scripts/submit_all.sh
# Benchmark (bench.pbs): seeds sharing one GPU run ~2-3x slower EACH, so total time is about the same as
# running them one after another -> use PER_SEED=1 when several GPUs are free.
# Run from the repository root. Text configs need scripts/prepare_text.sh to have been run (CPU, no queue).
cd "$(dirname "$0")/.." || exit 1
CONFIGS="${CONFIGS:-fmrec idom_id idom_text idom_idtext}"
SEEDS="${SEEDS:-1997 2024 42}"
if [ "${PER_SEED:-0}" = "1" ]; then GROUPS_="$SEEDS"; else GROUPS_="${SEEDS// /:}"; fi
for c in $CONFIGS; do
    case "$c" in fmrec) short=fm ;; idom_id) short=id ;; idom_text) short=tx ;; idom_idtext) short=idtx ;; *) short=$c ;; esac
    for g in $GROUPS_; do
        v="CONFIG=$c,SEEDS=$g"
        [ -n "$EPOCHS" ] && v="$v,EPOCHS=$EPOCHS"
        [ -n "$PARALLEL" ] && v="$v,PARALLEL=$PARALLEL"
        [ -n "$CONDA_ENV" ] && v="$v,CONDA_ENV=$CONDA_ENV"
        [ -n "$TAG" ] && v="$v,TAG=$TAG"
        name="I${short}"; [ "${PER_SEED:-0}" = "1" ] && name="I${short}_${g}"
        echo -n "$c (seeds ${g//:/ }) -> "
        qsub -N "$name" -v "$v" scripts/idom_run.pbs
    done
done

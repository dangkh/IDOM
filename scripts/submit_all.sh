#!/bin/bash
# Submit the step-1 grid: 4 configs x 3 seeds = 12 PBS jobs (finished runs are skipped by the job itself).
#   bash scripts/submit_all.sh
#   EPOCHS=41 SEEDS=1997 bash scripts/submit_all.sh                 # quick test, 4 jobs
#   CONFIGS="idom_idtext" SEEDS="1997 2024 42" bash scripts/submit_all.sh
# Run from the repository root. Text configs need scripts/prepare_text.pbs to have finished.
cd "$(dirname "$0")/.." || exit 1
CONFIGS="${CONFIGS:-fmrec idom_id idom_text idom_idtext}"
SEEDS="${SEEDS:-1997 2024 42}"
for c in $CONFIGS; do
    for s in $SEEDS; do
        v="CONFIG=$c,SEED=$s"
        [ -n "$EPOCHS" ] && v="$v,EPOCHS=$EPOCHS"
        [ -n "$CONDA_ENV" ] && v="$v,CONDA_ENV=$CONDA_ENV"
        case "$c" in fmrec) short=fm ;; idom_id) short=id ;; idom_text) short=tx ;; idom_idtext) short=idtx ;; *) short=$c ;; esac
        echo -n "$c seed $s -> "
        qsub -N "I${short}_${s}" -v "$v" scripts/idom_run.pbs   # short name: some PBS versions cap it at 15 chars
    done
done

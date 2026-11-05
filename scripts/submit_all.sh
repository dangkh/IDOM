#!/bin/bash
# Submit exactly one PBS job per config (4 jobs); each job runs all SEEDS on its GPU.
#   bash scripts/submit_all.sh
#   EPOCHS=41 SEEDS=1997 bash scripts/submit_all.sh                 # quick test
#   CONFIGS="idom_idtext" bash scripts/submit_all.sh
# Run from the repository root. Text configs need scripts/prepare_text.sh to have been run (CPU, no queue).
cd "$(dirname "$0")/.." || exit 1
CONFIGS="${CONFIGS:-fmrec idom_id idom_text idom_idtext}"
SEEDS="${SEEDS:-1997 2024 42}"
SEEDS="${SEEDS// /:}"
for c in $CONFIGS; do
    case "$c" in fmrec) short=fm ;; idom_id) short=id ;; idom_text) short=tx ;; idom_idtext) short=idtx ;; *) short=$c ;; esac
    v="CONFIG=$c,SEEDS=$SEEDS"
    [ -n "$EPOCHS" ] && v="$v,EPOCHS=$EPOCHS"
    [ -n "$PARALLEL" ] && v="$v,PARALLEL=$PARALLEL"
    [ -n "$CONDA_ENV" ] && v="$v,CONDA_ENV=$CONDA_ENV"
    echo -n "$c (seeds ${SEEDS//:/ }) -> "
    qsub -N "I${short}" -v "$v" scripts/idom_run.pbs
done

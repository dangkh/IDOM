#!/bin/bash
# Step 0 (once, CPU only - no GPU / no queue needed; a few minutes on the login node):
# recover id->ASIN for amazon_beauty, verify it exactly, build item texts, encode them.
#   bash scripts/prepare_text.sh /path/to/amazon2014
#   ENCODER=sentence-transformers/all-mpnet-base-v2 bash scripts/prepare_text.sh /path/to/amazon2014
# The folder must contain reviews_Beauty_5.json.gz and meta_Beauty.json.gz (Amazon 2014).
cd "$(dirname "$0")/.." || exit 1
RAW_DIR="${1:?usage: bash scripts/prepare_text.sh /folder/with/reviews_Beauty_5.json.gz+meta_Beauty.json.gz}"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-Astar}"
ENCODER="${ENCODER:-sentence-transformers/all-MiniLM-L6-v2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"   # be gentle on a shared login node
python -u scripts/prepare_beauty_text.py \
    --reviews "$RAW_DIR/reviews_Beauty_5.json.gz" --meta "$RAW_DIR/meta_Beauty.json.gz" \
    --encoder "$ENCODER" --device cpu || { echo "text preparation failed"; exit 1; }
ls -la datasets/data/amazon_beauty/

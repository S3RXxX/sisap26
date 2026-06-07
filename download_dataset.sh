#!/usr/bin/env bash
# download_datasets.sh
#
# Downloads all SISAP 2026 benchmark datasets from HuggingFace.
#
# Usage:
#   ./download_datasets.sh [--small-only]
#
#   --small-only   Download only the small development datasets (fast, < 1 GB).
#                  Skips the large full-scale datasets (wikipedia ~15 GB, nq ~7 GB).
#
# After running this script every dataset is ready to use:
#   python search.py --task task1 --dataset wikipedia-small
#   python search.py --task task2 --dataset llama-dev
#   python search.py --task task3 --dataset fiqa-dev

set -euo pipefail

# ---------------------------------------------------------------------------
# Check for hf CLI
# ---------------------------------------------------------------------------

if ! command -v hf &>/dev/null; then
    echo "Error: hf command not found."
    echo "Please install it with: pip install -U huggingface_hub[cli]"
    exit 1
fi

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

SMALL_ONLY=false
for arg in "$@"; do
    [[ "$arg" == "--small-only" ]] && SMALL_ONLY=true
done

REPO="SISAP-Challenges/SISAP2026"

# ---------------------------------------------------------------------------
# Helper function to download a dataset directory
# ---------------------------------------------------------------------------

download_dataset() {
    local dataset_dir="$1"
    local dataset_name="$2"
    
    echo ""
    echo "-- $dataset_name --"
    
    # Download the entire directory using hf
    hf download "$REPO" \
        --repo-type dataset \
        --include "$dataset_dir/*" \
        --local-dir data
    
    echo "  done: data/$dataset_dir/"
}

# ---------------------------------------------------------------------------
# Task 1 – K-nearest neighbor graph  (k=15, dot product, normalized vectors)
# ---------------------------------------------------------------------------

echo ""
echo "=== Task 1: K-nearest neighbor graph ==="

download_dataset "wikipedia-small" "wikipedia-small (682 MB)"

if [[ "$SMALL_ONLY" == false ]]; then
    download_dataset "wikipedia" "wikipedia (full, ~15 GB)"
else
    echo ""
    echo "-- wikipedia (full) skipped (--small-only) --"
fi

download_dataset "task-1-spot-check" "task-1-spot-check (validation dataset)"



# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

echo ""
echo "All done. Datasets available:"
echo ""
echo "  Task 1 (all-kNN graph, k=15):"
echo "    wikipedia-small   data/wikipedia-small/"
if [[ "$SMALL_ONLY" == false ]]; then
echo "    wikipedia         data/wikipedia/"
fi
echo ""
echo ""
echo "  Spot-check (validation datasets):"
echo "    task-1-spot-check data/task-1-spot-check/"
echo ""
echo "Run search.py with any of these dataset names, e.g.:"
echo "  aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
echo ""
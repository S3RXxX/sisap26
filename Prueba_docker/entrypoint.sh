#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# entrypoint.sh  –  resolve env-vars and launch run_sisap2026.py
#
# All hyperparameters are read from environment variables set by:
#   • ENV lines in the Dockerfile  (defaults)
#   • -e VAR=value flags in docker run  (runtime overrides)
#
# To add a new hyperparameter:
#   1. Add  ENV NEW_PARAM=default_value  to the Dockerfile.
#   2. Read it here and append it to the python3 call below.
#   3. Parse it in run_sisap2026.py (argparse) and forward to sisap_bench.cpp.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

echo "========================================"
echo "  PiPNN SISAP 2026 Benchmark"
echo "========================================"
echo "  K              = ${K}"
echo "  BEAM_WIDTH     = ${BEAM_WIDTH}"
echo "  MAX_DEGREE     = ${MAX_DEGREE}"
echo "  ALPHA          = ${ALPHA}"
echo "  LEAF_SIZE      = ${LEAF_SIZE}"
echo "  MIN_LEAF_SIZE  = ${MIN_LEAF_SIZE}"
echo "  K_ENTRY        = ${K_ENTRY}"
echo "  ENTRY_SAMPLE   = ${ENTRY_SAMPLE}"
echo "  HASH_BITS      = ${HASH_BITS}"
echo "  RESERVOIR_CAP  = ${RESERVOIR_CAP}"
echo "  NUM_REPLICAS   = ${NUM_REPLICAS}"
echo "  FINAL_PRUNE    = ${FINAL_PRUNE}"
echo "  BACK_EDGE      = ${BACK_EDGE}"
echo "  SEED           = ${SEED}"
echo "  RAND           = ${RAND}"
echo "  NUM_THREADS    = ${NUM_THREADS}"
echo "  OMP_NUM_THREADS= ${OMP_NUM_THREADS}"
echo "  MEMORY_LIMIT_GB= ${MEMORY_LIMIT_GB}"
echo "  WORK_DIR       = ${WORK_DIR}"
echo "========================================"

# ── Optional soft RAM limit ───────────────────────────────────────────────────
if [ "${MEMORY_LIMIT_GB}" -gt 0 ] 2>/dev/null; then
    BYTES=$(( MEMORY_LIMIT_GB * 1024 * 1024 * 1024 ))
    echo "Applying soft memory limit: ${MEMORY_LIMIT_GB} GB"
    ulimit -v "${BYTES}" || echo "Warning: ulimit -v not supported in this environment; ignoring."
fi

# ── OMP thread count ──────────────────────────────────────────────────────────
if [ "${OMP_NUM_THREADS}" != "0" ]; then
    export OMP_NUM_THREADS
fi

# ── Launch benchmark ──────────────────────────────────────────────────────────
exec python3 /app/run_sisap2026.py \
    --work    "${WORK_DIR}"      \
    --k       "${K}"             \
    --bw      "${BEAM_WIDTH}"    \
    --max_degree     "${MAX_DEGREE}"     \
    --alpha          "${ALPHA}"          \
    --leaf_size      "${LEAF_SIZE}"      \
    --min_leaf_size  "${MIN_LEAF_SIZE}"  \
    --k_entry        "${K_ENTRY}"        \
    --entry_sample   "${ENTRY_SAMPLE}"   \
    --hash_bits      "${HASH_BITS}"      \
    --reservoir_cap  "${RESERVOIR_CAP}"  \
    --num_replicas   "${NUM_REPLICAS}"   \
    --final_prune    "${FINAL_PRUNE}"    \
    --back_edge      "${BACK_EDGE}"      \
    --num_threads    "${NUM_THREADS}"    \
    --seed           "${SEED}"           \
    --randomness     "${RAND}"           \
    "$@"   # any extra args passed directly to docker run are forwarded here

#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# entrypoint.sh  –  resolve env-vars / config, compile, and launch sisap_bench
#
# There is no Python step any more: sisap_bench reads the HDF5 file directly
# and does everything (load, quantise to int8, build, query, write results)
# in one C++ binary. This script only:
#   1. parses config_pipnn.json / the task's config.json with jq to get the
#      HDF5 path, the train dataset name, the ground-truth path, and k,
#   2. compiles sisap_bench.cpp once (cached across runs via mtime, same as
#      before, so -march=native still matches the host CPU),
#   3. applies the optional soft RAM limit,
#   4. execs the binary with all hyperparameters forwarded as argv.
#
# All hyperparameters are read from environment variables set by:
#   • ENV lines in the Dockerfile  (defaults)
#   • -e VAR=value flags in docker run  (runtime overrides)
#
# To add a new hyperparameter:
#   1. Add  ENV NEW_PARAM=default_value  to the Dockerfile.
#   2. Append it to the sisap_bench argv list below.
#   3. Parse it in sisap_bench.cpp's argv list.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

echo "========================================"
echo "  PiPNN SISAP 2026 Benchmark"
echo "========================================"
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
echo "  COOCKED        = ${COOCKED}"
echo "  NUM_THREADS    = ${NUM_THREADS}"
echo "  OMP_NUM_THREADS= ${OMP_NUM_THREADS}"
echo "  ALLKNN_SAMPLE  = ${ALLKNN_SAMPLE}"
echo "  MEMORY_LIMIT_GB= ${MEMORY_LIMIT_GB}"
echo "  WORK_DIR       = ${WORK_DIR}"
echo "  OUTPUT_PATH    = ${OUTPUT_PATH}"
echo "========================================"

mkdir -p "${WORK_DIR}"

# ── Resolve dataset paths from config_pipnn.json ─────────────────────────────
# config_pipnn.json  -> {"input": "<h5 file>", "task_description": "<config.json>"}
#
# Paths in config_pipnn.json may be given as absolute container paths
# (e.g. "/data/wikipedia-small/...") or as paths relative to the mounted
# data volume (e.g. "wikipedia-small/..." or the older
# "data/sisap_work/wikipedia-small/..." convention). Anything that isn't
# already absolute is resolved against /data, which is where `docker run -v
# .../data:/data` (and the provided docker-compose.yml) mount your host data
# directory. This is what the old version got wrong: a relative path in the
# JSON was being resolved against /app (the image's WORKDIR) instead of
# /data (the actual mount point), so it silently looked in the wrong place.
resolve_path() {
    local p="$1"
    if [[ "$p" = /* ]]; then
        echo "$p"
    else
        echo "/data/${p#data/}"   # strip a leading "data/" too, for the old convention
    fi
}

RAW_INPUT_H5=$(jq -r '.input' /app/config_pipnn.json)
RAW_TASK_JSON=$(jq -r '.task_description' /app/config_pipnn.json)
INPUT_H5=$(resolve_path "${RAW_INPUT_H5}")
TASK_JSON=$(resolve_path "${RAW_TASK_JSON}")

if [ ! -f "${TASK_JSON}" ] || [ ! -f "${INPUT_H5}" ]; then
    echo "----------------------------------------" >&2
    [ ! -f "${TASK_JSON}" ] && echo "ERROR: task description file not found: ${TASK_JSON}  (from config_pipnn.json: \"${RAW_TASK_JSON}\")" >&2
    [ ! -f "${INPUT_H5}" ]  && echo "ERROR: input HDF5 file not found: ${INPUT_H5}  (from config_pipnn.json: \"${RAW_INPUT_H5}\")" >&2
    echo "" >&2
    echo "Checked under /data. Is your data volume mounted there? Contents of /data:" >&2
    find /data -maxdepth 3 2>/dev/null >&2 || echo "  (nothing mounted at /data)" >&2
    echo "----------------------------------------" >&2
    exit 1
fi

# task's own config.json -> {"k": N, "data": "train", "gt_I": ["otest","knns"]}
K=$(jq -r '.k' "${TASK_JSON}")
TRAIN_DS=$(jq -r '.data' "${TASK_JSON}")
GT_PATH=$(jq -r '.gt_I | join("/")' "${TASK_JSON}")
K=$((K + 1))   # sisap_bench, like the old script, queries k+1 and drops self-match

# dataset_name — mirrors eval.py's _discover_datasets():
#   cfg.get("dataset_name", Path(cfg_path).parent.name)
# i.e. prefer an explicit "dataset_name" key in the task's config.json, and
# otherwise fall back to the name of the folder that config.json lives in
# (e.g. "wikipedia-small", "wikipedia"). This gets written into the output
# results.h5 as an attribute so eval.py can look the run up by dataset.
DATASET_NAME=$(jq -r '.dataset_name // empty' "${TASK_JSON}")
if [ -z "${DATASET_NAME}" ]; then
    DATASET_NAME=$(basename "$(dirname "${TASK_JSON}")")
fi

echo "  INPUT_H5   = ${INPUT_H5}"
echo "  TASK_JSON  = ${TASK_JSON}"
echo "  TRAIN_DS   = ${TRAIN_DS}"
echo "  GT_PATH    = ${GT_PATH}"
echo "  K (k+1)    = ${K}"
echo "  DATASET    = ${DATASET_NAME}"
echo "========================================"

# ── Optional soft RAM limit ───────────────────────────────────────────────────
if [ "${MEMORY_LIMIT_GB}" -gt 0 ] 2>/dev/null; then
    BYTES=$(( MEMORY_LIMIT_GB * 1024 * 1024 * 1024 ))
    echo "Applying soft memory limit: ${MEMORY_LIMIT_GB} GB"
    ulimit -v "${BYTES}" || echo "Warning: ulimit -v not supported in this environment; ignoring."
fi

# ── OMP thread count ──────────────────────────────────────────────────────────
# Docker's ENV already puts OMP_NUM_THREADS=0 in the environment even before
# this script runs, and libgomp treats 0 as an invalid explicit value (it
# crashes with "Invalid value for environment variable OMP_NUM_THREADS: 0"
# instead of falling back to "all cores"). So when the value is 0 we must
# unset it, not just skip re-exporting it.
if [ "${OMP_NUM_THREADS}" = "0" ]; then
    unset OMP_NUM_THREADS
else
    export OMP_NUM_THREADS
fi

# ── Compile sisap_bench once (cached by mtime, -march=native for this host) ──
BENCH_BIN="${WORK_DIR}/sisap_bench"
NEEDS_BUILD=1
if [ -x "${BENCH_BIN}" ] \
   && [ "${BENCH_BIN}" -nt /app/sisap_bench.cpp ] \
   && [ "${BENCH_BIN}" -nt /app/pipnn_dot.hpp ]; then
    NEEDS_BUILD=0
fi
if [ "${NEEDS_BUILD}" -eq 1 ]; then
    echo "Compiling sisap_bench ..."
    g++ -O3 -std=c++17 -fopenmp -I/app /app/sisap_bench.cpp \
        -I/usr/include/hdf5/serial \
        -L/usr/lib/x86_64-linux-gnu/hdf5/serial \
        -lhdf5_cpp -lhdf5 -march=native -o "${BENCH_BIN}"
    echo "  OK"
else
    echo "${BENCH_BIN} is up to date."
fi

# ── Launch benchmark ──────────────────────────────────────────────────────────
exec "${BENCH_BIN}" \
    "${INPUT_H5}" \
    "${TRAIN_DS}" \
    "${GT_PATH}" \
    "${K}" \
    "${OUTPUT_PATH}" \
    "${ALLKNN_SAMPLE}" \
    "${BEAM_WIDTH}" \
    "${MAX_DEGREE}" \
    "${ALPHA}" \
    "${LEAF_SIZE}" \
    "${MIN_LEAF_SIZE}" \
    "${K_ENTRY}" \
    "${ENTRY_SAMPLE}" \
    "${HASH_BITS}" \
    "${RESERVOIR_CAP}" \
    "${NUM_REPLICAS}" \
    "${FINAL_PRUNE}" \
    "${BACK_EDGE}" \
    "${NUM_THREADS}" \
    "${SEED}" \
    "${RAND}" \
    "${COOCKED}" \
    "${DATASET_NAME}" \
    "$@"   # any extra args passed directly to docker run (e.g. --save-index PATH)
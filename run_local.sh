#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_local.sh  –  compile + run sisap_bench directly, no Docker required.
#
# Same logic as entrypoint.sh (config parsing, compile caching, memory limit,
# OMP_NUM_THREADS fix, auto-named output), just without any container/volume
# assumptions: you pass real filesystem paths instead of relying on a /data
# mount.
#
# USAGE
#   ./run_local.sh --input PATH --config PATH [options...]
#
#   --input PATH        Path to the dataset's .h5 file (required)
#   --config PATH       Path to that task's config.json, {"k":.., "data":..}
#                        (required)
#   --output PATH        Exact output .h5 path (default: auto-generated,
#                        see below)
#   --output-dir DIR     Base dir for auto-generated output (default: ./output)
#   --work-dir DIR        Where the compiled binary is cached (default: ./.sisap_work)
#   --save-index PATH     Forwarded to sisap_bench
#   --load-index PATH     Forwarded to sisap_bench
#   -h, --help            Show this help
#
#   Alternatively, drop a config_pipnn.json next to this script (same format
#   as the Docker image uses):
#     {"input": "<h5 path>", "task_description": "<config.json path>"}
#   and just run ./run_local.sh with no --input/--config — paths in it may be
#   absolute or relative to this script's directory.
#
# HYPERPARAMETERS are read from environment variables, with the same names
# and defaults as the Dockerfile:
#   BEAM_WIDTH, MAX_DEGREE, ALPHA, LEAF_SIZE, MIN_LEAF_SIZE, K_ENTRY,
#   ENTRY_SAMPLE, HASH_BITS, RESERVOIR_CAP, NUM_REPLICAS, FINAL_PRUNE,
#   BACK_EDGE, SEED, RAND, COOCKED, ALLKNN_SAMPLE, NUM_THREADS,
#   OMP_NUM_THREADS, MEMORY_LIMIT_GB
#
# EXAMPLE
#   BEAM_WIDTH=64 MAX_DEGREE=64 ./run_local.sh \
#       --input data/wikipedia-small/benchmark-dev-wikipedia-bge-m3-small.h5 \
#       --config data/wikipedia-small/config.json
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Defaults (mirrors the Dockerfile's ENV defaults) ─────────────────────────
: "${BEAM_WIDTH:=32}"
: "${MAX_DEGREE:=64}"
: "${ALPHA:=1.2}"
: "${LEAF_SIZE:=512}"
: "${MIN_LEAF_SIZE:=32}"
: "${K_ENTRY:=12}"
: "${ENTRY_SAMPLE:=3000}"
: "${HASH_BITS:=12}"
: "${RESERVOIR_CAP:=128}"
: "${NUM_REPLICAS:=1}"
: "${FINAL_PRUNE:=1}"
: "${BACK_EDGE:=1}"
: "${SEED:=42}"
: "${RAND:=0}"
: "${COOCKED:=0}"
: "${ALLKNN_SAMPLE:=0}"
: "${NUM_THREADS:=0}"
: "${OMP_NUM_THREADS:=0}"
: "${MEMORY_LIMIT_GB:=0}"

WORK_DIR="${SCRIPT_DIR}/.sisap_work"
OUTPUT_BASE_DIR="${SCRIPT_DIR}/output"
OUTPUT_PATH=""
INPUT_H5=""
TASK_JSON=""
SAVE_INDEX=""
LOAD_INDEX=""

usage() { sed -n '2,45p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
    case "$1" in
        --input)       INPUT_H5="$2"; shift 2 ;;
        --config)      TASK_JSON="$2"; shift 2 ;;
        --output)      OUTPUT_PATH="$2"; shift 2 ;;
        --output-dir)  OUTPUT_BASE_DIR="$2"; shift 2 ;;
        --work-dir)    WORK_DIR="$2"; shift 2 ;;
        --save-index)  SAVE_INDEX="$2"; shift 2 ;;
        --load-index)  LOAD_INDEX="$2"; shift 2 ;;
        -h|--help)     usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

# Parse JSON: prefer jq; fall back to python3 (near-universal) if jq isn't
# installed, so this still works on a bare-bones host without Docker or jq.
json_get() {
    local expr="$1" file="$2" pyexpr="$3"
    if command -v jq >/dev/null 2>&1; then
        jq -r "${expr}" "${file}"
    else
        python3 -c "
import json,sys
cfg = json.load(open(sys.argv[1]))
print(${pyexpr})
" "${file}"
    fi
}

# ── Fall back to config_pipnn.json next to this script if --input/--config
#    weren't given, same convention the Docker image uses ────────────────────
if [ -z "${INPUT_H5}" ] || [ -z "${TASK_JSON}" ]; then
    CFG="${SCRIPT_DIR}/config_pipnn.json"
    if [ -f "${CFG}" ]; then
        resolve_rel() { local p="$1"; [[ "$p" = /* ]] && echo "$p" || echo "${SCRIPT_DIR}/${p}"; }
        [ -z "${INPUT_H5}" ]  && INPUT_H5=$(resolve_rel "$(json_get '.input' "${CFG}" "cfg['input']")")
        [ -z "${TASK_JSON}" ] && TASK_JSON=$(resolve_rel "$(json_get '.task_description' "${CFG}" "cfg['task_description']")")
    fi
fi

if [ -z "${INPUT_H5}" ] || [ -z "${TASK_JSON}" ]; then
    echo "ERROR: --input and --config are required (or provide config_pipnn.json next to this script)." >&2
    usage
    exit 1
fi

echo "========================================"
echo "  PiPNN SISAP 2026 Benchmark (local, no Docker)"
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
echo "========================================"

mkdir -p "${WORK_DIR}"

if [ ! -f "${INPUT_H5}" ]; then
    echo "ERROR: input HDF5 file not found: ${INPUT_H5}" >&2
    exit 1
fi
if [ ! -f "${TASK_JSON}" ]; then
    echo "ERROR: task description file not found: ${TASK_JSON}" >&2
    exit 1
fi

# ── Parse the task's config.json ─────────────────────────────────────────────
K=$(json_get '.k' "${TASK_JSON}" "cfg.get('k','')")
TRAIN_DS=$(json_get '.data' "${TASK_JSON}" "cfg.get('data','')")
DATASET_NAME=$(json_get '.dataset_name // empty' "${TASK_JSON}" "cfg.get('dataset_name','')")
TASK_NAME=$(json_get '.task // "task1"' "${TASK_JSON}" "cfg.get('task','task1')")

if [ -z "${DATASET_NAME}" ]; then
    DATASET_NAME=$(basename "$(dirname "${TASK_JSON}")")
fi
if [ -z "${TASK_NAME}" ] || [ "${TASK_NAME}" = "null" ]; then
    TASK_NAME="task1"
fi

if [ -z "${K}" ] || [ -z "${TRAIN_DS}" ] || [ "${K}" = "null" ] || [ "${TRAIN_DS}" = "null" ]; then
    echo "ERROR: could not read 'k' and/or 'data' from ${TASK_JSON}" >&2
    exit 1
fi

# ── Output filename ───────────────────────────────────────────────────────────
if [ -z "${OUTPUT_PATH}" ]; then
    PARAMS_TAG="bw${BEAM_WIDTH}_deg${MAX_DEGREE}_ef${ENTRY_SAMPLE}"
    OUTPUT_PATH="${OUTPUT_BASE_DIR}/results/${TASK_NAME}/PiPNN_${PARAMS_TAG}.h5"
fi
mkdir -p "$(dirname "${OUTPUT_PATH}")"

echo "  INPUT_H5   = ${INPUT_H5}"
echo "  TASK_JSON  = ${TASK_JSON}"
echo "  TRAIN_DS   = ${TRAIN_DS}"
echo "  K          = ${K}"
echo "  DATASET    = ${DATASET_NAME}"
echo "  TASK       = ${TASK_NAME}"
echo "  OUTPUT_PATH= ${OUTPUT_PATH}"
echo "========================================"

# ── Optional soft RAM limit ───────────────────────────────────────────────────
if [ "${MEMORY_LIMIT_GB}" -gt 0 ] 2>/dev/null; then
    BYTES=$(( MEMORY_LIMIT_GB * 1024 * 1024 * 1024 ))
    echo "Applying soft memory limit: ${MEMORY_LIMIT_GB} GB"
    ulimit -v "${BYTES}" || echo "Warning: ulimit -v not supported in this environment; ignoring."
fi

# ── OMP thread count ──────────────────────────────────────────────────────────
if [ "${OMP_NUM_THREADS}" = "0" ]; then
    unset OMP_NUM_THREADS
else
    export OMP_NUM_THREADS
fi

# ── Locate HDF5 C++ headers/libs ─────────────────────────────────────────────
# Tries, in order: pkg-config, the h5c++ wrapper compiler, then a few common
# install locations. Override by exporting HDF5_CFLAGS/HDF5_LIBS yourself if
# none of these match your system.
HDF5_CFLAGS="${HDF5_CFLAGS:-}"
HDF5_LIBS="${HDF5_LIBS:-}"
if [ -z "${HDF5_CFLAGS}" ] && [ -z "${HDF5_LIBS}" ]; then
    if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists hdf5 2>/dev/null; then
        HDF5_CFLAGS="$(pkg-config --cflags hdf5)"
        HDF5_LIBS="$(pkg-config --libs hdf5) -lhdf5_cpp"
    else
        for d in /usr/include/hdf5/serial /usr/include/hdf5 /usr/local/include /opt/homebrew/include; do
            if [ -f "${d}/H5Cpp.h" ]; then HDF5_CFLAGS="-I${d}"; break; fi
        done
        for d in /usr/lib/x86_64-linux-gnu/hdf5/serial /usr/lib/aarch64-linux-gnu/hdf5/serial \
                 /usr/lib/hdf5 /usr/local/lib /opt/homebrew/lib; do
            if [ -f "${d}/libhdf5_cpp.so" ] || [ -f "${d}/libhdf5_cpp.a" ] || [ -f "${d}/libhdf5_cpp.dylib" ]; then
                HDF5_LIBS="-L${d} -lhdf5_cpp -lhdf5"; break
            fi
        done
    fi
fi
if [ -z "${HDF5_CFLAGS}" ] || [ -z "${HDF5_LIBS}" ]; then
    echo "ERROR: could not locate HDF5 C++ headers/libs automatically." >&2
    echo "  Install them (e.g. 'apt install libhdf5-dev' / 'brew install hdf5')," >&2
    echo "  or set HDF5_CFLAGS and HDF5_LIBS yourself, e.g.:" >&2
    echo "    export HDF5_CFLAGS='-I/path/to/include'" >&2
    echo "    export HDF5_LIBS='-L/path/to/lib -lhdf5_cpp -lhdf5'" >&2
    exit 1
fi
echo "  HDF5_CFLAGS= ${HDF5_CFLAGS}"
echo "  HDF5_LIBS  = ${HDF5_LIBS}"
echo "========================================"

# ── Compile sisap_bench once (cached by mtime, -march=native for this host) ──
BENCH_BIN="${WORK_DIR}/sisap_bench"
NEEDS_BUILD=1
if [ -x "${BENCH_BIN}" ] \
   && [ "${BENCH_BIN}" -nt "${SCRIPT_DIR}/sisap_bench.cpp" ] \
   && [ "${BENCH_BIN}" -nt "${SCRIPT_DIR}/pipnn_dot.hpp" ]; then
    NEEDS_BUILD=0
fi
if [ "${NEEDS_BUILD}" -eq 1 ]; then
    echo "Compiling sisap_bench ..."
    if ! g++ -O3 -std=c++17 -fopenmp -I"${SCRIPT_DIR}" "${SCRIPT_DIR}/sisap_bench.cpp" \
            ${HDF5_CFLAGS} ${HDF5_LIBS} -march=native -o "${BENCH_BIN}" 2>/tmp/sisap_build.log; then
        echo "  -march=native failed, retrying without it (e.g. cross-arch build) ..."
        g++ -O3 -std=c++17 -fopenmp -I"${SCRIPT_DIR}" "${SCRIPT_DIR}/sisap_bench.cpp" \
            ${HDF5_CFLAGS} ${HDF5_LIBS} -o "${BENCH_BIN}"
    fi
    echo "  OK"
else
    echo "${BENCH_BIN} is up to date."
fi

# ── Launch benchmark ──────────────────────────────────────────────────────────
EXTRA_ARGS=()
[ -n "${SAVE_INDEX}" ] && EXTRA_ARGS+=(--save-index "${SAVE_INDEX}")
[ -n "${LOAD_INDEX}" ] && EXTRA_ARGS+=(--load-index "${LOAD_INDEX}")

exec "${BENCH_BIN}" \
    "${INPUT_H5}" \
    "${TRAIN_DS}" \
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
    "${EXTRA_ARGS[@]}"
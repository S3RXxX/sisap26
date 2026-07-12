# ─────────────────────────────────────────────────────────────────────────────
# PiPNN SISAP 2026 Benchmark – Docker image (TIRA-compatible)
# ─────────────────────────────────────────────────────────────────────────────
#
# The image no longer bakes in a dataset or a fixed config_pipnn.json: the
# executable at /app/sisap_bench takes the dataset, task description, and
# output location as explicit CLI flags, exactly matching TIRA's submission
# convention:
#
#   pip3 install --upgrade tira
#
#   tira-cli code-submission \
#       --path . \
#       --command '/app/sisap_bench --input $inputDataset/*.h5 --task-description $inputDataset/config.json --output $outputDir' \
#       --task sisap-2026 \
#       --dataset task-1-spot-check-20260602-training \
#       --dry-run
#
# TIRA substitutes $inputDataset (a directory containing the dataset's .h5
# file and config.json) and $outputDir (where the results file should be
# written) at evaluation time, so nothing dataset-specific needs to be
# copied into the image or mounted as a volume ahead of time.
#
# HOW TO BUILD:
#   docker build -t pipnn-sisap .
#
# HOW TO RUN LOCALLY (mount your own data/output directories):
#   docker run --rm \
#     -v $(pwd)/data/wikipedia-small:/data/wikipedia-small:ro \
#     -v $(pwd)/output:/output \
#     pipnn-sisap \
#     --input /data/wikipedia-small/benchmark-dev-wikipedia-bge-m3-small.h5 \
#     --task-description /data/wikipedia-small/config.json \
#     --output /output
#
# --output may be a directory (results.h5 is written inside it) or an exact
# file path.
#
# HOW TO OVERRIDE HYPERPARAMETERS:
#   docker run --rm ... -e BEAM_WIDTH=256 -e MAX_DEGREE=64 ... pipnn-sisap --input ... --task-description ... --output ...
#
# MEMORY LIMIT:
#   Set MEMORY_LIMIT_GB to soft-limit RAM via ulimit inside the container.
#   For a hard Docker-level limit use --memory:
#     docker run --rm --memory=16g ... pipnn-sisap --input ...
#
# ADD MORE HYPERPARAMETERS:
#   1. Add a new ENV line below with its default value.
#   2. Read it via env_int/env_float/env_bool in sisap_bench.cpp's main().
#   3. Override at runtime with -e NEW_PARAM=value in docker run.
# ─────────────────────────────────────────────────────────────────────────────

FROM ubuntu:24.04

# ── System dependencies ───────────────────────────────────────────────────────
# No Python: sisap_bench.cpp reads the HDF5 file, parses config.json itself,
# quantises to int8, builds, queries, and writes results all in one C++
# binary.
RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ \
        libgomp1 \
        ca-certificates \
        pkg-config libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Copy source files ─────────────────────────────────────────────────────────
WORKDIR /app
COPY pipnn_dot.hpp     .
COPY sisap_bench.cpp   .

# The wrapper script IS the executable referenced in a TIRA --command
# (/app/sisap_bench). It recompiles with -march=native for whatever host
# actually runs the container — see the script's own header comment for why
# that happens at run time rather than here at build time.
COPY sisap_bench       .
RUN sed -i 's/\r//' sisap_bench && chmod +x sisap_bench

# Sanity-compile once at build time (without -march=native) so a broken
# build fails fast in `docker build` rather than silently at first run. This
# particular binary is discarded — the wrapper script above always
# (re)compiles its own copy for the real execution host.
RUN g++ -O3 -std=c++17 -fopenmp -I. sisap_bench.cpp \
        -I/usr/include/hdf5/serial \
        -L/usr/lib/x86_64-linux-gnu/hdf5/serial \
        -lhdf5_cpp -lhdf5 -o /tmp/sisap_bench_buildcheck \
    && rm -f /tmp/sisap_bench_buildcheck

# ── Default hyperparameters ───────────────────────────────────────────────────
# Query / retrieval
# Beam width during graph search (higher = better recall, slower)
ENV BEAM_WIDTH=32

# Graph build parameters
# Max out-degree per node (R). Higher → better recall, more memory
ENV MAX_DEGREE=32
# RobustPrune directional factor. Range [1.0, 2.0]
ENV ALPHA=1.2
# Max points per leaf cluster (Cmax)
ENV LEAF_SIZE=512
# Leaf merge threshold (Cmin)
ENV MIN_LEAF_SIZE=32
# Number of diverse graph entry points
ENV K_ENTRY=12
# Sample size used to select entry points
ENV ENTRY_SAMPLE=3000
# LSH bits for HashPrune (≤ 16)
ENV HASH_BITS=12
# HashPrune reservoir size (lmax). IMPORTANT: this is allocated at its full
# size for EVERY point right at the start of the build (n * RESERVOIR_CAP *
# 10 bytes), not something that grows gradually — for a multi-million-point
# dataset this is very likely the single largest structure in the whole
# pipeline, often bigger than the (int8-quantized) dataset itself. e.g. at
# n=6.5M: RESERVOIR_CAP=128 -> ~8.3GB just for this array. If you're hitting
# OOM on a large dataset, lowering this is the single most direct lever —
# it shrinks memory linearly and also speeds up graph construction (each
# insert scans up to RESERVOIR_CAP existing entries).
ENV RESERVOIR_CAP=32
# Independent RBC graph replications (increases build time)
ENV NUM_REPLICAS=1
# Apply RobustPrune after HashPrune (0/1)
ENV FINAL_PRUNE=0
# Run back-edge consolidation pass (0/1)
ENV BACK_EDGE=0
# Random seed for reproducibility
ENV SEED=42
# Randomness
ENV RAND=0
# COOCKED
ENV COOCKED=0
# Vectors are always stored/queried as int8 (already-normalised embeddings,
# no other precision path)

# Number of train vectors sampled as all-kNN queries (0 = use the full
# training set as queries, reusing the same in-memory buffer, no copy).
# Can also be overridden per-run with --allknn-sample on the CLI.
ENV ALLKNN_SAMPLE=0

# Runtime / system
# OMP threads for build+query (0 = all available cores)
ENV NUM_THREADS=0
# Docker-level OMP override (0 = all cores)
ENV OMP_NUM_THREADS=0
# Soft RAM limit in GB via ulimit (0 = unlimited)
ENV MEMORY_LIMIT_GB=0

# Where the host-native compiled binary is cached across invocations of the
# /app/sisap_bench wrapper script within the same container.
ENV SISAP_BUILD_CACHE=/tmp/sisap_work

ENTRYPOINT ["/app/sisap_bench"]
CMD ["--help"]
# ─────────────────────────────────────────────────────────────────────────────
# PiPNN SISAP 2026 Benchmark – Docker image
# ─────────────────────────────────────────────────────────────────────────────
#
# HOW TO BUILD:
#   docker build -t pipnn-sisap .
#
# HOW TO RUN (minimal – all defaults):
#   docker run --rm \
#     -v $(pwd)/data:/data \
#     pipnn-sisap
#
# HOW TO RUN (full example with all hyperparameters):
#   docker run --rm \
#     -v $(pwd)/data:/data \
#     -e K=15 \
#     -e BEAM_WIDTH=256 \
#     -e MAX_DEGREE=64 \
#     -e ALPHA=1.2 \
#     -e LEAF_SIZE=512 \
#     -e MIN_LEAF_SIZE=32 \
#     -e K_ENTRY=12 \
#     -e ENTRY_SAMPLE=3000 \
#     -e HASH_BITS=12 \
#     -e RESERVOIR_CAP=128 \
#     -e NUM_REPLICAS=1 \
#     -e FINAL_PRUNE=1 \
#     -e BACK_EDGE=1 \
#     -e NUM_THREADS=0 \
#     -e SEED=42 \
#     -e OMP_NUM_THREADS=8 \
#     -e MEMORY_LIMIT_GB=16 \
#     pipnn-sisap
#
# MEMORY LIMIT:
#   Set MEMORY_LIMIT_GB to soft-limit RAM via ulimit inside the container.
#   For a hard Docker-level limit use --memory flag:
#     docker run --rm --memory=16g -v $(pwd)/data:/data pipnn-sisap
#
# ADD MORE HYPERPARAMETERS:
#   1. Add a new ENV line below with its default value.
#   2. Reference it as $NEW_PARAM in the CMD / entrypoint call.
#   3. Parse it in sisap_bench.cpp (add a new argv slot or use a config file).
#   4. Override at runtime with -e NEW_PARAM=value in docker run.
#
# VOLUME:
#   Mount a host directory to /data.  The HDF5 file will be downloaded there
#   (or reused if already present), and all binary work files will be written
#   inside /data/sisap_work/.
# ─────────────────────────────────────────────────────────────────────────────

FROM ubuntu:24.04

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ \
        libgomp1 \
        python3 \
        python3-pip \
        python3-venv \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
RUN pip3 install --no-cache-dir --break-system-packages \
        h5py \
        numpy \
        huggingface_hub

# ── Copy source files ─────────────────────────────────────────────────────────
WORKDIR /app
COPY pipnn_dot.hpp     .
COPY sisap_bench.cpp   .
COPY run_sisap2026.py  .

# ── Default hyperparameters ───────────────────────────────────────────────────
# Query / retrieval
# Number of nearest neighbours to retrieve
ENV K=15

# Beam width during graph search (higher = better recall, slower)
ENV BEAM_WIDTH=256        

# Graph build parameters (passed via sisap_bench / make_config overrides)
# Max out-degree per node (R). Higher → better recall, more memory
ENV MAX_DEGREE=64
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
# HashPrune reservoir size (lmax)
ENV RESERVOIR_CAP=128
# Independent RBC graph replications (increases build time)
ENV NUM_REPLICAS=1
# Apply RobustPrune after HashPrune (0/1)
ENV FINAL_PRUNE=1
# Run back-edge consolidation pass (0/1)
ENV BACK_EDGE=1
# Random seed for reproducibility
ENV SEED=42
# Randomness
ENV RAND=0
# COOCKED
ENV COOCKED=0

# Runtime / system
# OMP threads for build+query (0 = all available cores)
ENV NUM_THREADS=0
# Docker-level OMP override (0 = all cores)     
ENV OMP_NUM_THREADS=0
# Soft RAM limit in GB via ulimit (0 = unlimited)
ENV MEMORY_LIMIT_GB=0

# Data path
# Where binary files and the compiled binary are written
ENV WORK_DIR=/data/sisap_work

# ── Entrypoint ────────────────────────────────────────────────────────────────
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

VOLUME ["/data"]

ENTRYPOINT ["/app/entrypoint.sh"]

#docker run --rm -v ${PWD}/data:/data -e K=15 -e RAND=1 -e BEAM_WIDTH=256 -e MAX_DEGREE=64 -e ALPHA=1.1 -e NUM_THREADS=0 -e NUM_THREADS=0 -e MEMORY_LIMIT_GB=16 pipnn-sisap

#!/usr/bin/env python3
"""
run_sisap2026.py – PiPNN benchmark for SISAP 2026 (wikipedia-small & wikipedia).

Vectors are stored as float16 to fit the full ~3.6M x 1024 wikipedia
dataset (~7.4 GB) in 24 GB RAM. The C++ side memory-maps the train file
as float16 and converts only small per-batch / per-node working sets to
float32 internally.

Usage
─────
  python run_sisap2026.py                              # wikipedia-small
  python run_sisap2026.py --dataset wikipedia          # full 14.9 GB dataset
  python run_sisap2026.py --dataset wikipedia --save-index idx.bin
  python run_sisap2026.py --dataset wikipedia --load-index idx.bin --bw 512

Flags
─────
  --dataset   wikipedia-small | wikipedia   (default: wikipedia-small)
  --h5        override local HDF5 path
  --k         number of neighbours          (default: from config.json)
  --bw        query beam width              (default: 256)
  --work      working directory for binaries / cache   (default: sisap_work)
  --allknn-sample N   vectors sampled from train for allknn recall
                      (default: 50000; use 0 for full run – can take hours)
  --save-index PATH   save built index to PATH after building
  --load-index PATH   load pre-built index; skip build step
"""

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import time

import numpy as np

try:
    from huggingface_hub import hf_hub_download
    HF_HUB = True
except ImportError:
    HF_HUB = False

# ── Dataset registry ──────────────────────────────────────────────────────────
# DATASETS = {
#     "wikipedia-small": {
#         "dir":  "wikipedia-small",
#         "file": "benchmark-dev-wikipedia-bge-m3-small.h5",
#         "size_mb": 682,
#     },
#     "wikipedia": {
#         "dir":  "wikipedia",
#         "file": "benchmark-dev-wikipedia-bge-m3.h5",
#         "size_mb": 14_900,
#     },
# }
REPO_ID   = "SISAP-Challenges/SISAP2026"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHUNK = 100_000   # rows per I/O chunk

# ── Memory estimate table ─────────────────────────────────────────────────────
PREC_BYTES = {"float32": 4, "float16": 2, "int8": 1}
PREC_SUFFIX = {"float32": "_f32", "float16": "_f16", "int8": "_i8"}
# ── Logging ───────────────────────────────────────────────────────────────────
def log(msg, end="\n"):
    print(msg, end=end, flush=True)


# ── Download ──────────────────────────────────────────────────────────────────
# def download(repo_id, hf_path, local_path):
#     if os.path.exists(local_path):
#         log(f"  {os.path.basename(local_path)} already present – skip.")
#         return
#     log(f"  Downloading {hf_path} ...")
#     t0 = time.time()
#     if HF_HUB:
#         cached = hf_hub_download(
#             repo_id=repo_id, filename=hf_path, repo_type="dataset",
#             local_dir=os.path.dirname(local_path) or ".")
#         if os.path.abspath(cached) != os.path.abspath(local_path):
#             shutil.copy2(cached, local_path)
#     else:
#         url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{hf_path}"
#         subprocess.check_call(["wget", "-q", "--show-progress", "-O", local_path, url])
#     log(f"  Done in {time.time()-t0:.0f}s  ({os.path.getsize(local_path)/1e6:.0f} MB)")

def mem_estimate(n, d, precision):
    vecs_gb = n * d * PREC_BYTES[precision] / 1e9
    graph_gb = n * 64 * 4 / 1e9          # flat graph (max_degree=64, id_t=4B)
    return vecs_gb, graph_gb

# ── Binary writers (float16 vectors, int32 ground truth) ──────────────────────

# ── Normalise + quantise ──────────────────────────────────────────────────────
def to_precision(chunk_f32: np.ndarray, precision: str) -> np.ndarray:
    """L2-normalise rows then convert to the requested precision."""
    v = chunk_f32.astype(np.float32, copy=False)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    v /= np.where(n == 0, 1.0, n)
    if precision == "float32":
        return v
    if precision == "float16":
        return v.astype(np.float16)
    # int8: round(v * 127), clipped to [-127, 127]
    return np.clip(np.round(v * 127.0), -127, 127).astype(np.int8)

# ── Binary writers ────────────────────────────────────────────────────────────
def write_vecs(path: str, h5_dset, precision: str):
    """Stream h5_dset -> L2-normalised binary with (N,D) int32 header."""
    N, D = h5_dset.shape
    nb = PREC_BYTES[precision]
    with open(path, "wb") as f:
        f.write(struct.pack("ii", N, D))
        for i in range(0, N, CHUNK):
            raw = h5_dset[i:i+CHUNK]
            f.write(to_precision(raw, precision).tobytes())
            log(f"\r    {min(i+CHUNK, N):>10,}/{N:,}", end="")
    size_mb = (8 + N * D * nb) / 1e6
    log(f"\r    wrote {path}  [{N:,} x {D}]  ({precision}, {size_mb:.1f} MB)")


def write_vecs_sample(path: str, h5_dset, indices: np.ndarray, precision: str):
    """Write a row-subset (indices must be sorted) as the given precision."""
    N, D = len(indices), h5_dset.shape[1]
    nb = PREC_BYTES[precision]
    with open(path, "wb") as f:
        f.write(struct.pack("ii", N, D))
        for i in range(0, N, CHUNK):
            idx_chunk = indices[i:i+CHUNK]
            raw = h5_dset[idx_chunk.tolist()]
            f.write(to_precision(raw, precision).tobytes())
            log(f"\r    {min(i+CHUNK, N):>8,}/{N:,}", end="")
    size_mb = (8 + N * D * nb) / 1e6
    log(f"\r    wrote {path}  [{N:,} x {D}] (sample, {precision}, {size_mb:.1f} MB)")


def write_gt(path: str, arr: np.ndarray):
    """Write int32 ground-truth with (N,K) header."""
    N, K = arr.shape
    with open(path, "wb") as f:
        f.write(struct.pack("ii", N, K))
        f.write(arr.astype(np.int32).tobytes())
    log(f"    wrote {path}  [{N:,} x {K}]  (int32)")


# ── HDF5 tree ─────────────────────────────────────────────────────────────────
def h5_tree(h5, indent=0):
    import h5py
    for key in h5.keys():
        item = h5[key]
        if isinstance(item, h5py.Dataset):
            log(f"{'  '*indent}  /{key}: shape={item.shape} dtype={item.dtype}")
        else:
            log(f"{'  '*indent}  /{key}/")
            h5_tree(item, indent+1)


# ── Compilation ───────────────────────────────────────────────────────────────
def compile_binary(src_cpp, out_bin, include_dir):
    sources = [src_cpp, os.path.join(include_dir, "pipnn_dot.hpp")]
    if (os.path.exists(out_bin)
            and all(not os.path.exists(s)
                    or os.path.getmtime(s) <= os.path.getmtime(out_bin)
                    for s in sources)):
        log(f"  {out_bin} is up to date.")
        return
    log(f"  Compiling {os.path.basename(src_cpp)} ...")
    
    cmd = ["g++", "-O3", "-std=c++17", "-fopenmp", 
           f"-I{include_dir}", src_cpp,"-I/usr/include/hdf5/serial",
            "-L/usr/lib/x86_64-linux-gnu/hdf5/serial",
            "-lhdf5_cpp",
            "-lhdf5", "-march=native", "-o", out_bin]

    log("  " + " ".join(cmd))
    subprocess.check_call(cmd)
    log("  OK")

def bin_path(work: str, name: str, precision: str) -> str:
    """Return precision-stamped binary filename, e.g. sisap_work/train_f16.bin"""
    return os.path.join(work, name + PREC_SUFFIX[precision] + ".bin")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="SISAP 2026 PiPNN benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input",       default="/data/sisap_work/wikipedia-small/benchmark-dev-wikipedia-bge-m3-small.h5",
                    help="dataset path")
    ap.add_argument("--task_description",       default="/data/sisap_work/wikipedia-small/config.json",
                    help="config path")
    ap.add_argument("--precision",     default="float16",
                    choices=["float32", "float16", "int8"],
                    help="Vector storage precision (auto-detected by C++ from file size)")
    ap.add_argument("--bw",            type=int, default=256,
                    help="Query beam width")
    ap.add_argument("--output",          default="sisap_work",
                    help="output directory")
    ap.add_argument("--allknn-sample", type=int, default=0,
                    metavar="N",
                    help="Vectors sampled from train for allknn recall "
                         "(0 = full run, can take hours for wikipedia)")
    ap.add_argument("--save-index",    default=None, metavar="PATH")
    ap.add_argument("--load-index",    default=None, metavar="PATH")
    # ── Graph build hyperparameters (forwarded to sisap_bench) ────────────────
    ap.add_argument("--randomness",          type=int,   default=1,
                        help="Random rbc orphans")
    ap.add_argument("--coocked",          type=int,   default=0,
                        help="Coocked")
    ap.add_argument("--max_degree",    type=int,   default=64,
                        help="Max out-degree per node (R)")
    ap.add_argument("--alpha",         type=float, default=1.2,
                        help="RobustPrune alpha factor [1.0–2.0]")
    ap.add_argument("--leaf_size",     type=int,   default=512,
                        help="Max points per leaf cluster (Cmax)")
    ap.add_argument("--min_leaf_size", type=int,   default=32,
                        help="Leaf merge threshold (Cmin)")
    ap.add_argument("--k_entry",       type=int,   default=12,
                        help="Diverse graph entry points")
    ap.add_argument("--entry_sample",  type=int,   default=3000,
                        help="Sample size for entry-point selection")
    ap.add_argument("--hash_bits",     type=int,   default=12,
                        help="LSH bits for HashPrune (<=16)")
    ap.add_argument("--reservoir_cap", type=int,   default=128,
                        help="HashPrune reservoir size (lmax)")
    ap.add_argument("--num_replicas",  type=int,   default=1,
                        help="Independent RBC graph replications")
    ap.add_argument("--final_prune",   type=int,   default=1,
                        help="Apply RobustPrune after HashPrune (0/1)")
    ap.add_argument("--back_edge",     type=int,   default=1,
                        help="Run back-edge consolidation pass (0/1)")
    ap.add_argument("--num_threads",   type=int,   default=0,
                        help="OMP threads (0 = all available)")
    ap.add_argument("--seed",          type=int,   default=42,
                        help="Random seed")
    args = ap.parse_args()
    prepro_time = time.time()
    # ds   = DATASETS[args.dataset]
    prec, work = args.precision, args.output
    os.makedirs(work, exist_ok=True)
    # Starts data loading/preprocessing
    # ── Download ──────────────────────────────────────────────────────────
    log("=" * 62)
    log(f"Dataset : {args.input}")
    log(f"Precision : {prec}  ({PREC_BYTES[prec]} byte/element)")
    log("=" * 62)

    h5_local = args.input
    # hf_h5    = f"{ds['dir']}/{ds['file']}"
    # download(REPO_ID, hf_h5, h5_local) # Hacer algo que lo descarge desde fuera
    # cfg_local = os.path.join(work, f"config_{args.dataset}.json")
    cfg_local = args.task_description
    try:
        with open(cfg_local) as f:
            cfg_json = json.load(f)
        k    = cfg_json["k"]
        gt_I = cfg_json["gt_I"]
        train_data = cfg_json["data"] # name of the training data
        log(f"  config.json : k={k}  gt_I={gt_I}")
    except:
        log(f"  Couldn't find config file {cfg_local}")
        exit()

    # ── HDF5 inspection & streaming export (float16) ──────────────────────
    log("\n" + "=" * 62)
    log("HDF5 structure")
    log("=" * 62)

    try:
        import h5py
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "h5py"])
        import h5py

    train_bin    = bin_path(work, "train",    prec)
    allknn_q_bin = bin_path(work, "allknn_q", prec)
    allknn_gt_bin= os.path.join(work, "allknn_gt.bin")

    with h5py.File(h5_local, "r") as h5:
        h5_tree(h5)
        log("")

        # ── train ──────────────────────────────────────────────────────
        Nt, D = h5[train_data].shape
        v_gb, g_gb = mem_estimate(Nt, D, prec)
        log(f"train : {Nt:,} x {D}")
        log(f"  RAM estimate: {v_gb:.2f} GB vectors + {g_gb:.2f} GB graph"
            f"  = {v_gb+g_gb:.2f} GB total")
        if not os.path.exists(train_bin):
            log(f"  exporting {os.path.basename(train_bin)} ...")
            write_vecs(train_bin, h5["train"], prec)
        else:
            log(f"  {train_bin} cached.")

        # ── allknn ground truth ────────────────────────────────────────
        log(f"\nallknn knns (gt_I = {gt_I}) ...")
        node = h5
        for part in gt_I:
            node = node[part]
        allknn_gt_raw = node[:].astype(np.int32)
        if allknn_gt_raw.min() > 0:
            log("  1-based -> 0-based")
            allknn_gt_raw -= 1

        # ── allknn queries: sample or full ────────────────────────────
        sample_n = args.allknn_sample
        if sample_n == 0 or sample_n >= Nt:
            log(f"  allknn: full run ({Nt:,} queries)")
            if not os.path.exists(allknn_q_bin):
                write_vecs(allknn_q_bin, h5["train"], prec)
            write_gt(allknn_gt_bin, allknn_gt_raw)
        else:
            # Sort indices so query file and GT rows are in the same order
            rng = np.random.default_rng(42)
            idx = np.sort(rng.choice(Nt, size=sample_n, replace=False))
            log(f"  allknn: sample {sample_n:,}/{Nt:,} train vectors")
            if not os.path.exists(allknn_q_bin):
                write_vecs_sample(allknn_q_bin, h5["train"], idx, prec)
            write_gt(allknn_gt_bin, allknn_gt_raw[idx])
    # ── Compile ───────────────────────────────────────────────────────────
    log("\n" + "=" * 62)
    log("Compile sisap_bench")
    log("=" * 62)
    bench_cpp = os.path.join(SCRIPT_DIR, "sisap_bench.cpp")
    bench_bin = os.path.join(work, "sisap_bench")
    compile_binary(bench_cpp, bench_bin, SCRIPT_DIR)

    # ── Run ───────────────────────────────────────────────────────────────
    log("\n" + "=" * 62)
    log(f"Benchmark  (k={k}  bw={args.bw})")
    log("=" * 62 + "\n")
    args.randomness = 1
    args.coocked = 0
    args.back_edge = 1
    args.final_prune = 1
    args.num_replicas = 1

    prepro_time = time.time() - prepro_time
    cmd = [
        bench_bin,
        train_bin, allknn_q_bin, allknn_gt_bin, str(k),
        str(prepro_time),
         str(args.bw),
        str(args.max_degree), str(args.alpha),
        str(args.leaf_size), str(args.min_leaf_size),
        str(args.k_entry), str(args.entry_sample),
        str(args.hash_bits), str(args.reservoir_cap),
        str(args.num_replicas), str(args.final_prune),
        str(args.back_edge), str(args.num_threads),
        str(args.seed), str(args.randomness), str(args.coocked)
    ]
    if args.save_index:
        cmd += ["--save-index", args.save_index]
    if args.load_index:
        cmd += ["--load-index", args.load_index]

    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
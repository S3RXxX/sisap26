#!/usr/bin/env python3
"""
run_sisap2026.py – PiPNN benchmark on SISAP 2026 wikipedia-small dataset.

Pipeline
────────
1. Download benchmark-dev-wikipedia-bge-m3-small.h5 from HuggingFace
   (skipped if the file already exists locally).
2. Inspect the HDF5 structure and print a summary.
3. Extract and L2-normalise:
      train         – base vectors to index
      itest/queries – in-distribution test queries
      itest ground-truth  (path read from config.json: ["allknn","knns"] or ["itest","knns"])
      otest/queries – out-of-distribution test queries
      otest/knns    – out-of-distribution ground truth
4. Write each split to a simple binary file:
      header: int32 N, int32 D
      data:   N×D float32  (vectors) or  N×K int32  (GT indices, 0-based)
5. Compile sisap_bench.cpp → ./sisap_bench   (once; skipped if up to date).
6. Run ./sisap_bench and stream its output to the terminal.

Requirements
────────────
    pip install h5py numpy huggingface_hub
    g++ ≥ 9 with -std=c++17 -fopenmp -march=native
    pipnn_dot.hpp and sisap_bench.cpp in the same directory as this script.
"""

import os
import sys
import struct
import subprocess
import json
import time
import argparse

import numpy as np

# ── Optional fast download via huggingface_hub ────────────────────────────────
try:
    from huggingface_hub import hf_hub_download
    HF_HUB = True
except ImportError:
    HF_HUB = False

# ── Constants ─────────────────────────────────────────────────────────────────
REPO_ID   = "SISAP-Challenges/SISAP2026"
H5_PATH   = "wikipedia-small/benchmark-dev-wikipedia-bge-m3-small.h5"
H5_LOCAL  = "benchmark-dev-wikipedia-bge-m3-small.h5"
CFG_PATH  = "wikipedia-small/config.json"
K_DEFAULT = 15
BEAM_DEFAULT = 256

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BIN_DIR    = os.path.join(SCRIPT_DIR, "build_sisap")

# ── Helpers ───────────────────────────────────────────────────────────────────
def log(msg):
    print(msg, flush=True)


def download_file(repo_id, repo_path, local_path):
    if os.path.exists(local_path):
        log(f"  {local_path} already present – skipping download.")
        return

    log(f"  Downloading {repo_path} ...")
    t0 = time.time()

    if HF_HUB:
        cached = hf_hub_download(
            repo_id=repo_id,
            filename=repo_path,
            repo_type="dataset",
            local_dir=os.path.dirname(local_path) or ".",
        )
        # hf_hub_download may write to a different path; copy if needed
        if os.path.abspath(cached) != os.path.abspath(local_path):
            import shutil
            shutil.copy2(cached, local_path)
    else:
        url = (f"https://huggingface.co/datasets/{repo_id}"
               f"/resolve/main/{repo_path}")
        subprocess.check_call(["wget", "-q", "--show-progress", "-O",
                                local_path, url])

    log(f"  Done ({time.time()-t0:.1f} s, {os.path.getsize(local_path)/1e6:.0f} MB).")


def l2_normalise(v: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation in float32."""
    v = v.astype(np.float32, copy=False)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (v / norms).astype(np.float32)


def write_vecs(path: str, arr: np.ndarray):
    """Write float32 matrix with (N, D) header."""
    N, D = arr.shape
    with open(path, "wb") as f:
        f.write(struct.pack("ii", N, D))
        f.write(arr.astype(np.float32).tobytes())
    log(f"    wrote {path}  [{N} × {D}]")


def write_gt(path: str, arr: np.ndarray):
    """Write int32 ground-truth matrix with (N, K) header."""
    N, K = arr.shape
    with open(path, "wb") as f:
        f.write(struct.pack("ii", N, K))
        f.write(arr.astype(np.int32).tobytes())
    log(f"    wrote {path}  [{N} × {K}]")


def h5_tree(h5, indent=0):
    """Print HDF5 structure recursively."""
    import h5py
    for key in h5.keys():
        item = h5[key]
        if isinstance(item, h5py.Dataset):
            log(f"{'  '*indent}  /{key}: shape={item.shape} dtype={item.dtype}")
        else:
            log(f"{'  '*indent}  /{key}/")
            h5_tree(item, indent+1)


def resolve_gt_path(h5, path_list):
    """Navigate h5 using a list of keys, e.g. ['allknn','knns']."""
    node = h5
    for key in path_list:
        node = node[key]
    return node


def compile_binary(src_cpp: str, out_bin: str, hpp_dir: str):
    """Compile sisap_bench if the binary is older than the sources."""
    sources = [src_cpp, os.path.join(hpp_dir, "pipnn_dot.hpp")]
    need_rebuild = (
        not os.path.exists(out_bin)
        or any(os.path.getmtime(s) > os.path.getmtime(out_bin)
               for s in sources if os.path.exists(s))
    )
    if not need_rebuild:
        log(f"  {out_bin} is up to date.")
        return

    log(f"  Compiling {src_cpp} ...")
    cmd = [
        "g++", "-O3", "-std=c++17", "-fopenmp", "-march=native",
        f"-I{hpp_dir}",
        src_cpp,
        "-o", out_bin,
    ]
    log("  " + " ".join(cmd))
    subprocess.check_call(cmd)
    log("  Compilation successful.")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="SISAP 2026 wikipedia-small PiPNN benchmark")
    parser.add_argument("--h5",   default=H5_LOCAL,
                        help="Path to local HDF5 file (downloads if absent)")
    parser.add_argument("--k",    type=int, default=K_DEFAULT,
                        help="Number of nearest neighbours to retrieve")
    parser.add_argument("--bw",   type=int, default=BEAM_DEFAULT,
                        help="Beam width for queries")
    parser.add_argument("--work", default="sisap_work",
                        help="Working directory for binary files and binary")
    args = parser.parse_args()

    os.makedirs(args.work, exist_ok=True)

    # ── Step 1: Download ──────────────────────────────────────────────────
    log("=" * 60)
    log("Step 1 – Download")
    log("=" * 60)
    download_file(REPO_ID, H5_PATH,   args.h5)
    # Also try to fetch config.json to know gt_I path
    cfg_local = os.path.join(args.work, "config.json")
    try:
        download_file(REPO_ID, CFG_PATH, cfg_local)
        with open(cfg_local) as f:
            cfg_json = json.load(f)
        gt_I  = cfg_json.get("gt_I",  ["itest", "knns"])
        k_cfg = cfg_json.get("k",     args.k)
        log(f"  config.json: gt_I={gt_I}  k={k_cfg}")
        args.k = k_cfg  # honour the challenge k
    except Exception as e:
        log(f"  Warning: could not read config.json ({e}), using defaults.")
        gt_I = ["itest", "knns"]

    # ── Step 2: Inspect and extract HDF5 ─────────────────────────────────
    log("\n" + "=" * 60)
    log("Step 2 – HDF5 structure")
    log("=" * 60)
    try:
        import h5py
    except ImportError:
        log("Installing h5py ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "h5py"])
        import h5py

    with h5py.File(args.h5, "r") as h5:
        h5_tree(h5)

        # Train
        log("\nLoading train ...")
        train_raw = h5["train"][:]
        log(f"  train: {train_raw.shape} {train_raw.dtype}")


        # allknn queries
        log("Loading allknn queries ...")
        allknn_q_raw = h5["train"][:]
        log(f"  allknn/queries: {allknn_q_raw.shape}")

        # allknn ground truth
        log("Loading allknn ground truth ...")
        allknn_raw = h5["allknn"]["knns"][:]
        log(f"  allknn/knns: {allknn_raw.shape}")

        # itest queries
        log("Loading itest queries ...")
        itest_q_raw = h5["itest"]["queries"][:]
        log(f"  itest/queries: {itest_q_raw.shape}")

        # itest ground truth  (path from config gt_I, e.g. allknn/knns)
        # log(f"Loading itest ground truth from /{'/'.join(gt_I)} ...")
        # try:
        #     igt_raw = resolve_gt_path(h5, gt_I)[:]
        # except KeyError:
        #     # Fallback: try itest/knns
        #     log(f"  Not found, falling back to itest/knns")
        #     igt_raw = h5["itest"]["knns"][:]
        # log(f"  shape: {igt_raw.shape}")
        # itest ground truth
        log("Loading itest ground truth ...")
        igt_raw = h5["itest"]["knns"][:]
        log(f"  itest/knns: {igt_raw.shape}")

        # otest queries
        log("Loading otest queries ...")
        otest_q_raw = h5["otest"]["queries"][:]
        log(f"  otest/queries: {otest_q_raw.shape}")

        # otest ground truth
        log("Loading otest ground truth ...")
        ogt_raw = h5["otest"]["knns"][:]
        log(f"  otest/knns: {ogt_raw.shape}")

    # ── Step 3: Normalise and write binary files ───────────────────────────
    log("\n" + "=" * 60)
    log("Step 3 – Normalise and export to binary")
    log("=" * 60)

    train_norm  = l2_normalise(train_raw)
    allknn_q_n   = l2_normalise(allknn_q_raw)
    itest_q_n   = l2_normalise(itest_q_raw)
    otest_q_n   = l2_normalise(otest_q_raw)

    # SISAP ground truth uses 1-based indices → convert to 0-based
    allknn_0  = allknn_raw.astype(np.int32)
    igt_0  = igt_raw.astype(np.int32)
    ogt_0  = ogt_raw.astype(np.int32)
    if igt_0.min() > 0:
        log("  GT indices appear 1-based; converting to 0-based.")
        igt_0 -= 1
        ogt_0 -= 1
        allknn_0 -= 1

    train_bin  = os.path.join(args.work, "train.bin")
    allknn_q_bin     = os.path.join(args.work, "allknn_q.bin")
    allknn_gt_bin    = os.path.join(args.work, "allknn_gt.bin")
    iq_bin     = os.path.join(args.work, "itest_q.bin")
    igt_bin    = os.path.join(args.work, "itest_gt.bin")
    oq_bin     = os.path.join(args.work, "otest_q.bin")
    ogt_bin    = os.path.join(args.work, "otest_gt.bin")

    write_vecs(train_bin,  train_norm)
    write_vecs(allknn_q_bin, allknn_q_n)
    write_gt  (allknn_gt_bin, allknn_0)
    write_vecs(iq_bin,     itest_q_n)
    write_gt  (igt_bin,    igt_0)
    write_vecs(oq_bin,     otest_q_n)
    write_gt  (ogt_bin,    ogt_0)

    # ── Step 4: Compile ───────────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("Step 4 – Compile sisap_bench")
    log("=" * 60)

    bench_cpp = os.path.join(SCRIPT_DIR, "sisap_bench.cpp")
    bench_bin = os.path.join(args.work,  "sisap_bench")
    compile_binary(bench_cpp, bench_bin, SCRIPT_DIR)

    # ── Step 5: Run ───────────────────────────────────────────────────────
    log("\n" + "=" * 60)
    log(f"Step 5 – Run benchmark  (k={args.k}  bw={args.bw})")
    log("=" * 60 + "\n")

    cmd = [
        bench_bin,
        train_bin, allknn_q_bin, allknn_gt_bin, iq_bin, igt_bin, oq_bin, ogt_bin,
        str(args.k), str(args.bw),
    ]
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()

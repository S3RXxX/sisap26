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
DATASETS = {
    "wikipedia-small": {
        "dir":  "wikipedia-small",
        "file": "benchmark-dev-wikipedia-bge-m3-small.h5",
        "size_mb": 682,
    },
    "wikipedia": {
        "dir":  "wikipedia",
        "file": "benchmark-dev-wikipedia-bge-m3.h5",
        "size_mb": 14_900,
    },
}
REPO_ID   = "SISAP-Challenges/SISAP2026"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Logging ───────────────────────────────────────────────────────────────────
def log(msg, end="\n"):
    print(msg, end=end, flush=True)


# ── Download ──────────────────────────────────────────────────────────────────
def download(repo_id, hf_path, local_path):
    if os.path.exists(local_path):
        log(f"  {os.path.basename(local_path)} already present – skip.")
        return
    log(f"  Downloading {hf_path} ...")
    t0 = time.time()
    if HF_HUB:
        cached = hf_hub_download(
            repo_id=repo_id, filename=hf_path, repo_type="dataset",
            local_dir=os.path.dirname(local_path) or ".")
        if os.path.abspath(cached) != os.path.abspath(local_path):
            shutil.copy2(cached, local_path)
    else:
        url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{hf_path}"
        subprocess.check_call(["wget", "-q", "--show-progress", "-O", local_path, url])
    log(f"  Done in {time.time()-t0:.0f}s  ({os.path.getsize(local_path)/1e6:.0f} MB)")


# ── Normalisation + float16 conversion ────────────────────────────────────────
def l2_norm_f16(v: np.ndarray) -> np.ndarray:
    """L2-normalise rows in float32, then cast to float16."""
    v32 = v.astype(np.float32, copy=False)
    n = np.linalg.norm(v32, axis=1, keepdims=True)
    v32 = v32 / np.where(n == 0, 1.0, n)
    return v32.astype(np.float16)


# ── Binary writers (float16 vectors, int32 ground truth) ──────────────────────
CHUNK = 100_000   # rows per I/O chunk

def write_vecs_f16(path, h5_dset):
    """Stream h5_dset -> L2-normalised float16 binary with (N,D) header."""
    N, D = h5_dset.shape
    with open(path, "wb") as f:
        f.write(struct.pack("ii", N, D))
        for i in range(0, N, CHUNK):
            chunk = l2_norm_f16(h5_dset[i:i+CHUNK])
            f.write(chunk.tobytes())
            log(f"\r    {min(i+CHUNK, N):>10,}/{N:,}", end="")
    size_mb = (8 + N * D * 2) / 1e6
    log(f"\r    wrote {path}  [{N:,} x {D}]  ({size_mb:.1f} MB, float16)")


def write_vecs_f16_sample(path, h5_dset, indices: np.ndarray):
    """Write the row-subset indices (assumed already sorted) of h5_dset as float16."""
    N, D = len(indices), h5_dset.shape[1]
    with open(path, "wb") as f:
        f.write(struct.pack("ii", N, D))
        for i in range(0, N, CHUNK):
            rows = h5_dset[indices[i:i+CHUNK].tolist()]
            f.write(l2_norm_f16(rows).tobytes())
            log(f"\r    {min(i+CHUNK, N):>8,}/{N:,}", end="")
    log(f"\r    wrote {path}  [{N:,} x {D}] (sample, float16)")


def write_gt(path, arr: np.ndarray):
    """Write int32 ground-truth with (N,K) header."""
    N, K = arr.shape
    with open(path, "wb") as f:
        f.write(struct.pack("ii", N, K))
        f.write(arr.astype(np.int32).tobytes())
    log(f"    wrote {path}  [{N:,} x {K}]")


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
    cmd = ["g++", "-O3", "-std=c++17", "-fopenmp", "-march=native",
           f"-I{include_dir}", src_cpp, "-o", out_bin]
    log("  " + " ".join(cmd))
    subprocess.check_call(cmd)
    log("  OK")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",       default="wikipedia-small",
                    choices=list(DATASETS))
    ap.add_argument("--h5",            default=None,
                    help="Override local HDF5 path")
    ap.add_argument("--k",             type=int, default=None)
    ap.add_argument("--bw",            type=int, default=256,
                    help="Query beam width")
    ap.add_argument("--work",          default="sisap_work")
    ap.add_argument("--allknn-sample", type=int, default=50_000,
                    metavar="N",
                    help="Vectors sampled from train for allknn recall "
                         "(0 = full run, can take hours for wikipedia)")
    ap.add_argument("--save-index",    default=None, metavar="PATH")
    ap.add_argument("--load-index",    default=None, metavar="PATH")
    args = ap.parse_args()

    ds   = DATASETS[args.dataset]
    work = args.work
    os.makedirs(work, exist_ok=True)

    # ── Download ──────────────────────────────────────────────────────────
    log("=" * 62)
    log(f"Dataset : {args.dataset}  (~{ds['size_mb']:,} MB HDF5)")
    log("=" * 62)

    h5_local = args.h5 or os.path.join(work, ds["file"])
    hf_h5    = f"{ds['dir']}/{ds['file']}"
    download(REPO_ID, hf_h5, h5_local)

    hf_cfg   = f"{ds['dir']}/config.json"
    cfg_local = os.path.join(work, f"config_{args.dataset}.json")
    try:
        download(REPO_ID, hf_cfg, cfg_local)
        with open(cfg_local) as f:
            cfg_json = json.load(f)
        k    = cfg_json.get("k", 15)
        gt_I = cfg_json.get("gt_I", ["allknn", "knns"])
        log(f"  config.json : k={k}  gt_I={gt_I}")
    except Exception as e:
        log(f"  Warning: config.json unavailable ({e}). Using k=15.")
        k, gt_I = 15, ["allknn", "knns"]

    if args.k:
        k = args.k

    # ── HDF5 inspection & streaming export (float16) ──────────────────────
    log("\n" + "=" * 62)
    log("HDF5 structure")
    log("=" * 62)

    try:
        import h5py
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "h5py"])
        import h5py

    train_bin    = os.path.join(work, "train_f16.bin")
    allknn_q_bin = os.path.join(work, "allknn_q_f16.bin")
    allknn_gt_bin= os.path.join(work, "allknn_gt.bin")
    iq_bin       = os.path.join(work, "itest_q_f16.bin")
    igt_bin      = os.path.join(work, "itest_gt.bin")
    oq_bin       = os.path.join(work, "otest_q_f16.bin")
    ogt_bin      = os.path.join(work, "otest_gt.bin")

    with h5py.File(h5_local, "r") as h5:
        h5_tree(h5)
        log("")

        # ── train ──────────────────────────────────────────────────────
        Nt, D = h5["train"].shape
        log(f"train : {Nt:,} x {D}")
        log(f"  float32: {Nt*D*4/1e9:.2f} GB   float16: {Nt*D*2/1e9:.2f} GB")
        if not os.path.exists(train_bin):
            log("  exporting train_f16.bin ...")
            write_vecs_f16(train_bin, h5["train"])
        else:
            log(f"  {train_bin} cached.")

        # ── allknn ground truth ────────────────────────────────────────
        log(f"\nallknn knns (gt_I = {gt_I}) ...")
        node = h5
        for part in gt_I:
            node = node[part]
        allknn_gt_raw = node[:]
        allknn_0 = allknn_gt_raw.astype(np.int32)
        if allknn_0.min() > 0:
            log("  1-based -> 0-based")
            allknn_0 -= 1

        # ── allknn queries: sample or full ────────────────────────────
        sample_n = args.allknn_sample
        if sample_n == 0 or sample_n >= Nt:
            log(f"  allknn: full run ({Nt:,} queries)")
            if not os.path.exists(allknn_q_bin):
                write_vecs_f16(allknn_q_bin, h5["train"])
            write_gt(allknn_gt_bin, allknn_0)
        else:
            rng = np.random.default_rng(42)
            idx = np.sort(rng.choice(Nt, size=sample_n, replace=False))
            log(f"  allknn: sample {sample_n:,}/{Nt:,} train vectors")
            if not os.path.exists(allknn_q_bin):
                write_vecs_f16_sample(allknn_q_bin, h5["train"], idx)
            write_gt(allknn_gt_bin, allknn_0[idx])

        # ── itest ─────────────────────────────────────────────────────
        log("\nitest ...")
        if not os.path.exists(iq_bin):
            write_vecs_f16(iq_bin, h5["itest"]["queries"])
        else:
            log(f"  {iq_bin} cached.")
        igt_raw = h5["itest"]["knns"][:].astype(np.int32)
        if igt_raw.min() > 0:
            igt_raw -= 1
        write_gt(igt_bin, igt_raw)

        # ── otest ─────────────────────────────────────────────────────
        log("\notest ...")
        if not os.path.exists(oq_bin):
            write_vecs_f16(oq_bin, h5["otest"]["queries"])
        else:
            log(f"  {oq_bin} cached.")
        ogt_raw = h5["otest"]["knns"][:].astype(np.int32)
        if ogt_raw.min() > 0:
            ogt_raw -= 1
        write_gt(ogt_bin, ogt_raw)

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

    cmd = [
        bench_bin,
        train_bin, allknn_q_bin, allknn_gt_bin,
        iq_bin, igt_bin,
        oq_bin, ogt_bin,
        str(k), str(args.bw),
    ]
    if args.save_index:
        cmd += ["--save-index", args.save_index]
    if args.load_index:
        cmd += ["--load-index", args.load_index]

    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
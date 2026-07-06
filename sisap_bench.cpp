/**
 * sisap_bench.cpp  –  PiPNN SISAP 2026 benchmark, pure C++ pipeline.
 *
 * This replaces the old run_sisap2026.py + sisap_bench.cpp pair with a
 * single C++ program. Rationale for the rewrite (see chat for the full
 * write-up):
 *
 *   1. "Query count mismatch 10000 vs. 200000" was caused by a bug in the
 *      old run_split(): IndexDot::query() re-assigns its output vectors on
 *      every call, so after chunking queries in batches of 10000 the `ids`/
 *      `scores` vectors held only the *last* chunk, and the H5 writer was
 *      told n=10000 (the chunk size) instead of the true query count. Fixed
 *      here by running every query in a single call and writing exactly
 *      that many rows.
 *
 *   2. The OOM/SIGKILL on the dev/test datasets came from doubling the
 *      resident data: the old pipeline wrote a full second copy of the
 *      training set to serve as the "allknn" query set, then loaded THAT
 *      copy fully into RAM as float32 (4x its int8 size) just to hand it to
 *      IndexDot::query(). For the full wikipedia dataset that is an extra
 *      ~3.6M x 1024 x 4 bytes ≈ 14.7 GB on top of everything else. This
 *      version never makes a query copy for the all-kNN case: it queries
 *      directly off the same int8 buffer used to build the index via
 *      IndexDot::query_i8(), decoding one row at a time into a small
 *      thread-local scratch buffer.
 *
 *   3. No Python step: this program reads the HDF5 file directly (train
 *      vectors + ground truth) and quantises straight to int8 while
 *      streaming it in, instead of shelling out to a Python export script.
 *      Per the task, the input embeddings are already L2-normalised, so
 *      there is no normalisation pass here — just round(v*127) clipped to
 *      [-127,127].
 *
 * Usage:
 *   ./sisap_bench <input.h5> <train_dataset_name> <gt_dataset_path> <k> \
 *                 <output.h5> <allknn_sample> <bw> <max_degree> <alpha> \
 *                 <leaf_size> <min_leaf_size> <k_entry> <entry_sample> \
 *                 <hash_bits> <reservoir_cap> <num_replicas> <final_prune> \
 *                 <back_edge> <num_threads> <seed> <randomness> <coocked> \
 *                 [--save-index PATH] [--load-index PATH]
 *
 *   <train_dataset_name>  e.g. "train"          (config.json "data")
 *   <gt_dataset_path>     e.g. "otest/knns"      (config.json "gt_I" joined with '/')
 *   <allknn_sample>       0 = use the full training set as queries
 */

#include "pipnn_dot.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <H5Cpp.h>

using clk = std::chrono::steady_clock;
static double sec(clk::time_point t) {
    return std::chrono::duration<double>(clk::now() - t).count();
}

// ─────────────────────────────────────────────────────────────────────────────
// Quantise a row-major float32 chunk to int8 in place: round(v*127), clipped.
// No normalisation — the task's embeddings are already L2-normalised.
// ─────────────────────────────────────────────────────────────────────────────
static void quantize_i8(const float* src, int8_t* dst, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        float v = src[i] * 127.0f;
        v = std::round(v);
        if (v > 127.0f) v = 127.0f;
        if (v < -127.0f) v = -127.0f;
        dst[i] = (int8_t)v;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stream an HDF5 (N,D) float dataset straight into an int8 buffer of size
// N*D, chunk-by-chunk, so we never hold the full float32 copy in RAM.
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<int8_t> load_dataset_i8(H5::H5File& file, const std::string& name,
                                            int& N, int& D, size_t chunk_rows = 100000) {
    H5::DataSet ds = file.openDataSet(name);
    H5::DataSpace fsp = ds.getSpace();
    hsize_t dims[2];
    if (fsp.getSimpleExtentNdims() != 2 || fsp.getSimpleExtentDims(dims) < 0)
        throw std::runtime_error("Dataset '" + name + "' is not 2-D");
    N = (int)dims[0]; D = (int)dims[1];

    std::vector<int8_t> out((size_t)N * D);
    std::vector<float> buf(chunk_rows * D);

    printf("  loading '%s'  (%d x %d) as int8 ...\n", name.c_str(), N, D);
    for (hsize_t off = 0; off < (hsize_t)N; off += chunk_rows) {
        hsize_t rows = std::min<hsize_t>(chunk_rows, (hsize_t)N - off);
        hsize_t offset[2] = {off, 0};
        hsize_t count[2]  = {rows, (hsize_t)D};
        fsp.selectHyperslab(H5S_SELECT_SET, count, offset);
        H5::DataSpace msp(2, count);
        ds.read(buf.data(), H5::PredType::NATIVE_FLOAT, msp, fsp);
        quantize_i8(buf.data(), out.data() + (size_t)off * D, (size_t)rows * D);
        printf("\r    %10zu/%d", (size_t)(off + rows), N);
        fflush(stdout);
    }
    printf("\r    done  (%.2f GB)\n", (out.size()) / 1e9);
    return out;
}

// Ground truth: any integer dtype in the file is fine, HDF5 converts to int32.
static std::vector<int> load_gt(H5::H5File& file, const std::string& path, int& N, int& K) {
    H5::DataSet ds = file.openDataSet(path);
    H5::DataSpace fsp = ds.getSpace();
    hsize_t dims[2];
    if (fsp.getSimpleExtentNdims() != 2 || fsp.getSimpleExtentDims(dims) < 0)
        throw std::runtime_error("GT dataset '" + path + "' is not 2-D");
    N = (int)dims[0]; K = (int)dims[1];
    std::vector<int> out((size_t)N * K);
    ds.read(out.data(), H5::PredType::NATIVE_INT);
    printf("  loaded GT '%s'  (%d x %d)\n", path.c_str(), N, K);
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Result writer (unchanged format expected by eval.py)
// ─────────────────────────────────────────────────────────────────────────────
static void resultH5(const std::vector<int>& knns, const std::vector<float>& dists,
                      int k, int n, double buildT, double preproT, double queryT,
                      pipnn::Config cfg, const std::string& dataset_name, const char* output) {
    H5::H5File file(output, H5F_ACC_TRUNC);

    hsize_t dims[2] = {static_cast<hsize_t>(n), static_cast<hsize_t>(k)};
    H5::DataSpace dataspace(2, dims);

    H5::DataSet knnDataset = file.createDataSet("knns", H5::PredType::NATIVE_INT, dataspace);
    knnDataset.write(knns.data(), H5::PredType::NATIVE_INT);

    H5::DataSet distDataset = file.createDataSet("dists", H5::PredType::NATIVE_FLOAT, dataspace);
    distDataset.write(dists.data(), H5::PredType::NATIVE_FLOAT);

    H5::StrType strdatatype(H5::PredType::C_S1, H5T_VARIABLE);
    H5::DataSpace scalarSpace(H5S_SCALAR);

    {
        H5::Attribute attr = file.createAttribute("algo", strdatatype, scalarSpace);
        std::string value = "PIPNN";
        attr.write(strdatatype, value);
    }
    {
        H5::Attribute attr = file.createAttribute("task", strdatatype, scalarSpace);
        std::string value = "task1";
        attr.write(strdatatype, value);
    }
    // Which dataset this result was produced against — mirrors the key
    // eval.py's _discover_datasets() uses: cfg.get("dataset_name",
    // Path(cfg_path).parent.name). Written under both "dataset_name" (exact
    // match with that dict key) and "dataset" (the older SISAP convention),
    // so eval.py can look results up either way.
    {
        H5::Attribute attr = file.createAttribute("dataset_name", strdatatype, scalarSpace);
        attr.write(strdatatype, dataset_name);
    }
    {
        H5::Attribute attr = file.createAttribute("dataset", strdatatype, scalarSpace);
        attr.write(strdatatype, dataset_name);
    }
    {
        H5::Attribute attr = file.createAttribute("params", strdatatype, scalarSpace);
        std::ostringstream oss;
        oss << ",max_degree=" << cfg.max_degree
            << ",alpha=" << cfg.alpha
            << ",leaf_size=" << cfg.leaf_size
            << ",min_leaf_size=" << cfg.min_leaf_size
            << ",leader_frac=" << cfg.leader_frac
            << ",max_leaders=" << cfg.max_leaders
            << ",knn_k=" << cfg.knn_k
            << ",hash_bits=" << cfg.hash_bits
            << ",reservoir_cap=" << cfg.reservoir_cap
            << ",k_entry=" << cfg.k_entry
            << ",entry_sample=" << cfg.entry_sample
            << ",beam_width=" << cfg.beam_width
            << ",seed=" << cfg.seed;
        std::string value = oss.str();
        attr.write(strdatatype, value);
    }
    {
        H5::Attribute attr = file.createAttribute("buildtime", H5::PredType::NATIVE_DOUBLE, scalarSpace);
        double bpT = buildT + preproT;
        attr.write(H5::PredType::NATIVE_DOUBLE, &bpT);
    }
    {
        H5::Attribute attr = file.createAttribute("querytime", H5::PredType::NATIVE_DOUBLE, scalarSpace);
        attr.write(H5::PredType::NATIVE_DOUBLE, &queryT);
    }
    file.close();
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    if (argc < 24) {
        fprintf(stderr,
            "Usage: %s input.h5 train_dataset gt_path k output.h5 allknn_sample "
            "bw max_degree alpha leaf_size min_leaf_size k_entry entry_sample "
            "hash_bits reservoir_cap num_replicas final_prune back_edge "
            "num_threads seed randomness coocked dataset_name "
            "[--save-index PATH] [--load-index PATH]\n", argv[0]);
        return 1;
    }

    const std::string h5_path   = argv[1];
    const std::string train_ds  = argv[2];
    const std::string gt_path   = argv[3];
    const int   K           = std::stoi(argv[4]);
    const char* output      = argv[5];
    const long  ALLKNN_SAMP = std::stol(argv[6]);
    const int   BW          = std::stoi(argv[7]);
    const int   MAX_DEG     = std::stoi(argv[8]);
    const float ALPHA       = std::stof(argv[9]);
    const int   LEAF_SZ     = std::stoi(argv[10]);
    const int   MIN_LEAF    = std::stoi(argv[11]);
    const int   K_ENTRY     = std::stoi(argv[12]);
    const int   ENT_SAMP    = std::stoi(argv[13]);
    const int   HBITS       = std::stoi(argv[14]);
    const int   RES_CAP     = std::stoi(argv[15]);
    const int   REPLICAS    = std::stoi(argv[16]);
    const bool  F_PRUNE     = std::stoi(argv[17]) != 0;
    const bool  B_EDGE      = std::stoi(argv[18]) != 0;
    const int   N_THREADS   = std::stoi(argv[19]);
    const uint64_t SEED     = (uint64_t)std::stoull(argv[20]);
    const bool  RAND        = std::stoi(argv[21]) != 0;
    const bool  COOCKED     = std::stoi(argv[22]) != 0;
    const std::string DATASET_NAME = argv[23];

    std::string save_idx, load_idx;
    for (int i = 24; i < argc - 1; ++i) {
        if (!std::strcmp(argv[i], "--save-index")) save_idx = argv[++i];
        if (!std::strcmp(argv[i], "--load-index")) load_idx = argv[++i];
    }

#if PIPNN_AVX2
    printf("AVX2+FMA  enabled\n");
#else
    printf("AVX2      disabled\n");
#endif

    auto t_prepro = clk::now();

    printf("Opening %s ...\n", h5_path.c_str());
    H5::H5File file(h5_path, H5F_ACC_RDONLY);

    int Nt, D;
    std::vector<int8_t> train = load_dataset_i8(file, train_ds, Nt, D);

    int Ngt, Kgt;
    std::vector<int> gt = load_gt(file, gt_path, Ngt, Kgt);

    if (Kgt < K) {
        fprintf(stderr, "GT has only %d neighbours per row, need k=%d\n", Kgt, K);
        return 1;
    }

    // ── Query set: reuse the train buffer directly (no copy) for a full
    //    all-kNN run, or gather a small subset when sampling is requested.
    std::vector<int8_t> query_sample; // only populated when sampling
    const int8_t* query_ptr;
    int nq;
    std::vector<int> gt_used; // GT rows aligned with the queries actually used
    if (ALLKNN_SAMP <= 0 || ALLKNN_SAMP >= Nt) {
        printf("  allknn: full run (%d queries, reusing train buffer)\n", Nt);
        query_ptr = train.data();
        nq = Nt;
        gt_used = gt; // already aligned 1:1 with train rows
    } else {
        printf("  allknn: sampling %ld/%d train vectors\n", ALLKNN_SAMP, Nt);
        std::mt19937_64 rng(42);
        std::vector<int> idx(Nt);
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), rng);
        idx.resize(ALLKNN_SAMP);
        std::sort(idx.begin(), idx.end());

        nq = (int)ALLKNN_SAMP;
        query_sample.resize((size_t)nq * D);
        gt_used.resize((size_t)nq * Kgt);
        for (int i = 0; i < nq; ++i) {
            std::memcpy(query_sample.data() + (size_t)i * D,
                        train.data() + (size_t)idx[i] * D, D);
            std::memcpy(gt_used.data() + (size_t)i * Kgt,
                        gt.data() + (size_t)idx[i] * Kgt, Kgt * sizeof(int));
        }
        query_ptr = query_sample.data();
    }

    auto cfg = pipnn::make_config(D, Nt);
    cfg.beam_width     = BW;
    cfg.max_degree     = MAX_DEG;
    cfg.alpha          = ALPHA;
    cfg.leaf_size      = LEAF_SZ;
    cfg.min_leaf_size  = MIN_LEAF;
    cfg.k_entry        = K_ENTRY;
    cfg.entry_sample   = ENT_SAMP;
    cfg.hash_bits      = HBITS;
    cfg.reservoir_cap  = RES_CAP;
    cfg.num_replicas   = REPLICAS;
    cfg.final_prune    = F_PRUNE;
    cfg.back_edge      = B_EDGE;
    cfg.num_threads    = N_THREADS;
    cfg.seed           = SEED;
    cfg.randomness     = RAND;
    cfg.coocked        = COOCKED;

    printf("Config : leaf_size=%-4d  max_degree=%-3d  alpha=%.1f  k_entry=%d  bw=%d\n\n",
           cfg.leaf_size, cfg.max_degree, cfg.alpha, cfg.k_entry, BW);

    double prepro_s = sec(t_prepro);
    printf("Prepro time is %f\n", prepro_s);

    pipnn::IndexDot idx(cfg);
    double build_s = 0.0;
    if (!load_idx.empty() && std::ifstream(load_idx).good()) {
        printf("Loading pre-built index from %s ...\n", load_idx.c_str());
        auto t = clk::now();
        if (!idx.load(load_idx)) { fprintf(stderr, "Load failed\n"); return 1; }
        idx.set_data_i8(train.data());
        printf("  Loaded in %.2f s\n", sec(t));
    } else {
        printf("Building index on %d x %d vectors (int8) ...\n", Nt, D);
        auto t = clk::now();
        idx.build_i8(train.data(), Nt);
        build_s = sec(t);
        auto st = idx.stats();
        printf("  Build time : %.2f s\n", build_s);
        printf("  Avg degree : %.1f\n", st.avg_deg);
        printf("  Bidir      : %.1f%%\n\n", 100.0 * st.frac_bidir);
        if (!save_idx.empty()) {
            printf("Saving index to %s ...\n", save_idx.c_str());
            idx.save(save_idx);
        }
    }

    // ── Query (single call — fixes the old chunked-write bug) ─────────────
    printf("\n-- allknn (%d queries, recall@%d) ----\n", nq, K);
    std::vector<pipnn::id_t> ids;
    std::vector<float> scores;
    auto t = clk::now();
    idx.query_i8(query_ptr, nq, K, ids, scores, BW);
    double query_s = sec(t);

    long total_hits = 0;
    for (int qi = 0; qi < nq; ++qi) {
        const pipnn::id_t* r = ids.data() + (size_t)qi * K;
        const int* g = gt_used.data() + (size_t)qi * Kgt;
        for (int i = 0; i < K; ++i)
            for (int j = 0; j < K; ++j)
                if ((int)r[i] + 1 == g[j]) { ++total_hits; break; }
    }
    double rec = (double)total_hits / (double)((long long)nq * K);
    printf("  %-6d  %.4f     %.0f\n", BW, rec, nq / query_s);

    // Convert ids to 1-based for the output file (matches GT convention),
    // over the FULL result buffer (nq*K), not a hardcoded chunk size.
    std::vector<int> ids_out(ids.size());
    for (size_t i = 0; i < ids.size(); ++i)
        ids_out[i] = (ids[i] == pipnn::NO_ID) ? 0 : (int)ids[i] + 1;

    resultH5(ids_out, scores, K, nq, build_s, prepro_s, query_s, cfg, DATASET_NAME, output);

    printf("\nDone.\n");
    return 0;
}
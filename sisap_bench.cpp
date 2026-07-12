/**
 * sisap_bench.cpp  –  PiPNN SISAP 2026 benchmark, pure C++ pipeline.
 *
 * TIRA-compatible CLI: this binary is invoked directly (no wrapper script)
 * as
 *   sisap_bench --input <dataset.h5> --task-description <config.json> \
 *               --output <output.h5 or output dir>
 *
 * matching the `tira-cli code-submission --command '{EXE} --input
 * $inputDataset/[star].h5 --task-description $inputDataset/config.json
 * --output $outputDir'` convention (written as [star] here only to avoid
 * confusing the C comment parser — use a literal * in the actual command):
 * TIRA supplies the dataset/task-description paths and output location at
 * *run* time, so nothing about the dataset is baked into the image — no
 * more local config_pipnn.json indirection.
 *
 * All PiPNN hyperparameters (BEAM_WIDTH, MAX_DEGREE, ...) are read straight
 * from environment variables (see the Dockerfile ENV defaults), since
 * TIRA's fixed --command template has no room for a hyperparameter sweep of
 * CLI flags — that's exactly what Docker ENV defaults (baked in at build
 * time, or overridden per-run with `docker run -e VAR=value`) are for.
 *
 * Design notes carried over from earlier fixes (see chat history):
 *
 *   1. Queries run in a single call (not chunked), so the H5 writer is
 *      always given the true query count — fixes the old "query count
 *      mismatch" bug.
 *
 *   2. The all-kNN query set reuses the same int8 buffer used to build the
 *      index (via IndexDot::query_i8()) instead of materialising a second
 *      float32 copy of the training set — avoids ~4x the training set's
 *      size in extra RAM.
 *
 *   3. No Python: this program reads the HDF5 file directly and quantises
 *      straight to int8 while streaming it in. The task's embeddings are
 *      already L2-normalised, so there's no normalisation pass — just
 *      round(v*127) clipped to [-127,127].
 *
 *   4. No ground truth is loaded and no recall is computed here — that's
 *      done externally against the output file.
 *
 *   5. Per-leaf graph construction uses a bounded top-k accumulator instead
 *      of a dense O(leaf_size^2) distance matrix — see IndexDot::build_impl
 *      in pipnn_dot.hpp.
 */

#include "pipnn_dot.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <H5Cpp.h>

namespace fs = std::filesystem;

using clk = std::chrono::steady_clock;
static double sec(clk::time_point t) {
    return std::chrono::duration<double>(clk::now() - t).count();
}

// ─────────────────────────────────────────────────────────────────────────────
// Environment variable helpers (hyperparameters come from Docker ENV, not
// argv — TIRA's --command only ever passes --input/--task-description/
// --output).
// ─────────────────────────────────────────────────────────────────────────────
static int env_int(const char* name, int def) {
    const char* v = std::getenv(name);
    return (v && *v) ? std::atoi(v) : def;
}
static long env_long(const char* name, long def) {
    const char* v = std::getenv(name);
    return (v && *v) ? std::atol(v) : def;
}
static float env_float(const char* name, float def) {
    const char* v = std::getenv(name);
    return (v && *v) ? std::atof(v) : def;
}
static uint64_t env_u64(const char* name, uint64_t def) {
    const char* v = std::getenv(name);
    return (v && *v) ? (uint64_t)std::stoull(v) : def;
}
static bool env_bool(const char* name, bool def) {
    const char* v = std::getenv(name);
    return (v && *v) ? (std::atoi(v) != 0) : def;
}

// ─────────────────────────────────────────────────────────────────────────────
// Minimal parser for a FLAT json object of string/number/bool values, e.g.
//   {"k": 16, "data": "train", "dataset_name": "wikipedia-small"}
// This is all config.json ever contains (no more gt_I, so no arrays needed),
// so a hand-rolled parser avoids pulling in a JSON dependency.
// ─────────────────────────────────────────────────────────────────────────────
static std::map<std::string, std::string> parse_flat_json(const std::string& text) {
    std::map<std::string, std::string> kv;
    size_t i = 0, n = text.size();
    auto skip_ws = [&] { while (i < n && std::isspace((unsigned char)text[i])) ++i; };
    auto parse_string = [&]() -> std::string {
        ++i; // opening quote
        std::string s;
        while (i < n && text[i] != '"') {
            if (text[i] == '\\' && i + 1 < n) { s += text[i + 1]; i += 2; }
            else { s += text[i]; ++i; }
        }
        if (i < n) ++i; // closing quote
        return s;
    };
    skip_ws();
    if (i < n && text[i] == '{') ++i;
    while (i < n) {
        skip_ws();
        if (i >= n || text[i] == '}') break;
        if (text[i] != '"') { ++i; continue; }
        std::string key = parse_string();
        skip_ws();
        if (i < n && text[i] == ':') ++i;
        skip_ws();
        std::string value;
        if (i < n && text[i] == '"') {
            value = parse_string();
        } else {
            size_t start = i;
            int depth = 0;
            while (i < n) {
                char c = text[i];
                if (c == '[' || c == '{') ++depth;
                else if (c == ']' || c == '}') { if (depth == 0) break; --depth; }
                else if (c == ',' && depth == 0) break;
                ++i;
            }
            value = text.substr(start, i - start);
            while (!value.empty() && std::isspace((unsigned char)value.back())) value.pop_back();
        }
        kv[key] = value;
        skip_ws();
        if (i < n && text[i] == ',') ++i;
    }
    return kv;
}

static std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open file: " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
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

// ─────────────────────────────────────────────────────────────────────────────
// Result writer (unchanged format expected by eval.py)
// ─────────────────────────────────────────────────────────────────────────────
static void resultH5(const std::vector<int>& knns, const std::vector<float>& dists,
                      int k, int n, double buildT, double preproT, double queryT,
                      pipnn::Config cfg, const std::string& dataset_name,
                      const std::string& task_name, const std::string& output) {
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
        attr.write(strdatatype, task_name);
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

static void print_usage() {
    fprintf(stderr,
        "Usage: sisap_bench --input <dataset.h5> --task-description <config.json> "
        "--output <output.h5 or dir>\n"
        "                 [--allknn-sample N] [--dataset-name NAME]\n"
        "                 [--save-index PATH] [--load-index PATH]\n\n"
        "All PiPNN hyperparameters (BEAM_WIDTH, MAX_DEGREE, ALPHA, ...) are "
        "read from environment variables — see the Dockerfile for the full "
        "list and defaults.\n");
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    std::string input_h5, task_desc, output_arg, save_idx, load_idx, dataset_name_override;
    long allknn_sample_override = -1; // -1 = "not given on the CLI, use env/default"

    for (int i = 1; i < argc; ++i) {
        auto next = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "%s requires a value\n", flag); exit(1); }
            return argv[++i];
        };
        std::string a = argv[i];
        if      (a == "--input")           input_h5 = next("--input");
        else if (a == "--task-description") task_desc = next("--task-description");
        else if (a == "--output")          output_arg = next("--output");
        else if (a == "--allknn-sample")   allknn_sample_override = std::stol(next("--allknn-sample"));
        else if (a == "--dataset-name")    dataset_name_override = next("--dataset-name");
        else if (a == "--save-index")      save_idx = next("--save-index");
        else if (a == "--load-index")      load_idx = next("--load-index");
        else if (a == "-h" || a == "--help") { print_usage(); return 0; }
        else { fprintf(stderr, "Unknown argument: %s\n", a.c_str()); print_usage(); return 1; }
    }
    if (input_h5.empty() || task_desc.empty() || output_arg.empty()) {
        print_usage();
        return 1;
    }

    // ── Parse the task's config.json ({"k":16,"data":"train",...}) ────────
    auto cfg_kv = parse_flat_json(read_file(task_desc));
    if (!cfg_kv.count("k"))    { fprintf(stderr, "config.json missing 'k'\n"); return 1; }
    if (!cfg_kv.count("data")) { fprintf(stderr, "config.json missing 'data'\n"); return 1; }
    // Query/output k+1 neighbors, not k: the challenge's own eval script
    // drops the nearest match before scoring (on an all-kNN run the query
    // set IS the training set, so each point's own nearest "neighbor" is
    // almost always itself; the eval script assumes rank-0 is that
    // self-loop and trims it). Without the +1 here, that trim would leave
    // only k-1 genuine neighbors for every point instead of k.
    const int K = std::stoi(cfg_kv["k"]) + 1;
    const std::string train_ds = cfg_kv["data"];

    // dataset_name: --dataset-name flag > config.json's "dataset_name" >
    // the name of the folder config.json lives in (mirrors eval.py's
    // _discover_datasets(): cfg.get("dataset_name", Path(cfg_path).parent.name))
    std::string dataset_name = !dataset_name_override.empty() ? dataset_name_override
                              : cfg_kv.count("dataset_name") ? cfg_kv["dataset_name"]
                              : fs::path(task_desc).parent_path().filename().string();
    std::string task_name = cfg_kv.count("task") ? cfg_kv["task"] : "task1";

    // ── Resolve output path: a directory (existing, or given with a
    //    trailing slash) gets "results.h5" appended; anything else is used
    //    as an exact file path. Parent directories are created either way.
    fs::path output_path(output_arg);
    bool looks_like_dir = output_arg.back() == '/' || output_arg.back() == '\\'
                          || (fs::exists(output_path) && fs::is_directory(output_path));
    if (looks_like_dir) output_path /= "results.h5";
    if (!output_path.parent_path().empty())
        fs::create_directories(output_path.parent_path());
    const std::string output = output_path.string();

    // ── Hyperparameters from environment (Docker ENV defaults / -e overrides) ──
    const long  ALLKNN_SAMP = allknn_sample_override >= 0 ? allknn_sample_override
                                                           : env_long("ALLKNN_SAMPLE", 0);
    const int   BW          = env_int("BEAM_WIDTH", 32);
    const int   MAX_DEG     = env_int("MAX_DEGREE", 64);
    const float ALPHA       = env_float("ALPHA", 1.2f);
    const int   LEAF_SZ     = env_int("LEAF_SIZE", 512);
    const int   MIN_LEAF    = env_int("MIN_LEAF_SIZE", 32);
    const int   K_ENTRY     = env_int("K_ENTRY", 12);
    const int   ENT_SAMP    = env_int("ENTRY_SAMPLE", 3000);
    const int   HBITS       = env_int("HASH_BITS", 12);
    const int   RES_CAP     = env_int("RESERVOIR_CAP", 128);
    const int   REPLICAS    = env_int("NUM_REPLICAS", 1);
    const bool  F_PRUNE     = env_bool("FINAL_PRUNE", false);
    const bool  B_EDGE      = env_bool("BACK_EDGE", false);
    const int   N_THREADS   = env_int("NUM_THREADS", 0);
    const uint64_t SEED     = env_u64("SEED", 42);
    const bool  RAND        = env_bool("RAND", false);
    const bool  COOCKED     = env_bool("COOCKED", false);

#if PIPNN_AVX2
    printf("AVX2+FMA  enabled\n");
#else
    printf("AVX2      disabled\n");
#endif

    printf("input             = %s\n", input_h5.c_str());
    printf("task_description  = %s\n", task_desc.c_str());
    printf("output            = %s\n", output.c_str());
    printf("dataset_name      = %s\n", dataset_name.c_str());
    printf("task              = %s\n", task_name.c_str());
    printf("k (config+1)      = %d\n\n", K);

    auto t_prepro = clk::now();

    printf("Opening %s ...\n", input_h5.c_str());
    H5::H5File file(input_h5, H5F_ACC_RDONLY);

    int Nt, D;
    std::vector<int8_t> train = load_dataset_i8(file, train_ds, Nt, D);

    // ── Query set: reuse the train buffer directly (no copy) for a full
    //    all-kNN run, or gather a small subset when sampling is requested.
    std::vector<int8_t> query_sample; // only populated when sampling
    const int8_t* query_ptr;
    int nq;
    if (ALLKNN_SAMP <= 0 || ALLKNN_SAMP >= Nt) {
        printf("  allknn: full run (%d queries, reusing train buffer)\n", Nt);
        query_ptr = train.data();
        nq = Nt;
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
        for (int i = 0; i < nq; ++i) {
            std::memcpy(query_sample.data() + (size_t)i * D,
                        train.data() + (size_t)idx[i] * D, D);
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
        printf("PiPNN computing stats...\n"); fflush(stdout);
        auto st = idx.stats();
        printf("PiPNN stats done\n"); fflush(stdout);
        printf("  Build time : %.2f s\n", build_s);
        printf("  Avg degree : %.1f\n", st.avg_deg);
        printf("  Bidir      : %.1f%%\n\n", 100.0 * st.frac_bidir);
        if (!save_idx.empty()) {
            printf("Saving index to %s ...\n", save_idx.c_str());
            idx.save(save_idx);
        }
    }

    // ── Query (single call — every query in one shot, so the result file
    //    always gets the true query count) ──────────────────────────────
    printf("\n-- allknn (%d queries, k=%d) ----\n", nq, K);
    std::vector<pipnn::id_t> ids;
    std::vector<float> scores;
    auto t = clk::now();
    idx.query_i8(query_ptr, nq, K, ids, scores, BW);
    double query_s = sec(t);
    printf("  bw=%-6d  qps=%.0f\n", BW, nq / query_s);

    // Convert ids to 1-based for the output file (a common ground-truth
    // convention). Recall is computed externally against this file, so
    // nothing here needs to know the ground truth.
    std::vector<int> ids_out(ids.size());
    for (size_t i = 0; i < ids.size(); ++i)
        ids_out[i] = (ids[i] == pipnn::NO_ID) ? 0 : (int)ids[i] + 1;

    resultH5(ids_out, scores, K, nq, build_s, prepro_s, query_s, cfg,
              dataset_name, task_name, output);

    printf("\nDone. Wrote %s\n", output.c_str());
    return 0;
}
/**
 * sisap_bench.cpp  –  PiPNN SISAP 2026 benchmark (float16 storage)
 *
 * Binary file format (written by run_sisap2026.py):
 *   header : int32 N, int32 D
 *   data   : N*D  uint16 (float16, vectors)
 *            N*K  int32  (ground-truth, 0-based)
 *
 * Train vectors are memory-mapped as float16 (half the size of float32),
 * so e.g. 3.6M x 1024 = 7.4 GB fits comfortably in 24 GB RAM.
 *
 * Usage:
 *   ./sisap_bench train.bin allknn_q.bin allknn_gt.bin \
 *                itest_q.bin itest_gt.bin \
 *                otest_q.bin otest_gt.bin \
 *                <k> [bw] [--save-index PATH] [--load-index PATH]
 */

#include "pipnn_dot.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef _WIN32
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#  define HAVE_MMAP 1
#endif

using clk = std::chrono::steady_clock;
static double sec(clk::time_point t) {
    return std::chrono::duration<double>(clk::now() - t).count();
}

// ─────────────────────────────────────────────────────────────────────────────
// Memory-mapped float16 vector file
// ─────────────────────────────────────────────────────────────────────────────
struct MmapVecs16 {
    int    n = 0, d = 0;
    size_t map_size = 0;
    void*  map_ptr  = nullptr;
    int    fd_      = -1;
    std::vector<uint16_t> buf_;  // fallback when mmap unavailable

    void open(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("Cannot open: " + path);
        std::fread(&n, 4, 1, f);
        std::fread(&d, 4, 1, f);
        std::fclose(f);

        map_size = 8 + (size_t)n * d * sizeof(uint16_t);

#ifdef HAVE_MMAP
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ >= 0) {
            map_ptr = ::mmap(nullptr, map_size, PROT_READ, MAP_SHARED, fd_, 0);
            if (map_ptr == MAP_FAILED) {
                map_ptr = nullptr; ::close(fd_); fd_ = -1;
            } else {
                ::madvise(map_ptr, map_size, MADV_WILLNEED);
                printf("  mmap (f16): %.2f GB  (%d x %d)\n",
                       map_size / 1e9, n, d);
                return;
            }
        }
#endif
        printf("  loading into RAM (f16): %.2f GB  (%d x %d)\n",
               map_size / 1e9, n, d);
        FILE* f2 = std::fopen(path.c_str(), "rb");
        std::fseek(f2, 8, SEEK_SET);
        buf_.resize((size_t)n * d);
        std::fread(buf_.data(), sizeof(uint16_t), (size_t)n * d, f2);
        std::fclose(f2);
    }

    const uint16_t* data() const {
        if (map_ptr)
            return reinterpret_cast<const uint16_t*>(
                       static_cast<const char*>(map_ptr) + 8);
        return buf_.data();
    }

    ~MmapVecs16() {
#ifdef HAVE_MMAP
        if (map_ptr && map_ptr != MAP_FAILED) ::munmap(map_ptr, map_size);
        if (fd_ >= 0) ::close(fd_);
#endif
    }
    MmapVecs16() = default;
    MmapVecs16(const MmapVecs16&) = delete;
    MmapVecs16& operator=(const MmapVecs16&) = delete;
};

// ─────────────────────────────────────────────────────────────────────────────
// Query vectors – small enough to load fully and convert to float32 once
// ─────────────────────────────────────────────────────────────────────────────
struct QueryVecs {
    int n = 0, d = 0;
    std::vector<float> data;  // converted to float32 for the query() API

    void open(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("Cannot open: " + path);
        std::fread(&n, 4, 1, f);
        std::fread(&d, 4, 1, f);
        std::vector<uint16_t> raw((size_t)n * d);
        std::fread(raw.data(), sizeof(uint16_t), (size_t)n * d, f);
        std::fclose(f);

        data.resize((size_t)n * d);
        pipnn::f16_to_f32(raw.data(), data.data(), n * d);
        printf("  loaded (f16->f32): %.2f MB  (%d x %d)\n",
               data.size()*4/1e6, n, d);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Ground-truth loader
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<int> read_gt(const std::string& path, int& N, int& K) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open GT: " + path);
    f.read(reinterpret_cast<char*>(&N), 4);
    f.read(reinterpret_cast<char*>(&K), 4);
    std::vector<int> g((size_t)N * K);
    f.read(reinterpret_cast<char*>(g.data()), (size_t)N * K * 4);
    return g;
}

// ─────────────────────────────────────────────────────────────────────────────
// Chunked query + recall@k  (only checks against the top-k GT entries)
// ─────────────────────────────────────────────────────────────────────────────
static void run_split(const char* label,
                      pipnn::IndexDot& idx,
                      const float* queries, int nq, int D,
                      const std::vector<int>& gt, int gt_k,
                      int k, const std::vector<int>& bws,
                      int chunk = 20000) {
    printf("\n-- %s (%d queries, recall@%d) --------------------------\n",
           label, nq, k);
    printf("  %-6s  recall@%-2d  QPS\n", "bw", k);

    for (int bw : bws) {
        if (bw < k) continue;

        long   total_hits = 0;
        double total_s    = 0.0;
        std::vector<pipnn::id_t> ids;
        std::vector<float>       scores;

        for (int q0 = 0; q0 < nq; q0 += chunk) {
            int bn = std::min(chunk, nq - q0);
            auto t = clk::now();
            idx.query(queries + (size_t)q0 * D, bn, k, ids, scores, bw);
            total_s += sec(t);

            for (int qi = 0; qi < bn; ++qi) {
                const pipnn::id_t* r = ids.data() + (size_t)qi * k;
                const int*         g = gt.data()  + (size_t)(q0 + qi) * gt_k;
                for (int i = 0; i < k; ++i)
                    for (int j = 0; j < k; ++j)   // only top-k GT entries
                        if ((int)r[i] == g[j]) { ++total_hits; break; }
            }
        }
        double rec = (double)total_hits / (double)((long long)nq * k);
        printf("  %-6d  %.4f     %.0f\n", bw, rec, nq / total_s);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    if (argc < 9) {
        fprintf(stderr,
            "Usage: %s train.bin allknn_q.bin allknn_gt.bin "
            "itest_q.bin itest_gt.bin otest_q.bin otest_gt.bin k [bw] "
            "[--save-index PATH] [--load-index PATH]\n", argv[0]);
        return 1;
    }

    const char* train_f = argv[1];
    const char* allq_f  = argv[2];
    const char* allgt_f = argv[3];
    const char* iq_f    = argv[4];
    const char* igt_f   = argv[5];
    const char* oq_f    = argv[6];
    const char* ogt_f   = argv[7];
    const int   K       = std::stoi(argv[8]);
    int         BW      = (argc >= 10 && argv[9][0] != '-') ? std::stoi(argv[9]) : 256;

    std::string save_idx, load_idx;
    for (int i = 9; i < argc - 1; ++i) {
        if (!std::strcmp(argv[i], "--save-index")) save_idx = argv[++i];
        if (!std::strcmp(argv[i], "--load-index")) load_idx = argv[++i];
    }

#if PIPNN_AVX2
    printf("AVX2+FMA  enabled\n");
#else
    printf("AVX2      disabled (scalar fallback)\n");
#endif

    // ── Load data ─────────────────────────────────────────────────────────
    printf("\nLoading train vectors (float16) ...\n");
    MmapVecs16 train;
    train.open(train_f);
    const int Nt = train.n, D = train.d;

    printf("Loading allknn queries ...\n");
    QueryVecs allq; allq.open(allq_f);

    int Nallgt, Kallgt;
    printf("Loading allknn GT ...\n");
    auto allgt = read_gt(allgt_f, Nallgt, Kallgt);

    printf("Loading itest queries ...\n");
    QueryVecs iq; iq.open(iq_f);

    int Nigt, Kigt;
    printf("Loading itest GT ...\n");
    auto igt = read_gt(igt_f, Nigt, Kigt);

    printf("Loading otest queries ...\n");
    QueryVecs oq; oq.open(oq_f);

    int Nogt, Kogt;
    printf("Loading otest GT ...\n");
    auto ogt = read_gt(ogt_f, Nogt, Kogt);
    printf("\n");

    if (D != allq.d || D != iq.d || D != oq.d) {
        fprintf(stderr, "Dimension mismatch: train=%d allknn=%d itest=%d otest=%d\n",
                D, allq.d, iq.d, oq.d);
        return 1;
    }

    // ── Config ────────────────────────────────────────────────────────────
    auto cfg = pipnn::make_config(D, Nt);
    cfg.beam_width = BW;
    printf("Config : leaf_size=%-4d  max_degree=%-3d  alpha=%.1f  "
           "k_entry=%d  bw=%d\n\n",
           cfg.leaf_size, cfg.max_degree, cfg.alpha, cfg.k_entry, BW);

    // ── Build or load index ───────────────────────────────────────────────
    pipnn::IndexDot idx(cfg);

    if (!load_idx.empty() && std::ifstream(load_idx).good()) {
        printf("Loading pre-built index from %s ...\n", load_idx.c_str());
        auto t = clk::now();
        if (!idx.load(load_idx)) { fprintf(stderr, "Load failed\n"); return 1; }
        idx.set_data_f16(train.data());
        printf("  Loaded in %.2f s\n", sec(t));
    } else {
        printf("Building index on %d x %d vectors (float16) ...\n", Nt, D);
        auto t = clk::now();
        idx.build_f16(train.data(), Nt);
        double build_s = sec(t);
        auto   st      = idx.stats();
        printf("  Build time : %.2f s\n",    build_s);
        printf("  Avg degree : %.1f\n",       st.avg_deg);
        printf("  Bidir      : %.1f%%\n\n",   100.0 * st.frac_bidir);

        if (!save_idx.empty()) {
            printf("Saving index to %s ...\n", save_idx.c_str());
            idx.save(save_idx);
        }
    }

    // ── Beam-width sweep ─────────────────────────────────────────────────
    std::vector<int> bws = { BW/4, BW/2, BW, BW*2 };

    run_split("allknn", idx, allq.data.data(), allq.n, D, allgt, Kallgt, K, bws);
    run_split("itest",  idx, iq.data.data(),   iq.n,   D, igt,   Kigt,   K, bws);
    run_split("otest",  idx, oq.data.data(),   oq.n,   D, ogt,   Kogt,   K, bws);

    printf("\nDone.\n");
    return 0;
}
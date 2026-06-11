/**
 * sisap_bench.cpp – PiPNN dot-product benchmark for SISAP 2026
 *
 * Reads simple binary files written by run_sisap2026.py:
 *   header:  int32 N, int32 D
 *   data:    N*D float32 (vectors) or N*K int32 (ground-truth indices, 0-based)
 *
 * Usage:
 *   ./sisap_bench <train.bin> <allknn_q.bin> <allknn_gt.bin> <itest_q.bin> <itest_gt.bin>
 *                            <otest_q.bin> <otest_gt.bin> <k> [beam_width]
 */
#include "pipnn_dot.hpp"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>

using clk = std::chrono::steady_clock;
static double sec(clk::time_point t) {
    return std::chrono::duration<double>(clk::now() - t).count();
}

// ── Binary I/O ───────────────────────────────────────────────────────────────
static void read_header(std::ifstream& f, int& N, int& D) {
    f.read(reinterpret_cast<char*>(&N), 4);
    f.read(reinterpret_cast<char*>(&D), 4);
    if (!f) throw std::runtime_error("Failed to read header");
}

static std::vector<float> read_vecs(const std::string& path, int& N, int& D) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    read_header(f, N, D);
    std::vector<float> v((size_t)N * D);
    f.read(reinterpret_cast<char*>(v.data()), (size_t)N * D * 4);
    if (!f) throw std::runtime_error("Short read: " + path);
    return v;
}

static std::vector<int> read_gt(const std::string& path, int& N, int& K) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    read_header(f, N, K);
    std::vector<int> g((size_t)N * K);
    f.read(reinterpret_cast<char*>(g.data()), (size_t)N * K * 4);
    if (!f) throw std::runtime_error("Short read: " + path);
    return g;
}

// ── Recall@k  ────────────────────────────────────────────────────────────────
static float recall(const std::vector<pipnn::id_t>& res,
                    const std::vector<int>& gt,
                    int nq, int k, int gt_k) {
    int hits = 0;
    std::cout << "gt_k: " << gt_k << std::endl;
    std::cout << "k: " << k << std::endl;
    for (int qi = 0; qi < nq; ++qi) {
        const pipnn::id_t* r = res.data() + (size_t)qi * k;
        const int*         g = gt.data()  + (size_t)qi * gt_k;
        for (int i = 0; i < k; ++i)
            for (int j = 0; j < k; ++j)
                if ((int)r[i] == g[j]) { ++hits; break; }
    }
    return (float)hits / (float)(nq * k);
}

// ── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    if (argc < 9) {
        fprintf(stderr,
            "Usage: %s train.bin allknn_q.bin allknn_gt.bin "
            "itest_q.bin itest_gt.bin otest_q.bin otest_gt.bin k [bw] "
            "[max_degree] [alpha] [leaf_size] [min_leaf_size] "
            "[k_entry] [entry_sample] [hash_bits] [reservoir_cap] "
            "[num_replicas] [final_prune] [back_edge] "
            "[num_threads] [seed]\n", argv[0]);
        return 1;
    }

    const std::string train_f  = argv[1];
    const std::string allq_f   = argv[2];
    const std::string allgt_f  = argv[3];
    const std::string iq_f     = argv[4];
    const std::string igt_f    = argv[5];
    const std::string oq_f     = argv[6];
    const std::string ogt_f    = argv[7];
    const int         K        = std::stoi(argv[8]);
    const int         BW       = (argc >= 10) ? std::stoi(argv[9])  : 256;
    const int         MAX_DEG  = (argc >= 11) ? std::stoi(argv[10]) : 64;
    const float       ALPHA    = (argc >= 12) ? std::stof(argv[11]) : 1.2f;
    const int         LEAF_SZ  = (argc >= 13) ? std::stoi(argv[12]) : 512;
    const int         MIN_LEAF = (argc >= 14) ? std::stoi(argv[13]) : 32;
    const int         K_ENTRY  = (argc >= 15) ? std::stoi(argv[14]) : 12;
    const int         ENT_SAMP = (argc >= 16) ? std::stoi(argv[15]) : 3000;
    const int         HBITS    = (argc >= 17) ? std::stoi(argv[16]) : 12;
    const int         RES_CAP  = (argc >= 18) ? std::stoi(argv[17]) : 128;
    const int         REPLICAS = (argc >= 19) ? std::stoi(argv[18]) : 1;
    const bool        F_PRUNE  = (argc >= 20) ? (std::stoi(argv[19]) != 0) : true;
    const bool        B_EDGE   = (argc >= 21) ? (std::stoi(argv[20]) != 0) : true;
    const int         N_THREADS= (argc >= 22) ? std::stoi(argv[21]) : 0;
    const uint64_t    SEED     = (argc >= 23) ? (uint64_t)std::stoull(argv[22]) : 42;

#if PIPNN_AVX2
    printf("AVX2+FMA enabled\n");
#else
    printf("Scalar fallback (no AVX2)\n");
#endif

    // ── Load data ─────────────────────────────────────────────────────────
    int Nt, D, Nallq, Dallq, Nallgt, Kgt0, Niq, Diq, Noq, Doq, Nigt, Kgt, Nogt, Kgt2;
    printf("Loading train ... "); fflush(stdout);
    auto train = read_vecs(train_f, Nt, D);
    printf("%d × %d  (%.1f MB)\n", Nt, D, Nt*(float)D*4/1e6);




    printf("Loading allknn queries ... "); fflush(stdout);
    auto allq = read_vecs(allq_f, Nallq, Dallq);
    printf("%d × %d\n", Nallq, Dallq);

    printf("Loading allknn ground truth ... "); fflush(stdout);
    auto allgt = read_gt(allgt_f, Nallgt, Kgt0);
    printf("%d × %d\n", Nallgt, Kgt0);



    printf("Loading itest queries ... "); fflush(stdout);
    auto iq = read_vecs(iq_f, Niq, Diq);
    printf("%d × %d\n", Niq, Diq);

    printf("Loading itest ground truth ... "); fflush(stdout);
    auto igt = read_gt(igt_f, Nigt, Kgt);
    printf("%d × %d\n", Nigt, Kgt);

    printf("Loading otest queries ... "); fflush(stdout);
    auto oq = read_vecs(oq_f, Noq, Doq);
    printf("%d × %d\n", Noq, Doq);

    printf("Loading otest ground truth ... "); fflush(stdout);
    auto ogt = read_gt(ogt_f, Nogt, Kgt2);
    printf("%d × %d\n\n", Nogt, Kgt2);

    if (D != Dallq || D != Diq || D != Doq) {
        fprintf(stderr, "Dimension mismatch: train=%d, allknn=%d, itest=%d, otest=%d\n",
                D, Dallq, Diq, Doq);
        return 1;
    }

    // ── Build ──────────────────────────────────────────────────────────────
    auto cfg = pipnn::make_config(D, Nt);
    cfg.beam_width    = BW;
    cfg.max_degree    = MAX_DEG;
    cfg.alpha         = ALPHA;
    cfg.leaf_size     = LEAF_SZ;
    cfg.min_leaf_size = MIN_LEAF;
    cfg.k_entry       = K_ENTRY;
    cfg.entry_sample  = ENT_SAMP;
    cfg.hash_bits     = HBITS;
    cfg.reservoir_cap = RES_CAP;
    cfg.num_replicas  = REPLICAS;
    cfg.final_prune   = F_PRUNE;
    cfg.back_edge     = B_EDGE;
    cfg.num_threads   = N_THREADS;
    cfg.seed          = SEED;
    printf("Config: leaf_size=%d  max_degree=%d  alpha=%.2f  "
           "k_entry=%d  beam_width=%d  final_prune=%d  back_edge=%d\n"
           "        hash_bits=%d  reservoir_cap=%d  num_replicas=%d  "
           "num_threads=%d  seed=%llu\n\n",
           cfg.leaf_size, cfg.max_degree, cfg.alpha, cfg.k_entry, BW,
           cfg.final_prune, cfg.back_edge, cfg.hash_bits, cfg.reservoir_cap,
           cfg.num_replicas, cfg.num_threads, (unsigned long long)cfg.seed);

    pipnn::IndexDot idx(cfg);
    printf("Building index on %d vectors (dim=%d) ...\n", Nt, D);
    auto t0 = clk::now();
    idx.build(train.data(), Nt);
    double build_s = sec(t0);
    auto   st      = idx.stats();
    printf("  Build time : %.2f s\n", build_s);
    printf("  Avg degree : %.1f\n", st.avg_deg);
    printf("  Bidir      : %.1f%%\n\n", 100.0 * st.frac_bidir);


    // ── allknn queries ─────────────────────────────────────────────────────
    printf("── allknn (%d queries, recall@%d) ──────────────────────────\n",
           Nallq, K);
    printf("  bw     recall@%-2d   QPS\n", K);

    for (int bw : {BW/4, BW/2, BW, BW*2}) {
        if (bw < K) continue;
        std::vector<pipnn::id_t> ids;
        std::vector<float>       scores;
        auto t = clk::now();
        idx.query(allq.data(), Nallq, K, ids, scores, bw);
        double q_s = sec(t);
        float  rec = recall(ids, allgt, Nallq, K, Kgt0);
        printf("  %-6d %.4f      %.0f\n", bw, rec, Nallq / q_s);
    }

    // ── itest queries ─────────────────────────────────────────────────────
    printf("── itest (%d queries, recall@%d) ──────────────────────────\n",
           Niq, K);
    printf("  bw     recall@%-2d   QPS\n", K);

    for (int bw : {BW/4, BW/2, BW, BW*2}) {
        if (bw < K) continue;
        std::vector<pipnn::id_t> ids;
        std::vector<float>       scores;
        auto t = clk::now();
        idx.query(iq.data(), Niq, K, ids, scores, bw);
        double q_s = sec(t);
        float  rec = recall(ids, igt, Niq, K, Kgt);
        printf("  %-6d %.4f      %.0f\n", bw, rec, Niq / q_s);
    }

    // ── otest queries ─────────────────────────────────────────────────────
    printf("\n── otest (%d queries, recall@%d) ──────────────────────────\n",
           Noq, K);
    printf("  bw     recall@%-2d   QPS\n", K);

    for (int bw : {BW/4, BW/2, BW, BW*2}) {
        if (bw < K) continue;
        std::vector<pipnn::id_t> ids;
        std::vector<float>       scores;
        auto t = clk::now();
        idx.query(oq.data(), Noq, K, ids, scores, bw);
        double q_s = sec(t);
        float  rec = recall(ids, ogt, Noq, K, Kgt2);
        printf("  %-6d %.4f      %.0f\n", bw, rec, Noq / q_s);
    }

    printf("\nDone.\n");
    return 0;
}

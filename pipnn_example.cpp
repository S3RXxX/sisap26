/**
 * pipnn_example.cpp  –  demonstrates both Index (v1) and IndexV2 (extended)
 *
 * Compile:
 *   g++ -O3 -std=c++17 -fopenmp -march=native pipnn_example.cpp -o pipnn_example
 *   ./pipnn_example
 */

// #include "pipnn_ext.hpp"   // already includes pipnn.hpp
#include "pipnn_opt.hpp"
#include <chrono>
#include <cstdio>

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<float> rand_vecs(int n, int d, uint64_t seed = 0) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd;
    std::vector<float> v((size_t)n * d);
    float v_norm = 0;
    for (auto& x : v) 
    {
        x = nd(rng);
        v_norm += x*x;
    }
    v_norm = std::sqrt(v_norm);
    for (auto& x : v) x = x/v_norm;
    return v;
}

static std::vector<pipnn::id_t> brute_force_knn(
        const std::vector<float>& data, int n,
        const std::vector<float>& queries, int nq,
        int d, int k) {
    std::vector<pipnn::id_t> gt((size_t)nq * k);
    for (int qi = 0; qi < nq; ++qi) {
        const float* q = queries.data() + (size_t)qi * d;
        std::vector<std::pair<float, uint32_t>> dv(n);
        for (int i = 0; i < n; ++i) {
            const float* p = data.data() + (size_t)i * d;
            float s = 0;
            for (int k2 = 0; k2 < d; ++k2) { float t = q[k2]-p[k2]; s += t*t; }
            dv[i] = {s, (uint32_t)i};
        }
        std::partial_sort(dv.begin(), dv.begin()+k, dv.end());
        for (int ki = 0; ki < k; ++ki)
            gt[(size_t)qi*k + ki] = dv[ki].second;
    }
    return gt;
}


static std::vector<pipnn::id_t> brute_force_mip(
        const std::vector<float>& data, int n,
        const std::vector<float>& queries, int nq,
        int d, int k) {
    std::vector<pipnn::id_t> gt((size_t)nq * k);

    for (int qi = 0; qi < nq; ++qi) {
        const float* q = queries.data() + (size_t)qi * d;
        std::vector<std::pair<float, uint32_t>> scores(n);

        for (int i = 0; i < n; ++i) {
            const float* p = data.data() + (size_t)i * d;
            float ip = 0.0f;
            for (int j = 0; j < d; ++j)
            {
                ip += q[j] * p[j];
            }

            scores[i] = {-ip, (uint32_t)i};
        }

        std::partial_sort(
            scores.begin(),
            scores.begin() + k,
            scores.end()); //,
            // [](const auto& a, const auto& b) {
            //     return a.first > b.first; // descending: larger IP is better
            // });

        for (int ki = 0; ki < k; ++ki)
            gt[(size_t)qi * k + ki] = scores[ki].second;
    }

    return gt;
}

using clk = std::chrono::steady_clock;
static long ms_since(clk::time_point t0) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(clk::now()-t0).count();
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    const int N   = 200'000;
    const int NQ  = 1'000;
    const int DIM = 10;
    const int K   = 10;

    printf("=== PiPNN demo  n=%d  nq=%d  dim=%d  k=%d ===\n\n", N, NQ, DIM, K);

    auto data    = rand_vecs(N,  DIM, 1);
    auto queries = rand_vecs(NQ, DIM, 2);

    // Ground-truth exact k-NN
    printf("Computing ground truth...\n");
    auto t0  = clk::now();
    // auto gt  = brute_force_knn(data, N, queries, NQ, DIM, K);
    auto gt  = brute_force_mip(data, N, queries, NQ, DIM, K);
    printf("  exact k-NN:  %ld ms\n\n", ms_since(t0));








    printf("── opt) pipnn::IndexOpt (back-edges + medoid) ──\n");
    {
        pipnn::IndexOpt::ConfigOpt cfg;
        cfg.dim            = DIM;
        cfg.max_degree     = 64;
        cfg.leaf_size      = 1024;
        cfg.top_fanout     = 10;
        cfg.second_fanout  = 3;
        cfg.knn_k          = 2;
        cfg.hash_bits      = 12;
        cfg.reservoir_cap  = 128;
        cfg.alpha          = 1.2f;
        cfg.beam_width     = 128;
        cfg.back_edge_pass = true;   // <- new
        // cfg.medoid_sample  = 2000;   // <- new

        cfg.use_mips      = true;

        pipnn::IndexOpt idx(cfg);
        t0 = clk::now();
        idx.build(data.data(), N);
        long bms = ms_since(t0);
        printf("  build:   %ld ms\n", bms);

        // Print graph stats
        auto s = idx.stats();
        printf("  graph: avg_deg=%.1f  max_deg=%.0f  bidir=%.1f%%\n",
               s.avg_degree, s.max_degree, 100.0*s.frac_bidirectional);
        printf("  entry point: node %u\n", idx.entry_points()[0]);

        std::vector<pipnn::id_t>   ids;
        std::vector<pipnn::dist_t> dists;
        t0 = clk::now();
        idx.query(queries.data(), NQ, K, ids, dists);
        long qms = ms_since(t0);
        printf("  query:   %ld ms  (%.0f QPS)\n", qms, qms>0?1000.0*NQ/qms:0.);
        printf("  recall@%d: %.4f\n\n", K, pipnn::recall_at_k(ids, gt, K, NQ));

        // ── Save & reload ────────────────────────────────
        printf("  Saving index to /tmp/pipnn_test.bin ...\n");
        if (idx.save("/tmp/pipnn_test.bin")) {
            pipnn::IndexOpt idx2(cfg);
            idx2.load("/tmp/pipnn_test.bin");
            idx2.set_data(data.data());

            ids.clear(); dists.clear();
            idx2.query(queries.data(), NQ, K, ids, dists);
            printf("  recall after reload: %.4f\n\n",
                   pipnn::recall_at_k(ids, gt, K, NQ));
        }
    }



















    // ─────────────────────────────────────────────────────
    // A)  Basic Index  (pipnn.hpp)
    // ─────────────────────────────────────────────────────
    printf("── A) pipnn::Index (v1) ─────────────────────\n");
    {
        pipnn::Config cfg;
        cfg.dim          = DIM;
        cfg.max_degree   = 64;
        cfg.leaf_size    = 1024;
        cfg.top_fanout   = 10;
        cfg.second_fanout= 3;
        cfg.knn_k        = 2;
        cfg.hash_bits    = 12;
        cfg.reservoir_cap= 128;
        cfg.alpha        = 1.2f;
        cfg.beam_width   = 128 *2;

        cfg.use_mips      = true; // true for inner product, false for L2 ////// (ip not working)

        pipnn::Index idx(cfg);
        t0 = clk::now();
        idx.build(data.data(), N);
        printf("  build:   %ld ms\n", ms_since(t0));

        std::vector<pipnn::id_t>   ids;
        std::vector<pipnn::dist_t> dists;
        t0 = clk::now();
        idx.query(queries.data(), NQ, K, ids, dists);
        long qms = ms_since(t0);
        printf("  query:   %ld ms  (%.0f QPS)\n", qms, qms>0?1000.0*NQ/qms:0.);
        printf("  recall@%d: %.4f\n\n", K, pipnn::recall_at_k(ids, gt, K, NQ));
    }

    // ─────────────────────────────────────────────────────
    // B)  IndexV2 with back-edge pass + medoid entry
    // ─────────────────────────────────────────────────────
    printf("── B) pipnn::IndexV2 (back-edges + medoid) ──\n");
    {
        pipnn::IndexV2::ConfigV2 cfg;
        cfg.dim            = DIM;
        cfg.max_degree     = 64;
        cfg.leaf_size      = 1024;
        cfg.top_fanout     = 10;
        cfg.second_fanout  = 3;
        cfg.knn_k          = 2;
        cfg.hash_bits      = 12;
        cfg.reservoir_cap  = 128;
        cfg.alpha          = 1.2f;
        cfg.beam_width     = 128;
        cfg.back_edge_pass = true;   // <- new
        cfg.medoid_sample  = 2000;   // <- new

        cfg.use_mips      = true;

        pipnn::IndexV2 idx(cfg);
        t0 = clk::now();
        idx.build(data.data(), N);
        long bms = ms_since(t0);
        printf("  build:   %ld ms\n", bms);

        // Print graph stats
        auto s = idx.stats();
        printf("  graph: avg_deg=%.1f  max_deg=%.0f  bidir=%.1f%%\n",
               s.avg_degree, s.max_degree, 100.0*s.frac_bidirectional);
        printf("  entry point: node %u\n", idx.entry_point());

        std::vector<pipnn::id_t>   ids;
        std::vector<pipnn::dist_t> dists;
        t0 = clk::now();
        idx.query(queries.data(), NQ, K, ids, dists);
        long qms = ms_since(t0);
        printf("  query:   %ld ms  (%.0f QPS)\n", qms, qms>0?1000.0*NQ/qms:0.);
        printf("  recall@%d: %.4f\n\n", K, pipnn::recall_at_k(ids, gt, K, NQ));

        // ── Save & reload ────────────────────────────────
        printf("  Saving index to /tmp/pipnn_test.bin ...\n");
        if (idx.save("/tmp/pipnn_test.bin")) {
            pipnn::IndexV2 idx2(cfg);
            idx2.load("/tmp/pipnn_test.bin");
            idx2.set_data(data.data());

            ids.clear(); dists.clear();
            idx2.query(queries.data(), NQ, K, ids, dists);
            printf("  recall after reload: %.4f\n\n",
                   pipnn::recall_at_k(ids, gt, K, NQ));
        }
    }

    // ─────────────────────────────────────────────────────
    // C)  Recall vs. beam-width sweep
    // ─────────────────────────────────────────────────────
    printf("── C) Recall vs. beam width (IndexV2) ───────\n");
    {
        pipnn::IndexV2::ConfigV2 cfg;
        cfg.dim            = DIM;
        cfg.max_degree     = 64;
        cfg.leaf_size      = 1024;
        cfg.top_fanout     = 10;
        cfg.second_fanout  = 3;
        cfg.knn_k          = 2;
        cfg.hash_bits      = 12;
        cfg.reservoir_cap  = 128;
        cfg.alpha          = 1.2f;
        cfg.back_edge_pass = true;
        cfg.medoid_sample  = 2000;

        cfg.use_mips      = true;

        pipnn::IndexV2 idx(cfg);
        idx.build(data.data(), N);

        printf("  beam_width  recall@%d   QPS\n", K);
        for (int bw : {32, 64, 128, 256, 512}) {
            std::vector<pipnn::id_t>   ids;
            std::vector<pipnn::dist_t> dists;
            auto t = clk::now();
            idx.query(queries.data(), NQ, K, ids, dists, bw);
            long ms = ms_since(t);
            printf("  %10d  %.4f     %.0f\n",
                   bw,
                   pipnn::recall_at_k(ids, gt, K, NQ),
                   ms > 0 ? 1000.0*NQ/ms : 0.);
        }
        printf("\n");
    }



    return 0;
}

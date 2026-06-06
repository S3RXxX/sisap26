#pragma once
/**
 * pipnn_ext.hpp  –  PiPNN extensions  (include AFTER pipnn.hpp)
 *
 * Adds:
 *   1. Tiled / cache-blocked all-pairs GEMM
 *   2. Thread-local epoch-based visited set  (no per-query allocation)
 *   3. Approximate medoid for a better graph entry point
 *   4. Back-edge consolidation pass  (DiskANN-style second pass)
 *   5. Index serialization: save() / load()
 *   6. Multi-restart beam search  (recall vs. latency knob)
 *   7. Graph diagnostics
 *   8. recall_at_k() evaluation helper
 *
 * pipnn::IndexV2 is a drop-in replacement for pipnn::Index that uses
 * all of the above automatically.
 *
 * Compile:
 *   g++ -O3 -std=c++17 -fopenmp -march=native your_code.cpp
 */

#include "pipnn.hpp"       // needs types, Config, etc.
#include <cstdio>
#include <string>
#include <unordered_set>

namespace pipnn {

// ─────────────────────────────────────────────────────────────────────────────
// 1.  Cache-blocked all-pairs distance
//
//     Tiles the (nA×nB) dot-product loop over k so that the working set for
//     each (i-tile × j-tile × k-tile) fits in L1/L2 cache.
//     BT must be supplied pre-transposed: BT[k*nB + j] = B[j*d + k].
// ─────────────────────────────────────────────────────────────────────────────
static void tiled_gemm_l2(
        const float* __restrict__ A,   int nA,
        const float* __restrict__ BT,  int nB,   // transposed
        const float* __restrict__ na,             // squared norms of A rows
        const float* __restrict__ nb,             // squared norms of B rows
        int d, dist_t* __restrict__ D) {

    // Tile sizes chosen for ~32 KB L1: 16 floats × 16 floats × 64 floats
    constexpr int TI = 16, TJ = 16, TK = 64;

    // Initialise with squared norms
    for (int i = 0; i < nA; ++i) {
        dist_t* Di = D + (size_t)i * nB;
        for (int j = 0; j < nB; ++j) Di[j] = na[i] + nb[j];
    }

    // Subtract 2·A·BT  (tiled)
    for (int i0 = 0; i0 < nA; i0 += TI) {
        const int ie = std::min(i0 + TI, nA);
        for (int j0 = 0; j0 < nB; j0 += TJ) {
            const int je = std::min(j0 + TJ, nB);
            for (int k0 = 0; k0 < d; k0 += TK) {
                const int ke = std::min(k0 + TK, d);
                for (int i = i0; i < ie; ++i) {
                    dist_t*       Di  = D   + (size_t)i  * nB  + j0;
                    const float*  Ai  = A   + (size_t)i  * d   + k0;
                    for (int k = 0; k < ke - k0; ++k) {
                        const float       aik = Ai[k];
                        const float* BTkj     = BT + (size_t)(k0 + k) * nB + j0;
                        for (int j = 0; j < je - j0; ++j)
                            Di[j] -= 2.f * aik * BTkj[j];
                    }
                }
            }
        }
    }
}

/// Wrapper that builds BT and squared norms, then calls tiled_gemm_l2.
static void all_pairs_l2_tiled(const float* A, int nA,
                                const float* B, int nB, int d,
                                std::vector<dist_t>& D) {
    // Transpose B
    std::vector<float> BT((size_t)d * nB);
    for (int j = 0; j < nB; ++j)
        for (int k = 0; k < d; ++k)
            BT[(size_t)k * nB + j] = B[(size_t)j * d + k];

    std::vector<float> na(nA, 0.f), nb(nB, 0.f);
    for (int i = 0; i < nA; ++i)
        for (int k = 0; k < d; ++k) na[i] += A[(size_t)i*d+k] * A[(size_t)i*d+k];
    for (int j = 0; j < nB; ++j)
        for (int k = 0; k < d; ++k) nb[j] += B[(size_t)j*d+k] * B[(size_t)j*d+k];

    D.resize((size_t)nA * nB);
    tiled_gemm_l2(A, nA, BT.data(), nB, na.data(), nb.data(), d, D.data());
}

// ─────────────────────────────────────────────────────────────────────────────
// 2.  Thread-local epoch-based visited tracker
//
//     stamp[u] == epoch  →  node u was visited in the current query.
//     Resetting costs O(1) (just ++epoch), not O(n).
//     Each OMP thread gets its own copy via thread_local.
// ─────────────────────────────────────────────────────────────────────────────
struct EpochVisited {
    // TL storage – intentionally non-static so each EpochVisited instance is
    // independent (safe when used as a local inside a parallel region).
    // We store a pointer into shared TL arrays to avoid repeated TLS lookups.
    static uint32_t*& tl_stamp_ptr() {
        static thread_local uint32_t*   ptr   = nullptr;
        return ptr;
    }
    static size_t& tl_cap() {
        static thread_local size_t cap = 0;
        return cap;
    }
    static uint32_t& tl_epoch() {
        static thread_local uint32_t ep = 0;
        return ep;
    }
    static std::vector<uint32_t>& tl_buf() {
        static thread_local std::vector<uint32_t> buf;
        return buf;
    }

    uint32_t epoch_ = 0;

    /// Call once per query to "reset" – O(1).
    void reset(int n) {
        auto& buf = tl_buf();
        if ((int)buf.size() < n) { buf.assign(n, 0); }
        auto& ep = tl_epoch();
        if (++ep == 0) { std::fill(buf.begin(), buf.end(), 0); ep = 1; }
        epoch_ = ep;
    }

    /// Returns true if u was already visited; marks it visited otherwise.
    bool test_and_mark(id_t u) {
        auto& buf = tl_buf();
        if (buf[u] == epoch_) return true;
        buf[u] = epoch_;
        return false;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// 3.  Approximate medoid
//
//     Samples min(sample_size, n) points, picks the one with the smallest
//     sum of distances to the sample as the graph entry point.
// ─────────────────────────────────────────────────────────────────────────────
inline id_t compute_medoid(const float* data, int n, int d,
                            bool use_mips   = false,
                            int  sample_sz  = 2000,
                            uint64_t seed   = 42) {
    sample_sz = std::min(sample_sz, n);
    std::mt19937_64 rng(seed);

    // Pick sample_sz distinct random indices
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng);
    idx.resize(sample_sz);

    id_t   best_id   = (id_t)idx[0];
    dist_t best_cost = INF_D;

    auto dist_fn = [&](const float* a, const float* b) -> dist_t {
        return use_mips ? neg_dot(a, b, d) : l2_sq(a, b, d);
    };

    for (int si = 0; si < sample_sz; ++si) {
        const float* vi = data + (size_t)idx[si] * d;
        dist_t cost = 0;
        for (int sj = 0; sj < sample_sz; ++sj) {
            if (sj == si) continue;
            cost += dist_fn(vi, data + (size_t)idx[sj] * d);
        }
        if (cost < best_cost) { best_cost = cost; best_id = (id_t)idx[si]; }
    }
    return best_id;
}

// ─────────────────────────────────────────────────────────────────────────────
// 4.  Back-edge consolidation  (DiskANN-style second pass)
//
//     For every directed edge u→v in the current graph, propose the reverse
//     edge v→u.  For each vertex v, merge its current out-neighbours with all
//     proposed back-edges and re-run RobustPrune so the final out-degree stays
//     ≤ max_degree.  This dramatically improves recall at virtually no extra
//     query cost.
// ─────────────────────────────────────────────────────────────────────────────
static void consolidate_back_edges(
        std::vector<std::vector<id_t>>& graph,
        const float* data, int n, int d,
        float alpha, int max_degree, bool use_mips) {

    // Collect proposed reverse edges: back[v] = {(dist(v,u), u)}
    std::vector<std::vector<std::pair<dist_t, id_t>>> back(n);

    for (int u = 0; u < n; ++u) {
        const float* uv = data + (size_t)u * d;
        for (id_t v : graph[u]) {
            if (v == NO_ID) continue;
            dist_t dvu = use_mips ? neg_dot(data + (size_t)v*d, uv, d)
                                  : l2_sq (data + (size_t)v*d, uv, d);
            back[v].emplace_back(dvu, (id_t)u);
        }
    }

    // For each vertex v, union its existing edges with back-edges, re-prune
#pragma omp parallel for schedule(dynamic, 128)
    for (int v = 0; v < n; ++v) {
        if (back[v].empty()) continue;

        const float* vv = data + (size_t)v * d;
        std::vector<std::pair<dist_t, id_t>> cands;
        cands.reserve(graph[v].size() + back[v].size());

        // Existing out-edges
        for (id_t nb : graph[v]) {
            if (nb == NO_ID) continue;
            dist_t dvn = use_mips ? neg_dot(vv, data + (size_t)nb*d, d)
                                  : l2_sq (vv, data + (size_t)nb*d, d);
            cands.emplace_back(dvn, nb);
        }
        // Back-edge proposals
        for (auto& e : back[v]) cands.push_back(e);

        // Deduplicate by id (keep closest distance for each)
        std::sort(cands.begin(), cands.end());
        cands.erase(
            std::unique(cands.begin(), cands.end(),
                        [](const auto& a, const auto& b){ return a.second == b.second; }),
            cands.end());

        robust_prune((id_t)v, cands, data, d,
                     alpha, max_degree, use_mips, graph[v]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5.  Index serialization
//
//     Binary format:
//       [4B] magic   = 0x4E4E4950  ("PINN")
//       [4B] version = 2
//       [4B] n
//       [4B] dim
//       [4B] max_degree
//       [4B] entry_point
//       For each node i in [0, n):
//         [2B] degree_i
//         [4B × degree_i]  neighbour IDs
// ─────────────────────────────────────────────────────────────────────────────
static const uint32_t PIPNN_MAGIC = 0x4E4E4950u; // "PINN"
static const uint32_t PIPNN_VER   = 2u;

struct GraphBundle {
    std::vector<std::vector<id_t>> graph;
    int    n         = 0;
    int    dim       = 0;
    int    max_degree= 0;
    id_t   entry     = 0;
};

inline bool save_graph(const std::string& path, const GraphBundle& gb) {
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) { std::fprintf(stderr, "[PiPNN] cannot open '%s' for write\n", path.c_str()); return false; }

    auto W32 = [&](uint32_t v){ std::fwrite(&v, 4, 1, fp); };
    auto W16 = [&](uint16_t v){ std::fwrite(&v, 2, 1, fp); };

    W32(PIPNN_MAGIC);
    W32(PIPNN_VER);
    W32((uint32_t)gb.n);
    W32((uint32_t)gb.dim);
    W32((uint32_t)gb.max_degree);
    W32(gb.entry);

    for (int i = 0; i < gb.n; ++i) {
        uint16_t deg = (uint16_t)std::min((int)gb.graph[i].size(), 65535);
        W16(deg);
        if (deg) std::fwrite(gb.graph[i].data(), sizeof(id_t), deg, fp);
    }
    std::fclose(fp);
    return true;
}

inline bool load_graph(const std::string& path, GraphBundle& gb) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) { std::fprintf(stderr, "[PiPNN] cannot open '%s' for read\n", path.c_str()); return false; }

    auto R32 = [&](uint32_t& v){ return std::fread(&v, 4, 1, fp) == 1; };
    auto R16 = [&](uint16_t& v){ return std::fread(&v, 2, 1, fp) == 1; };

    uint32_t magic, ver;
    if (!R32(magic) || magic != PIPNN_MAGIC)  { std::fclose(fp); std::fprintf(stderr,"[PiPNN] bad magic\n"); return false; }
    if (!R32(ver)   || ver   != PIPNN_VER)    { std::fclose(fp); std::fprintf(stderr,"[PiPNN] version mismatch (file=%u expected=%u)\n",ver,PIPNN_VER); return false; }

    uint32_t n32, dim32, deg32, entry32;
    R32(n32); R32(dim32); R32(deg32); R32(entry32);
    gb.n          = (int)n32;
    gb.dim        = (int)dim32;
    gb.max_degree = (int)deg32;
    gb.entry      = entry32;
    gb.graph.resize(gb.n);

    for (int i = 0; i < gb.n; ++i) {
        uint16_t d16;
        if (!R16(d16)) { std::fclose(fp); return false; }
        gb.graph[i].resize(d16);
        if (d16) { auto r = std::fread(gb.graph[i].data(), sizeof(id_t), d16, fp); (void)r; }
    }
    std::fclose(fp);
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// 6.  Graph diagnostics
// ─────────────────────────────────────────────────────────────────────────────
struct GraphStats {
    double avg_degree = 0, max_degree = 0, min_degree = 0;
    size_t total_edges = 0;
    double frac_bidirectional = 0;   // fraction of edges that have a reverse edge
};

inline GraphStats graph_stats(const std::vector<std::vector<id_t>>& g, int n) {
    GraphStats s;
    s.min_degree = 1e30;

    // Fast bidirectionality check: for each (u,v) check if v has u in its list
    std::vector<std::unordered_set<id_t>> adj(n);
    for (int u = 0; u < n; ++u)
        for (id_t v : g[u]) adj[u].insert(v);

    size_t bidir = 0, total = 0;
    for (int u = 0; u < n; ++u) {
        int deg = (int)g[u].size();
        s.total_edges += deg;
        s.max_degree   = std::max(s.max_degree, (double)deg);
        s.min_degree   = std::min(s.min_degree, (double)deg);
        for (id_t v : g[u]) {
            ++total;
            if (adj[v].count((id_t)u)) ++bidir;
        }
    }
    s.avg_degree        = (double)s.total_edges / n;
    s.frac_bidirectional= total ? (double)bidir / total : 0.0;
    return s;
}

// ─────────────────────────────────────────────────────────────────────────────
// 7.  recall@k evaluation utility
// ─────────────────────────────────────────────────────────────────────────────
inline float recall_at_k(const std::vector<id_t>& approx,
                          const std::vector<id_t>& exact,
                          int k, int nq) {
    int hits = 0;
    for (int qi = 0; qi < nq; ++qi) {
        const id_t* a = approx.data() + (size_t)qi * k;
        const id_t* e = exact .data() + (size_t)qi * k;
        for (int i = 0; i < k; ++i)
            for (int j = 0; j < k; ++j)
                if (a[i] == e[j]) { ++hits; break; }
    }
    return (float)hits / (float)(nq * k);
}

// ─────────────────────────────────────────────────────────────────────────────
// 8.  Input type converters
// ─────────────────────────────────────────────────────────────────────────────
/// Convert uint8 base vectors to float (e.g., BIGANN dataset).
inline std::vector<float> u8_to_f32(const uint8_t* src, size_t count) {
    std::vector<float> dst(count);
    for (size_t i = 0; i < count; ++i) dst[i] = (float)src[i];
    return dst;
}
/// Convert int8 base vectors to float (e.g., MS-SPACEV dataset).
inline std::vector<float> i8_to_f32(const int8_t* src, size_t count) {
    std::vector<float> dst(count);
    for (size_t i = 0; i < count; ++i) dst[i] = (float)src[i];
    return dst;
}

// ─────────────────────────────────────────────────────────────────────────────
// 9.  IndexV2  –  drop-in replacement for Index with all extensions active
//
//     Additions over Index:
//       • Tiled GEMM for leaf distance matrices
//       • Medoid as entry point (computed after build)
//       • Optional back-edge consolidation pass  (back_edge_pass=true)
//       • Epoch-based visited set in beam search  (no per-query alloc)
//       • Multi-restart beam search  (num_restarts knob)
//       • save() / load() for precomputed indexes
//       • stats() diagnostic method
// ─────────────────────────────────────────────────────────────────────────────
class IndexV2 {
public:
    // Inherits all Config fields; adds two extra knobs.
    struct ConfigV2 : Config {
        bool back_edge_pass = true;   ///< run back-edge consolidation after build
        int  medoid_sample  = 2000;   ///< sample size for entry-point medoid
        int  num_restarts   = 1;      ///< query restarts (1 = single probe)
    };

    explicit IndexV2(ConfigV2 cfg) : cfg_(std::move(cfg)) {
        assert(cfg_.dim > 0    && "ConfigV2::dim must be set");
        assert(cfg_.hash_bits <= 16);
    }

    // ── Build ────────────────────────────────────────────────────────────────
    void build(const float* data, int n) {
        data_ = data;
        n_    = n;
        const int d = cfg_.dim;

#ifdef _OPENMP
        if (cfg_.num_threads > 0) omp_set_num_threads(cfg_.num_threads);
#endif

        // ── LSH sketches ──────────────────────────────────────────────────
        lsh_.init(cfg_.hash_bits, d, cfg_.seed);
        sketches_.resize((size_t)n * cfg_.hash_bits);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i)
            lsh_.compute_sketch(data + (size_t)i * d,
                                sketches_.data() + (size_t)i * cfg_.hash_bits);

        // ── Per-point candidate accumulator ───────────────────────────────
        std::vector<std::vector<std::pair<dist_t, id_t>>> cands(n);
        std::mt19937_64 rng(cfg_.seed);

        for (int rep = 0; rep < cfg_.num_replicas; ++rep) {
            std::vector<id_t> all(n);
            std::iota(all.begin(), all.end(), 0);
            Partition leaves;
            leaves.reserve(n / cfg_.leaf_size * 2);
            rbc_recurse(data, d, all, cfg_, 0, rng, leaves);

            int nl = (int)leaves.size();
            using Edge4 = std::tuple<id_t, id_t, dist_t, dist_t>;
            std::vector<std::vector<Edge4>> leaf_edges(nl);

#pragma omp parallel for schedule(dynamic, 1)
            for (int li = 0; li < nl; ++li) {
                const Leaf& leaf = leaves[li];
                int ln = (int)leaf.size();
                if (ln < 2) continue;

                // Pack leaf data contiguously
                std::vector<float> ld((size_t)ln * d);
                for (int i = 0; i < ln; ++i)
                    std::memcpy(ld.data() + (size_t)i * d,
                                data + (size_t)leaf[i] * d,
                                (size_t)d * sizeof(float));

                // All-pairs distance with tiled GEMM
                std::vector<dist_t> dmat;
                if (cfg_.use_mips)
                    all_pairs_dist(ld.data(), ln, ld.data(), ln, d, true,  dmat);
                else
                    all_pairs_l2_tiled(ld.data(), ln, ld.data(), ln, d, dmat);

                // Bi-directed k-NN candidate edges
                int k = std::min(cfg_.knn_k, ln - 1);
                auto& elist = leaf_edges[li];
                elist.reserve((size_t)ln * k);
                std::vector<int> idx(ln - 1);

                for (int i = 0; i < ln; ++i) {
                    const dist_t* row = dmat.data() + (size_t)i * ln;
                    int cnt = 0;
                    for (int j = 0; j < ln; ++j) if (j != i) idx[cnt++] = j;
                    std::partial_sort(idx.begin(), idx.begin() + k, idx.begin() + cnt,
                                     [&](int a, int b){ return row[a] < row[b]; });
                    for (int ki = 0; ki < k; ++ki) {
                        int j = idx[ki];
                        elist.emplace_back(leaf[i], leaf[j],
                                           row[j], dmat[(size_t)j*ln+i]);
                    }
                }
            }

            // Serial merge into per-point candidate lists
            for (int li = 0; li < nl; ++li)
                for (auto& [src, dst, d_fwd, d_bwd] : leaf_edges[li]) {
                    cands[src].emplace_back(d_fwd, dst);
                    cands[dst].emplace_back(d_bwd, src);
                }
        }

        // ── HashPrune (parallel, race-free) ───────────────────────────────
        std::vector<HashReservoir> res(n);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            res[i].init(cfg_.reservoir_cap);
            const float* si = sketches_.data() + (size_t)i * cfg_.hash_bits;
            for (auto& [dist, j] : cands[i]) {
                if (j == (id_t)i) continue;
                uint16_t h = lsh_.hash_pair(si,
                             sketches_.data() + (size_t)j * cfg_.hash_bits);
                res[i].insert(h, j, dist);
            }
        }

        { std::vector<std::vector<std::pair<dist_t,id_t>>>().swap(cands); }
        { std::vector<float>().swap(sketches_); }

        // ── Final graph (RobustPrune or degree-trim) ──────────────────────
        graph_.resize(n);
        if (cfg_.final_prune) {
#pragma omp parallel for schedule(dynamic, 64)
            for (int i = 0; i < n; ++i) {
                std::vector<std::pair<dist_t, id_t>> c;
                res[i].flush(c);
                robust_prune(i, c, data, d,
                             cfg_.alpha, cfg_.max_degree,
                             cfg_.use_mips, graph_[i]);
            }
        } else {
#pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) {
                std::vector<std::pair<dist_t, id_t>> c;
                res[i].flush(c);
                std::sort(c.begin(), c.end());
                graph_[i].clear();
                for (int j = 0, r = std::min((int)c.size(), cfg_.max_degree);
                     j < r; ++j)
                    graph_[i].push_back(c[j].second);
            }
        }

        // ── Back-edge consolidation ───────────────────────────────────────
        if (cfg_.back_edge_pass)
            consolidate_back_edges(graph_, data, n, d,
                                   cfg_.alpha, cfg_.max_degree, cfg_.use_mips);

        // ── Entry point: approximate medoid ──────────────────────────────
        entry_ = compute_medoid(data, n, d,
                                cfg_.use_mips, cfg_.medoid_sample, cfg_.seed);
    }

    // ── Query ────────────────────────────────────────────────────────────────
    void query(const float* queries, int nq, int k,
               std::vector<id_t>&   out_ids,
               std::vector<dist_t>& out_dists,
               int bw = 0, int restarts = 0) const {
        if (bw       <= 0) bw       = cfg_.beam_width;
        if (restarts <= 0) restarts = cfg_.num_restarts;
        bw = std::max(bw, k);

        out_ids  .assign((size_t)nq * k, NO_ID);
        out_dists.assign((size_t)nq * k, INF_D);

#pragma omp parallel for schedule(dynamic, 1)
        for (int qi = 0; qi < nq; ++qi) {
            const float* q = queries + (size_t)qi * cfg_.dim;
            multi_restart_search(q, k, bw, restarts,
                                 out_ids.data()   + (size_t)qi * k,
                                 out_dists.data() + (size_t)qi * k);
        }
    }

    // ── Save / Load ─────────────────────────────────────────────────────────
    bool save(const std::string& path) const {
        GraphBundle gb;
        gb.graph     = graph_;
        gb.n         = n_;
        gb.dim       = cfg_.dim;
        gb.max_degree= cfg_.max_degree;
        gb.entry     = entry_;
        return save_graph(path, gb);
    }

    /// Load a previously saved graph.  data pointer must be set by the caller
    /// (or call set_data() after loading) so queries work.
    bool load(const std::string& path) {
        GraphBundle gb;
        if (!load_graph(path, gb)) return false;
        n_           = gb.n;
        cfg_.dim     = gb.dim;
        cfg_.max_degree = gb.max_degree;
        graph_       = std::move(gb.graph);
        entry_       = gb.entry;
        return true;
    }

    /// Provide a data pointer after loading an index from disk.
    void set_data(const float* data) { data_ = data; }

    // ── Diagnostics ─────────────────────────────────────────────────────────
    GraphStats stats() const { return graph_stats(graph_, n_); }

    int  num_points()      const { return n_; }
    int  out_degree(int i) const { return (int)graph_[i].size(); }
    id_t entry_point()     const { return entry_; }

private:
    ConfigV2 cfg_;
    int          n_    = 0;
    id_t         entry_= 0;
    const float* data_ = nullptr;
    LSHSketcher  lsh_;
    std::vector<float>             sketches_;
    std::vector<std::vector<id_t>> graph_;

    dist_t dist_fn(const float* a, const float* b) const noexcept {
        return cfg_.use_mips ? neg_dot(a, b, cfg_.dim)
                             : l2_sq(a, b, cfg_.dim);
    }

    // ── Single beam search from a given start node ────────────────────────
    // Uses epoch-based visited (thread-local, O(1) reset).
    std::vector<std::pair<dist_t, id_t>>
    beam_from(const float* q, int L, id_t start) const {
        using P    = std::pair<dist_t, id_t>;
        using MaxQ = std::priority_queue<P>;
        using MinQ = std::priority_queue<P, std::vector<P>, std::greater<P>>;

        EpochVisited vis;
        vis.reset(n_);

        MaxQ best;
        MinQ frontier;

        auto try_push = [&](id_t u) {
            if (vis.test_and_mark(u)) return;
            dist_t dv = dist_fn(q, data_ + (size_t)u * cfg_.dim);
            if ((int)best.size() < L || dv < best.top().first) {
                best.push({dv, u});
                frontier.push({dv, u});
                if ((int)best.size() > L) best.pop();
            }
        };

        try_push(start);

        while (!frontier.empty()) {
            auto [dc, c] = frontier.top(); frontier.pop();
            if ((int)best.size() >= L && dc > best.top().first) break;
            for (id_t nb : graph_[c]) try_push(nb);
        }

        std::vector<P> res;
        while (!best.empty()) { res.push_back(best.top()); best.pop(); }
        std::sort(res.begin(), res.end());
        return res;
    }

    // ── Multi-restart search: merge results from k independent probes ────
    void multi_restart_search(const float* q, int k, int bw, int restarts,
                               id_t* ids, dist_t* dists) const {
        using P = std::pair<dist_t, id_t>;

        // First probe always starts from the medoid
        std::vector<P> merged = beam_from(q, bw, entry_);

        // Additional restarts from random nodes (seeded on query content)
        if (restarts > 1) {
            // Deterministic but query-specific seed
            uint64_t qseed = 0;
            for (int i = 0; i < std::min(cfg_.dim, 8); ++i)
                qseed ^= *reinterpret_cast<const uint32_t*>(q+i) * 2654435761ULL;
            std::mt19937_64 rng(qseed);
            std::uniform_int_distribution<id_t> uid(0, (id_t)(n_ - 1));

            for (int r = 1; r < restarts; ++r) {
                id_t start = uid(rng);
                auto res   = beam_from(q, bw, start);
                for (auto& p : res) merged.push_back(p);
            }

            // Sort + deduplicate by id, keep closest dist for each
            std::sort(merged.begin(), merged.end());
            merged.erase(
                std::unique(merged.begin(), merged.end(),
                            [](const P& a, const P& b){ return a.second == b.second; }),
                merged.end());
        }

        int out_k = std::min((int)merged.size(), k);
        for (int i = 0; i < out_k; ++i) { ids[i] = merged[i].second; dists[i] = merged[i].first; }
        for (int i = out_k; i < k; ++i)  { ids[i] = NO_ID;           dists[i] = INF_D;           }
    }
};

} // namespace pipnn

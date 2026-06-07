#pragma once
/**
 * pipnn.hpp  –  PiPNN: Pick-in-Partitions Nearest Neighbours
 *
 * Header-only C++17 implementation based on:
 *   Rubel et al., "PiPNN: Ultra-Scalable Graph-Based Nearest Neighbor
 *   Indexing", arXiv:2602.21247 (2026).
 *
 * Key ideas implemented
 *   • Randomized Ball Carving (RBC) with multi-level fanout
 *   • Per-leaf all-pairs distance via cache-friendly GEMM (||a-b||² trick)
 *   • Bi-directed k-NN within each leaf as candidate generation
 *   • HashPrune: online, history-independent LSH-based pruning
 *   • Optional final RobustPrune pass (Vamana-style)
 *   • Greedy beam-search query
 *
 * Build:
 *   g++ -O3 -std=c++17 -fopenmp -march=native your_code.cpp
 *
 * Usage:
 *   pipnn::Config cfg;  cfg.dim = 128;
 *   pipnn::Index  idx(cfg);
 *   idx.build(data, n);                       // float* row-major, n vectors
 *
 *   std::vector<uint32_t> ids(nq * k);
 *   std::vector<float>    dists(nq * k);
 *   idx.query(queries, nq, k, ids, dists);    // ascending distance order
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <tuple>
#include <vector>
#ifdef _OPENMP
#  include <omp.h>
#endif

namespace pipnn {

// ─────────────────────────────────────────────────────────────────────────────
// Scalar types
// ─────────────────────────────────────────────────────────────────────────────
using id_t   = uint32_t;
using dist_t = float;

static constexpr id_t   NO_ID = id_t(-1);
static constexpr dist_t INF_D = std::numeric_limits<dist_t>::infinity();

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────
struct Config {
    // ── mandatory ───────────────────────────────────────────────────────────
    int     dim           = 0;    ///< vector dimensionality  (set this!)

    // ── graph ───────────────────────────────────────────────────────────────
    int     max_degree    = 64;   ///< R     – max out-degree per node
    float   alpha         = 1.2f; ///< α     – RobustPrune direction factor

    // ── partitioning ────────────────────────────────────────────────────────
    int     leaf_size     = 1024; ///< C_max – max points per leaf
    int     min_leaf_size = 128;  ///< C_min – leaves below this are merged
    float   leader_frac   = 0.05f;///< fraction of points chosen as leaders
    int     max_leaders   = 1000; ///< hard cap on leaders per sub-problem
    int     top_fanout    = 10;   ///< fanout at recursion depth 0
    int     second_fanout = 3;    ///< fanout at recursion depth 1

    // ── leaf building ───────────────────────────────────────────────────────
    int     knn_k         = 2;    ///< k for bi-directed k-NN inside leaves

    // ── HashPrune ───────────────────────────────────────────────────────────
    int     hash_bits     = 12;   ///< m   – LSH projection count  (≤ 16)
    int     reservoir_cap = 128;  ///< l_max – reservoir capacity per node

    // ── build ────────────────────────────────────────────────────────────────
    int     num_replicas  = 1;    ///< independent RBC replications
    bool    final_prune   = true; ///< apply RobustPrune after HashPrune
    bool    use_mips      = false;///< inner-product distance instead of L2
    int     num_threads   = 0;    ///< 0 = all available OMP threads
    uint64_t seed         = 42;

    // ── query ────────────────────────────────────────────────────────────────
    int     beam_width    = 128;  ///< default beam width L for beam search
};

// ─────────────────────────────────────────────────────────────────────────────
// Distance primitives
// ─────────────────────────────────────────────────────────────────────────────
inline dist_t l2_sq(const float* __restrict__ a,
                    const float* __restrict__ b, int d) noexcept {
    dist_t s = 0;
    for (int i = 0; i < d; ++i) { float t = a[i]-b[i]; s += t*t; }
    return s;
}
inline dist_t neg_dot(const float* __restrict__ a,
                      const float* __restrict__ b, int d) noexcept {
    dist_t s = 0;
    for (int i = 0; i < d; ++i) 
    {
        s += a[i]*b[i];
    }
    return s; // negated so smaller = more similar
}

// ─────────────────────────────────────────────────────────────────────────────
// All-pairs distance matrix via GEMM trick
//
//   ||a – b||² = ||a||² + ||b||² – 2·<a,b>
//
// A: nA×d,  B: nB×d  →  D[i*nB+j] = dist(A[i], B[j])
// B is transposed internally for cache-friendly access.
// ─────────────────────────────────────────────────────────────────────────────
static void all_pairs_dist(const float* A, int nA,
                           const float* B, int nB, int d,
                           bool use_mips,
                           std::vector<dist_t>& D) {
    // Transpose B: BT[k*nB+j] = B[j*d+k]  → inner loop is sequential

    std::vector<float> BT((size_t)d * nB);
    for (int j = 0; j < nB; ++j)
        for (int k = 0; k < d; ++k)
            BT[(size_t)k * nB + j] = B[(size_t)j * d + k];

    D.assign((size_t)nA * nB, 0.f);

    if (use_mips) {
            
        // D[i][j] = –<A[i], B[j]>
        for (int i = 0; i < nA; ++i) {
            dist_t* Di       = D.data() + (size_t)i * nB;
            const float* Ai  = A + (size_t)i * d;
            for (int k = 0; k < d; ++k) {
                float aik            = Ai[k];
                const float* BTk     = BT.data() + (size_t)k * nB;
                for (int j = 0; j < nB; ++j) Di[j] += aik * BTk[j];
            }
        }
    } else {
        // Squared norms
        std::vector<float> na(nA, 0.f), nb(nB, 0.f);
        for (int i = 0; i < nA; ++i)
            for (int k = 0; k < d; ++k) na[i] += A[(size_t)i*d+k] * A[(size_t)i*d+k];
        for (int j = 0; j < nB; ++j)
            for (int k = 0; k < d; ++k) nb[j] += B[(size_t)j*d+k] * B[(size_t)j*d+k];

        for (int i = 0; i < nA; ++i) {
            dist_t* Di      = D.data() + (size_t)i * nB;
            const float* Ai = A + (size_t)i * d;
            for (int j = 0; j < nB; ++j) Di[j] = na[i] + nb[j];
            for (int k = 0; k < d; ++k) {
                float aik        = Ai[k];
                const float* BTk = BT.data() + (size_t)k * nB;
                for (int j = 0; j < nB; ++j) Di[j] -= 2.f * aik * BTk[j];
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RobustPrune  (Algorithm 2 in the paper / Vamana-style)
//
// Given a candidate list for point p, retain at most R directionally
// diverse neighbors according to the α-RNG rule.
// Candidates with id == NO_ID are treated as already removed.
// ─────────────────────────────────────────────────────────────────────────────
static void robust_prune(
        id_t p,
        std::vector<std::pair<dist_t, id_t>>& cands, // may be modified
        const float* data, int d,
        float alpha, int R, bool use_mips,
        std::vector<id_t>& out) {

    std::sort(cands.begin(), cands.end());
    out.clear();
    const float* pv = data + (size_t)p * d;

    for (auto& [dp, y] : cands) {
        if (y == NO_ID || y == p) continue;
        if ((int)out.size() >= R) break;

        id_t yid = y;
        y = NO_ID;                              // mark as consumed
        out.push_back(yid);

        const float* yv = data + (size_t)yid * d;
        for (auto& [dz, z] : cands) {
            if (z == NO_ID) continue;
            const float* zv = data + (size_t)z * d;
            dist_t dyz = use_mips ? neg_dot(yv, zv, d) : l2_sq(yv, zv, d);
            if (alpha * dyz >= dz) z = NO_ID;    // prune z
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HashPrune Reservoir  (Section 3 of the paper)
//
// Online, history-independent pruning.  Each point maintains a small
// reservoir of capacity l_max.  Candidates are hashed via LSH into
// directional buckets; within each bucket only the nearest point is kept.
// When the reservoir is full the globally furthest entry is evicted if the
// incoming candidate is closer.
// ─────────────────────────────────────────────────────────────────────────────
struct HashReservoir {
    struct Slot { uint16_t hash; id_t id; dist_t dist; };

    int cap = 0, sz = 0;
    std::vector<Slot> buf;

    void init(int c) { cap = c; sz = 0; buf.resize(c); }

    void insert(uint16_t h, id_t id, dist_t d) {
        // O(sz) scan – sz ≤ 128, fast in practice
        for (int i = 0; i < sz; ++i) {
            if (buf[i].hash == h) {
                if (d >= buf[i].dist) buf[i] = {h, id, d};
                return;
            }
        }
        if (sz < cap) { buf[sz++] = {h, id, d}; return; }
        // Full: evict furthest if new is closer
        int fi = 0;
        for (int i = 1; i < sz; ++i)
            if (buf[i].dist <= buf[fi].dist) fi = i;
        if (d >= buf[fi].dist) buf[fi] = {h, id, d};
    }

    void flush(std::vector<std::pair<dist_t, id_t>>& out) const {
        out.clear();
        out.reserve(sz);
        for (int i = 0; i < sz; ++i) out.emplace_back(buf[i].dist, buf[i].id);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// LSH sketcher
//
// Generates m random hyperplanes.  For each point v computes
//   Sketch(v)[i] = <H_i, v>
// Hash h_p(c) packs sign(Sketch(c)[i] – Sketch(p)[i]) into m bits.
// ─────────────────────────────────────────────────────────────────────────────
struct LSHSketcher {
    int m = 0, d = 0;
    std::vector<float> H;          // m×d, each row is a unit hyperplane

    void init(int m_, int d_, uint64_t seed) {
        m = m_; d = d_;
        H.resize((size_t)m * d);
        std::mt19937_64 rng(seed);
        std::normal_distribution<float> nd;
        for (auto& v : H) v = nd(rng);
        // L2-normalise rows
        for (int i = 0; i < m; ++i) {
            float norm = 0;
            float* hi = H.data() + (size_t)i * d;
            for (int k = 0; k < d; ++k) norm += hi[k]*hi[k];
            norm = std::sqrt(norm) + 1e-9f;
            for (int k = 0; k < d; ++k) hi[k] /= norm;
        }
    }

    // sketch[i] = dot(H_i, v)
    void compute_sketch(const float* v, float* sketch) const {
        for (int i = 0; i < m; ++i) {
            float acc = 0;
            const float* hi = H.data() + (size_t)i * d;
            for (int k = 0; k < d; ++k) acc += hi[k] * v[k];
            sketch[i] = acc;
        }
    }

    // h_p(c) = bit-packed sign(sketch_c – sketch_p)
    uint16_t hash_pair(const float* sp, const float* sc) const {
        uint16_t h = 0;
        for (int i = 0; i < m; ++i)
            if (sc[i] >= sp[i]) h |= uint16_t(1u << i);
        return h;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Randomized Ball Carving (RBC)  –  Algorithm 5 in the paper
//
// Recursively partitions points by assigning each point to its nearest
// <fanout> randomly chosen leader nodes.  Returns a (possibly overlapping)
// list of leaves, each with at most C_max points.
// ─────────────────────────────────────────────────────────────────────────────
using Leaf      = std::vector<id_t>;
using Partition = std::vector<Leaf>;

static void rbc_recurse(
        const float* data, int d,
        const std::vector<id_t>& pts,
        const Config& cfg,
        int depth,
        std::mt19937_64& rng,
        Partition& out_leaves) {

    int n = (int)pts.size();
    if (n <= cfg.leaf_size) { out_leaves.push_back(pts); return; }

    // ── Sample leaders ───────────────────────────────────────────────────────
    int nl = std::min((int)(cfg.leader_frac * n), cfg.max_leaders);
    nl = std::max(nl, 2);

    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    perm.resize(nl);

    std::vector<id_t> leaders(nl);
    for (int i = 0; i < nl; ++i) leaders[i] = pts[perm[i]];

    // ── Fanout at this depth ─────────────────────────────────────────────────
    int fanout = 1;
    if      (depth == 0) fanout = cfg.top_fanout;
    else if (depth == 1) fanout = cfg.second_fanout;
    fanout = std::min(fanout, nl);

    // ── Assign each point to its nearest <fanout> leaders ───────────────────
    std::vector<std::vector<id_t>> buckets(nl);
    for (int i = 0; i < n; ++i) {
        const float* pi = data + (size_t)pts[i] * d;
        std::vector<std::pair<dist_t, int>> ld(nl);
        for (int j = 0; j < nl; ++j) {
            const float* lj = data + (size_t)leaders[j] * d;
            dist_t dd = cfg.use_mips ? neg_dot(pi, lj, d) : l2_sq(pi, lj, d);
            ld[j] = {dd, j};
        }
        std::partial_sort(ld.begin(), ld.begin() + fanout, ld.end(), std::greater<>());
        for (int f = 0; f < fanout; ++f)
            buckets[ld[f].second].push_back(pts[i]);
    }

    // ── Merge tiny buckets into the largest available bucket ─────────────────
    std::vector<id_t> orphans;
    for (auto& b : buckets) {
        if ((int)b.size() < cfg.min_leaf_size) {
            for (id_t p : b) orphans.push_back(p);
            b.clear();
        }
    }
    if (!orphans.empty()) {
        std::vector<int> valid;
        for (int i = 0; i < nl; ++i)
            if (!buckets[i].empty()) valid.push_back(i);
        if (!valid.empty())
            for (int oi = 0; oi < (int)orphans.size(); ++oi)
                buckets[valid[oi % valid.size()]].push_back(orphans[oi]);
    }

    // ── Recurse ──────────────────────────────────────────────────────────────
    for (auto& b : buckets)
        if (!b.empty())
            rbc_recurse(data, d, b, cfg, depth + 1, rng, out_leaves);
}

// ─────────────────────────────────────────────────────────────────────────────
// PiPNN Index
// ─────────────────────────────────────────────────────────────────────────────
class Index {
public:
    explicit Index(Config cfg) : cfg_(std::move(cfg)) {
        assert(cfg_.dim > 0    && "Config::dim must be set before building");
        assert(cfg_.hash_bits <= 16 && "hash_bits must be ≤ 16");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Build
    // data: row-major float matrix of shape [n, dim]
    // ─────────────────────────────────────────────────────────────────────────
    void build(const float* data, int n) {
        data_ = data;
        n_    = n;

#ifdef _OPENMP
        if (cfg_.num_threads > 0) omp_set_num_threads(cfg_.num_threads);
#endif

        const int d   = cfg_.dim;
        const int m   = cfg_.hash_bits;

        // ── 1. Compute LSH sketches for all points ────────────────────────────
        lsh_.init(m, d, cfg_.seed);
        sketches_.resize((size_t)n * m);

#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i)
            lsh_.compute_sketch(data + (size_t)i * d,
                                sketches_.data() + (size_t)i * m);

        // ── 2. RBC partitioning + leaf edge collection ────────────────────────
        // We collect directed edges (src → dst, dist) from all leaves/replicas.
        // Using per-replica/leaf edge lists and a final serial merge avoids
        // data races while keeping parallel leaf processing.

        // edge_cands[src] = vector of (dist, dst) — grown from leaves
        std::vector<std::vector<std::pair<dist_t, id_t>>> edge_cands(n);

        std::mt19937_64 rng(cfg_.seed);

        for (int rep = 0; rep < cfg_.num_replicas; ++rep) {

            // RBC partitioning (single-threaded – cheap, dominated by leaf work)
            std::vector<id_t> all(n);
            std::iota(all.begin(), all.end(), 0);
            Partition leaves;
            leaves.reserve(n / cfg_.leaf_size * 2);
            rbc_recurse(data, d, all, cfg_, 0, rng, leaves);

            int nl = (int)leaves.size();

            // Parallel leaf processing: build k-NN + collect bi-directed edges
            // We keep per-leaf edge lists to avoid write conflicts.
            using Edge4 = std::tuple<id_t, id_t, dist_t, dist_t>; // src,dst,d_fwd,d_bwd
            std::vector<std::vector<Edge4>> leaf_edges(nl);

#pragma omp parallel for schedule(dynamic, 1)
            for (int li = 0; li < nl; ++li) {
                const Leaf& leaf = leaves[li];
                int ln = (int)leaf.size();
                if (ln < 2) continue;

                // Pack leaf into contiguous buffer for GEMM
                std::vector<float> ldata((size_t)ln * d);
                for (int i = 0; i < ln; ++i)
                    std::memcpy(ldata.data() + (size_t)i * d,
                                data + (size_t)leaf[i] * d,
                                (size_t)d * sizeof(float));

                // All-pairs distance matrix via GEMM trick
                std::vector<dist_t> dmat;
                all_pairs_dist(ldata.data(), ln, ldata.data(), ln, d,
                               cfg_.use_mips, dmat);

                // Bi-directed k-NN edges within leaf
                int k = std::min(cfg_.knn_k, ln - 1);
                auto& elist = leaf_edges[li];
                elist.reserve((size_t)ln * k);

                // Reusable index array for partial sort
                std::vector<int> idx(ln - 1);

                for (int i = 0; i < ln; ++i) {
                    const dist_t* row = dmat.data() + (size_t)i * ln;

                    // Build index of all j ≠ i, then partial-sort by row[j]
                    int cnt = 0;
                    for (int j = 0; j < ln; ++j) if (j != i) idx[cnt++] = j;
                    std::partial_sort(idx.begin(), idx.begin() + k, idx.begin() + cnt,
                                     [&](int a, int b){ return row[a] < row[b]; });

                    for (int ki = 0; ki < k; ++ki) {
                        int j = idx[ki];
                        elist.emplace_back(
                            leaf[i], leaf[j],
                            row[j],                            // d(i→j)
                            dmat[(size_t)j * ln + i]);         // d(j→i)
                    }
                }
            }

            // Serial merge of leaf edges into per-point candidate lists
            for (int li = 0; li < nl; ++li) {
                for (auto& [src, dst, d_fwd, d_bwd] : leaf_edges[li]) {
                    edge_cands[src].emplace_back(d_fwd, dst);
                    edge_cands[dst].emplace_back(d_bwd, src);
                }
            }
        } // replicas

        // ── 3. HashPrune per point (parallel, race-free) ──────────────────────
        std::vector<HashReservoir> reservoirs(n);

#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            reservoirs[i].init(cfg_.reservoir_cap);
            const float* si = sketches_.data() + (size_t)i * m;
            for (auto& [dist, j] : edge_cands[i]) {
                if (j == (id_t)i) continue;
                const float* sj = sketches_.data() + (size_t)j * m;
                uint16_t h = lsh_.hash_pair(si, sj);
                reservoirs[i].insert(h, j, dist);
            }
        }

        // Free edge candidates – no longer needed
        { std::vector<std::vector<std::pair<dist_t,id_t>>>().swap(edge_cands); }
        { std::vector<float>().swap(sketches_); }

        // ── 4. Build final adjacency (+ optional RobustPrune) ─────────────────
        graph_.resize(n);

        if (cfg_.final_prune) {
#pragma omp parallel for schedule(dynamic, 64)
            for (int i = 0; i < n; ++i) {
                std::vector<std::pair<dist_t, id_t>> cands;
                reservoirs[i].flush(cands);
                robust_prune(i, cands, data, d,
                             cfg_.alpha, cfg_.max_degree,
                             cfg_.use_mips, graph_[i]);
            }
        } else {
#pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) {
                std::vector<std::pair<dist_t, id_t>> cands;
                reservoirs[i].flush(cands);
                std::sort(cands.begin(), cands.end());
                auto& nb = graph_[i];
                nb.clear();
                int r = std::min((int)cands.size(), cfg_.max_degree);
                for (int j = 0; j < r; ++j) nb.push_back(cands[j].second);
            }
        }

        entry_ = 0; // simple fixed entry point; could be improved with medoid
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Query
    //
    // queries: row-major float [nq × dim]
    // out_ids [qi*k .. qi*k+k-1]   – neighbour IDs, ascending distance
    // out_dists[qi*k .. qi*k+k-1]  – corresponding distances
    // bw: beam width (0 = use Config::beam_width)
    // ─────────────────────────────────────────────────────────────────────────
    void query(const float* queries, int nq, int k,
               std::vector<id_t>&   out_ids,
               std::vector<dist_t>& out_dists,
               int bw = 0) const {
        if (bw <= 0) bw = cfg_.beam_width;
        bw = std::max(bw, k);
        out_ids  .assign((size_t)nq * k, NO_ID);
        out_dists.assign((size_t)nq * k, INF_D);

#pragma omp parallel for schedule(dynamic, 1)
        for (int qi = 0; qi < nq; ++qi) {
            beam_search(queries + (size_t)qi * cfg_.dim, k, bw,
                        out_ids.data()   + (size_t)qi * k,
                        out_dists.data() + (size_t)qi * k);
        }
    }

    // ── Accessors ──────────────────────────────────────────────────────────
    int  num_points()      const { return n_; }
    int  out_degree(int i) const { return (int)graph_[i].size(); }

    const std::vector<id_t>& neighbors(int i) const { return graph_[i]; }

private:
    Config       cfg_;
    int          n_    = 0;
    const float* data_ = nullptr;
    id_t         entry_= 0;
    LSHSketcher  lsh_;
    std::vector<float>             sketches_; // n × hash_bits  (freed after build)
    std::vector<std::vector<id_t>> graph_;    // n × variable degree

    dist_t dist_fn(const float* a, const float* b) const noexcept {
        return cfg_.use_mips ? neg_dot(a, b, cfg_.dim)
                             : l2_sq(a, b, cfg_.dim);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Greedy beam search  (Algorithm 1 in the paper)
    //
    // Maintains:
    //   best      – max-heap of the L closest nodes seen so far
    //   frontier  – min-heap of unvisited nodes in best
    // ─────────────────────────────────────────────────────────────────────────
    void beam_search(const float* q, int k, int L,
                     id_t*   ids,
                     dist_t* dists) const {

        using P    = std::pair<dist_t, id_t>;
        using MaxQ = std::priority_queue<P>;
        using MinQ = std::priority_queue<P, std::vector<P>, std::greater<P>>;

        // Visited set – vector<bool> is memory-efficient; acceptable for
        // n ≤ ~50 M.  For larger datasets, swap for an unordered_set.
        std::vector<bool> vis(n_, false);

        MaxQ best;     // keeps the L closest nodes
        MinQ frontier; // unvisited candidates, cheapest first

        auto try_push = [&](id_t u) {
            if (vis[u]) return;
            vis[u]    = true;
            dist_t dv = dist_fn(q, data_ + (size_t)u * cfg_.dim);
            if ((int)best.size() < L || dv < best.top().first) {
                best.push({dv, u});
                frontier.push({dv, u});
                if ((int)best.size() > L) best.pop();
            }
        };

        try_push(entry_);

        while (!frontier.empty()) {
            auto [dc, c] = frontier.top();
            frontier.pop();
            // Early exit: all remaining frontier nodes are worse than the
            // current L-th best candidate, so no improvement is possible.
            if ((int)best.size() >= L && dc > best.top().first) break;
            for (id_t nb : graph_[c]) try_push(nb);
        }

        // Extract top-k in ascending order
        std::vector<P> res;
        res.reserve(best.size());
        while (!best.empty()) { res.push_back(best.top()); best.pop(); }
        std::sort(res.begin(), res.end());

        int out_k = std::min((int)res.size(), k);
        for (int i = 0; i < out_k; ++i) {
            ids  [i] = res[i].second;
            dists[i] = res[i].first;
        }
        // Pad remaining slots
        for (int i = out_k; i < k; ++i) { ids[i] = NO_ID; dists[i] = -INF_D; }
    }
};

} // namespace pipnn

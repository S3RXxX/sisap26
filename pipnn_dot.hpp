#pragma once
/**
 * pipnn_dot.hpp  –  PiPNN for normalized dot-product (cosine) similarity.
 *
 * Assumptions:
 *   • All base and query vectors are L2-normalised (unit norm).
 *   • Similarity is dot(a,b) ∈ [-1,1].  Higher = more similar.
 *
 * Two distance spaces are used deliberately:
 *
 *   BUILD distance  bd(a,b) = 1 - dot(a,b) ∈ [0,2]
 *       Always non-negative, so alpha-RNG pruning in RobustPrune is
 *       well-defined.  Using raw -dot (which is negative for similar
 *       vectors) causes alpha*dyz to be more negative than dyz, which
 *       over-prunes and halves avg_degree.
 *
 *   QUERY distance  qd(a,b) = -dot(a,b)
 *       Used only inside beam search (min-heap order).  Gives the same
 *       nearest-neighbour ranking as bd because both are monotone in dot.
 *
 * The graph is built using bd; queries traverse it using qd.
 * query() returns dot-product scores (higher = more similar).
 *
 * Compile:
 *   g++ -O3 -std=c++17 -fopenmp -march=native your_code.cpp
 *
 * Quick start:
 *   pipnn::Config cfg;  cfg.dim = 768;
 *   pipnn::IndexDot idx(cfg);
 *   idx.build(data, n);
 *   std::vector<uint32_t> ids(nq * k);
 *   std::vector<float>    scores(nq * k);   // dot-product scores
 *   idx.query(queries, nq, k, ids, scores);
 */

#include <unordered_set>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <tuple>
#include <vector>
#ifdef _OPENMP
#  include <omp.h>
#endif
#if defined(__AVX2__) && defined(__FMA__)
#  include <immintrin.h>
#  define PIPNN_AVX2 1
#else
#  define PIPNN_AVX2 0
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
    int   dim           = 0;     ///< vector dimension  (REQUIRED)
    int   max_degree    = 64;    ///< R – max out-degree per node
    float alpha         = 1.2f;  ///< α – RobustPrune directional factor
    int   leaf_size     = 256;   ///< Cmax – max points per leaf
    int   min_leaf_size = 32;    ///< Cmin – merge threshold
    float leader_frac   = 0.05f; ///< fraction of leaders per sub-problem
    int   max_leaders   = 1000;  ///< hard cap on leaders
    int   top_fanout    = 10;    ///< RBC fanout at depth 0
    int   second_fanout = 3;     ///< RBC fanout at depth 1
    int   knn_k         = 2;     ///< k for bi-directed kNN in leaves
    int   hash_bits     = 12;    ///< m – LSH bits for HashPrune (≤16)
    int   reservoir_cap = 128;   ///< lmax – HashPrune reservoir size
    int   num_replicas  = 1;     ///< independent RBC replications
    bool  final_prune   = true;  ///< apply RobustPrune after HashPrune
    bool  back_edge     = true;  ///< run back-edge consolidation pass
    int   k_entry       = 12;    ///< diverse graph entry points
    int   entry_sample  = 3000;  ///< sample size for entry-point selection
    int   beam_width    = 128;   ///< default query beam width
    int   num_threads   = 0;     ///< 0 = all available OMP threads
    uint64_t seed       = 42;
};

// ─────────────────────────────────────────────────────────────────────────────
// SIMD kernels
// ─────────────────────────────────────────────────────────────────────────────

/// Raw dot product  dot(a,b) ∈ [-1,1] for unit vectors.
inline float dot_avx(const float* __restrict__ a,
                     const float* __restrict__ b, int d) noexcept {
#if PIPNN_AVX2
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= d; i += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i), acc);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float s = _mm_cvtss_f32(lo);
    for (; i < d; ++i) s += a[i]*b[i];
    return s;
#else
    float s = 0; for (int i = 0; i < d; ++i) s += a[i]*b[i]; return s;
#endif
}

/// Build distance  bd(a,b) = 1 - dot(a,b) ∈ [0,2].
/// Always non-negative → alpha-RNG pruning is well-defined.
inline dist_t build_dist(const float* a, const float* b, int d) noexcept {
    return 1.0f - dot_avx(a, b, d);
}

/// Query distance  qd(a,b) = -dot(a,b) ∈ [-1,1].
/// Used in min-heap beam search: smaller = more similar.
inline dist_t query_dist(const float* a, const float* b, int d) noexcept {
    return -dot_avx(a, b, d);
}

// ─────────────────────────────────────────────────────────────────────────────
// All-pairs build-distance matrix  D[i*nB+j] = 1 - dot(A[i], B[j])
//
// BT is the pre-transposed B matrix: BT[k*nB+j] = B[j*d+k].
// Initialise each row to 1 then subtract the dot-product contribution.
// ─────────────────────────────────────────────────────────────────────────────
static void all_pairs_bd(const float* A, int nA,
                          const float* B, int nB, int d,
                          std::vector<dist_t>& D) {
    static thread_local std::vector<float> tl_BT;
    tl_BT.resize((size_t)d * nB);
    for (int j = 0; j < nB; ++j)
        for (int k = 0; k < d; ++k)
            tl_BT[(size_t)k * nB + j] = B[(size_t)j * d + k];

    D.resize((size_t)nA * nB);
    std::fill(D.begin(), D.end(), 1.0f);          // D[i][j] = 1 - dot

    for (int i = 0; i < nA; ++i) {
        dist_t*      Di  = D.data() + (size_t)i * nB;
        const float* Ai  = A + (size_t)i * d;
        for (int k = 0; k < d; ++k) {
            float        aik = Ai[k];
            const float* BTk = tl_BT.data() + (size_t)k * nB;
            for (int j = 0; j < nB; ++j) Di[j] -= aik * BTk[j];
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RobustPrune  (Vamana-style, uses build distance throughout)
// ─────────────────────────────────────────────────────────────────────────────
static void robust_prune(
        id_t p,
        std::vector<std::pair<dist_t, id_t>>& cands,  // build distances
        const float* data, int d,
        float alpha, int R,
        std::vector<id_t>& out) {

    std::sort(cands.begin(), cands.end());
    out.clear();
    const float* pv = data + (size_t)p * d;

    for (auto& [dp, y] : cands) {
        if (y == NO_ID || y == p) continue;
        if ((int)out.size() >= R) break;

        id_t yid = y;
        y = NO_ID;
        out.push_back(yid);

        const float* yv = data + (size_t)yid * d;
        for (auto& [dz, z] : cands) {
            if (z == NO_ID) continue;
            // build_dist(y,z) = 1 - dot(y,z) ∈ [0,2]: alpha scales correctly
            dist_t dyz = build_dist(yv, data + (size_t)z * d, d);
            if (alpha * dyz < dz) z = NO_ID;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HashPrune reservoir  (metric-agnostic: smaller dist = closer)
// ─────────────────────────────────────────────────────────────────────────────
struct HashReservoir {
    struct Slot { uint16_t hash; id_t id; dist_t dist; };
    int cap = 0, sz = 0;
    std::vector<Slot> buf;

    void init(int c) { cap = c; sz = 0; buf.resize(c); }

    void insert(uint16_t h, id_t id, dist_t d) {
        for (int i = 0; i < sz; ++i) {
            if (buf[i].hash == h) {
                if (d < buf[i].dist) buf[i] = {h, id, d};
                return;
            }
        }
        if (sz < cap) { buf[sz++] = {h, id, d}; return; }
        int fi = 0;
        for (int i = 1; i < sz; ++i) if (buf[i].dist > buf[fi].dist) fi = i;
        if (d < buf[fi].dist) buf[fi] = {h, id, d};
    }

    void flush(std::vector<std::pair<dist_t, id_t>>& out) const {
        out.clear();
        for (int i = 0; i < sz; ++i) out.emplace_back(buf[i].dist, buf[i].id);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// LSH sketcher for HashPrune  (hyperplane-based, metric-agnostic)
// ─────────────────────────────────────────────────────────────────────────────
struct LSHSketcher {
    int m = 0, d = 0;
    std::vector<float> H;  // m×d unit hyperplanes

    void init(int m_, int d_, uint64_t seed) {
        m = m_; d = d_;
        H.resize((size_t)m * d);
        std::mt19937_64 rng(seed);
        std::normal_distribution<float> nd;
        for (auto& v : H) v = nd(rng);
        for (int i = 0; i < m; ++i) {
            float nrm = 0;
            float* hi = H.data() + (size_t)i * d;
            for (int k = 0; k < d; ++k) nrm += hi[k]*hi[k];
            nrm = std::sqrt(nrm) + 1e-9f;
            for (int k = 0; k < d; ++k) hi[k] /= nrm;
        }
    }

    void sketch(const float* v, float* s) const {
        for (int i = 0; i < m; ++i) {
            float acc = 0;
            const float* hi = H.data() + (size_t)i * d;
            for (int k = 0; k < d; ++k) acc += hi[k]*v[k];
            s[i] = acc;
        }
    }

    uint16_t hash_pair(const float* sp, const float* sc) const {
        uint16_t h = 0;
        for (int i = 0; i < m; ++i)
            if (sc[i] >= sp[i]) h |= uint16_t(1u << i);
        return h;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Flat contiguous graph  (cache-friendly beam search)
// Neighbours of node i live at flat[i*R .. i*R+deg[i]-1].
// ─────────────────────────────────────────────────────────────────────────────
struct FlatGraph {
    int n = 0, R = 0;
    std::vector<id_t>    flat;
    std::vector<uint8_t> deg;

    void init(int n_, int R_) {
        n = n_; R = R_;
        flat.assign((size_t)n * R, NO_ID);
        deg .assign(n, 0);
    }
    void set(int i, const std::vector<id_t>& nbrs) {
        int d_ = (int)std::min((int)nbrs.size(), R);
        deg[i] = (uint8_t)d_;
        id_t* row = flat.data() + (size_t)i * R;
        for (int j = 0; j < d_; ++j) row[j] = nbrs[j];
    }
    std::vector<id_t> get(int i) const {
        const id_t* row = flat.data() + (size_t)i * R;
        return {row, row + deg[i]};
    }
    const id_t* row(int i)    const { return flat.data() + (size_t)i * R; }
    int          degree(int i) const { return deg[i]; }
};

// Back-edge consolidation: for every edge u→v propose v→u,
// merge with existing neighbours, re-prune.
static void back_edge_pass(FlatGraph& g, const float* data, int n, int d,
                            float alpha) {
    std::vector<std::vector<std::pair<dist_t,id_t>>> back(n);
    for (int u = 0; u < n; ++u) {
        const float* uv = data + (size_t)u * d;
        const id_t* row = g.row(u);
        for (int j = 0; j < g.degree(u); ++j) {
            id_t v = row[j]; if (v == NO_ID) continue;
            back[v].emplace_back(build_dist(data+(size_t)v*d, uv, d), (id_t)u);
        }
    }
#pragma omp parallel for schedule(dynamic, 128)
    for (int v = 0; v < n; ++v) {
        if (back[v].empty()) continue;
        const float* vv = data + (size_t)v * d;
        std::vector<std::pair<dist_t,id_t>> cands;
        cands.reserve(g.degree(v) + back[v].size());
        const id_t* rw = g.row(v);
        for (int j = 0; j < g.degree(v); ++j) {
            id_t nb = rw[j]; if (nb == NO_ID) continue;
            cands.emplace_back(build_dist(vv, data+(size_t)nb*d, d), nb);
        }
        for (auto& e : back[v]) cands.push_back(e);
        std::sort(cands.begin(), cands.end());
        cands.erase(
            std::unique(cands.begin(), cands.end(),
                        [](const auto& a, const auto& b){return a.second==b.second;}),
            cands.end());
        std::vector<id_t> out;
        robust_prune((id_t)v, cands, data, d, alpha, g.R, out);
        g.set(v, out);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Epoch-based visited set  (no per-query allocation)
// ─────────────────────────────────────────────────────────────────────────────
struct EpochVisited {
    static std::vector<uint32_t>& tl_buf() {
        static thread_local std::vector<uint32_t> b; return b;
    }
    static uint32_t& tl_ep() {
        static thread_local uint32_t e = 0; return e;
    }
    uint32_t ep_ = 0;

    void reset(int n) {
        auto& b = tl_buf();
        if ((int)b.size() < n) b.assign(n, 0);
        auto& e = tl_ep();
        if (++e == 0) { std::fill(b.begin(), b.end(), 0); e = 1; }
        ep_ = e;
    }
    bool test_and_mark(id_t u) {
        auto& b = tl_buf();
        if (b[u] == ep_) return true;
        b[u] = ep_; return false;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Diverse entry points  (greedy k-centres on a random sample)
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<id_t> diverse_entries(
        const float* data, int n, int d,
        int k, int sample_sz, uint64_t seed) {

    sample_sz = std::min(sample_sz, n);
    std::mt19937_64 rng(seed);
    std::vector<id_t> samp(n);
    std::iota(samp.begin(), samp.end(), 0);
    std::shuffle(samp.begin(), samp.end(), rng);
    samp.resize(sample_sz);

    // Seed: point maximising sum of dot products with others (medoid in bd)
    std::vector<float> cost(sample_sz, 0.f);
    for (int i = 0; i < sample_sz; ++i)
        for (int j = 0; j < sample_sz; ++j)
            cost[i] += build_dist(data+(size_t)samp[i]*d,
                                  data+(size_t)samp[j]*d, d);
    int si = (int)(std::min_element(cost.begin(),cost.end())-cost.begin());

    std::vector<id_t> entries;
    entries.push_back(samp[si]);

    std::vector<float> min_d(sample_sz, INF_D);
    for (int e = 1; e < k && e < n; ++e) {
        id_t prev = entries.back();
        float  best = -1; id_t best_id = samp[0];
        for (int i = 0; i < sample_sz; ++i) {
            float dd = build_dist(data+(size_t)samp[i]*d,
                                  data+(size_t)prev*d, d);
            min_d[i] = std::min(min_d[i], dd);
            if (min_d[i] > best) { best = min_d[i]; best_id = samp[i]; }
        }
        entries.push_back(best_id);
    }
    return entries;
}

// ─────────────────────────────────────────────────────────────────────────────
// RBC partitioning with batch GEMM (build distance = 1-dot)
// ─────────────────────────────────────────────────────────────────────────────
using Leaf      = std::vector<id_t>;
using Partition = std::vector<Leaf>;

static void rbc_recurse(
        const float* data, int d,
        const std::vector<id_t>& pts,
        const Config& cfg, int depth,
        std::mt19937_64& rng,
        Partition& out_leaves) {

    int n = (int)pts.size();
    if (n <= cfg.leaf_size) { out_leaves.push_back(pts); return; }

    int nl = std::min((int)(cfg.leader_frac * n), cfg.max_leaders);
    nl = std::max(nl, 2);

    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    perm.resize(nl);

    std::vector<id_t> leaders(nl);
    for (int i = 0; i < nl; ++i) leaders[i] = pts[perm[i]];

    int fanout = 1;
    if      (depth == 0) fanout = cfg.top_fanout;
    else if (depth == 1) fanout = cfg.second_fanout;
    fanout = std::min(fanout, nl);

    // Pre-transpose leaders (once per level).
    // BT[k*nl+j] = leader_j[k]  →  inner dot-product loop over j is contiguous.
    std::vector<float> BT((size_t)d * nl);
    for (int j = 0; j < nl; ++j) {
        const float* lj = data + (size_t)leaders[j] * d;
        for (int k = 0; k < d; ++k)
            BT[(size_t)k * nl + j] = lj[k];
    }

    // p2l[i*fanout .. +fanout-1] = top-fanout leader indices for point i
    std::vector<int> p2l((size_t)n * fanout);

    const int BATCH = 512;
#pragma omp parallel for schedule(dynamic, 4)
    for (int i0 = 0; i0 < n; i0 += BATCH) {
        int ie = std::min(i0 + BATCH, n), bn = ie - i0;

        static thread_local std::vector<float> tl_D;
        static thread_local std::vector<int>   tl_idx;
        tl_D  .resize((size_t)bn * nl);
        tl_idx.resize(nl);

        // D[i][j] = 1 - dot(pi, leader_j)
        // Initialise to 1, then subtract dot product
        std::fill(tl_D.begin(), tl_D.begin() + (size_t)bn*nl, 1.0f);
        for (int i = 0; i < bn; ++i) {
            float*       Di  = tl_D.data() + (size_t)i * nl;
            const float* pi  = data + (size_t)pts[i0+i] * d;
            for (int k = 0; k < d; ++k) {
                float pik        = pi[k];
                const float* BTk = BT.data() + (size_t)k * nl;
                for (int j = 0; j < nl; ++j) Di[j] -= pik * BTk[j];
            }
        }

        // Top-fanout selection using nth_element: O(nl) average vs O(nl log k)
        for (int i = 0; i < bn; ++i) {
            const float* row = tl_D.data() + (size_t)i * nl;
            std::iota(tl_idx.begin(), tl_idx.end(), 0);
            std::nth_element(tl_idx.begin(), tl_idx.begin() + fanout,
                             tl_idx.end(),
                             [&](int a, int b){ return row[a] < row[b]; });
            int* out = p2l.data() + (i0+i) * fanout;
            for (int f = 0; f < fanout; ++f) out[f] = tl_idx[f];
        }
    }

    // Serial bucket fill
    std::vector<std::vector<id_t>> buckets(nl);
    for (int i = 0; i < n; ++i)
        for (int f = 0; f < fanout; ++f)
            buckets[p2l[(size_t)i*fanout + f]].push_back(pts[i]);

    // Distribute orphaned (too-small) buckets round-robin across valid ones
    std::vector<id_t> orphans;
    for (auto& b : buckets) {
        if ((int)b.size() < cfg.min_leaf_size) {
            for (id_t p : b) orphans.push_back(p);
            b.clear();
        }
    }
    if (!orphans.empty()) {
        std::vector<int> valid;
        for (int i = 0; i < nl; ++i) if (!buckets[i].empty()) valid.push_back(i);
        if (!valid.empty())
            for (int oi = 0; oi < (int)orphans.size(); ++oi)
                buckets[valid[oi % valid.size()]].push_back(orphans[oi]);
    }

    for (auto& b : buckets)
        if (!b.empty())
            rbc_recurse(data, d, b, cfg, depth+1, rng, out_leaves);
}

// ─────────────────────────────────────────────────────────────────────────────
// IndexDot  –  PiPNN index for normalised dot-product similarity
// ─────────────────────────────────────────────────────────────────────────────
class IndexDot {
public:
    explicit IndexDot(Config cfg) : cfg_(std::move(cfg)) {
        assert(cfg_.dim > 0 && "Config::dim must be set");
        assert(cfg_.hash_bits <= 16);
    }

    // ── Build ────────────────────────────────────────────────────────────────
    void build(const float* data, int n) {
        data_ = data;  n_ = n;
        const int d = cfg_.dim, m = cfg_.hash_bits, R = cfg_.max_degree;

#ifdef _OPENMP
        if (cfg_.num_threads > 0) omp_set_num_threads(cfg_.num_threads);
#endif

        // LSH sketches (metric-agnostic; used by HashPrune)
        lsh_.init(m, d, cfg_.seed);
        sketches_.resize((size_t)n * m);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i)
            lsh_.sketch(data + (size_t)i*d, sketches_.data() + (size_t)i*m);

        // RBC partitioning + leaf candidate collection
        std::vector<std::vector<std::pair<dist_t,id_t>>> cands(n);
        std::mt19937_64 rng(cfg_.seed);

        for (int rep = 0; rep < cfg_.num_replicas; ++rep) {
            std::vector<id_t> all(n);
            std::iota(all.begin(), all.end(), 0);
            Partition leaves;
            leaves.reserve(n / std::max(cfg_.leaf_size,1) * 2);
            rbc_recurse(data, d, all, cfg_, 0, rng, leaves);

            int nl = (int)leaves.size();
            using E4 = std::tuple<id_t,id_t,dist_t,dist_t>;
            std::vector<std::vector<E4>> leaf_edges(nl);

#pragma omp parallel for schedule(dynamic, 1)
            for (int li = 0; li < nl; ++li) {
                const Leaf& leaf = leaves[li];
                int ln = (int)leaf.size(); if (ln < 2) continue;

                // Pack leaf data contiguously
                std::vector<float> ld((size_t)ln * d);
                for (int i = 0; i < ln; ++i)
                    std::memcpy(ld.data()+(size_t)i*d,
                                data+(size_t)leaf[i]*d, (size_t)d*4);

                // All-pairs build-distance matrix (1-dot)
                std::vector<dist_t> dmat;
                all_pairs_bd(ld.data(), ln, ld.data(), ln, d, dmat);

                // Bi-directed kNN edges within leaf
                int k = std::min(cfg_.knn_k, ln-1);
                auto& elist = leaf_edges[li]; elist.reserve((size_t)ln*k);
                std::vector<int> idx(ln-1);

                for (int i = 0; i < ln; ++i) {
                    const dist_t* row = dmat.data() + (size_t)i*ln;
                    int cnt = 0;
                    for (int j = 0; j < ln; ++j) if (j != i) idx[cnt++] = j;
                    std::nth_element(idx.begin(), idx.begin()+k,
                                     idx.begin()+cnt,
                                     [&](int a,int b){return row[a]<row[b];});
                    for (int ki = 0; ki < k; ++ki) {
                        int j = idx[ki];
                        elist.emplace_back(leaf[i], leaf[j],
                                           row[j], dmat[(size_t)j*ln+i]);
                    }
                }
            }

            for (int li = 0; li < nl; ++li)
                for (auto& [s,t,df,db] : leaf_edges[li]) {
                    cands[s].emplace_back(df, t);
                    cands[t].emplace_back(db, s);
                }
        }

        // HashPrune per point (parallel, race-free)
        std::vector<HashReservoir> res(n);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            res[i].init(cfg_.reservoir_cap);
            const float* si = sketches_.data() + (size_t)i*m;
            for (auto& [dist,j] : cands[i]) {
                if (j == (id_t)i) continue;
                uint16_t h = lsh_.hash_pair(si, sketches_.data()+(size_t)j*m);
                res[i].insert(h, j, dist);
            }
        }
        { decltype(cands)().swap(cands); }
        { std::vector<float>().swap(sketches_); }

        // Build flat graph with optional RobustPrune
        g_.init(n, R);
        if (cfg_.final_prune) {
#pragma omp parallel for schedule(dynamic, 64)
            for (int i = 0; i < n; ++i) {
                std::vector<std::pair<dist_t,id_t>> c; res[i].flush(c);
                std::vector<id_t> out;
                robust_prune(i, c, data, d, cfg_.alpha, R, out);
                g_.set(i, out);
            }
        } else {
#pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) {
                std::vector<std::pair<dist_t,id_t>> c; res[i].flush(c);
                std::sort(c.begin(), c.end());
                std::vector<id_t> out;
                for (int j=0,r=std::min((int)c.size(),R); j<r; j++)
                    out.push_back(c[j].second);
                g_.set(i, out);
            }
        }

        if (cfg_.back_edge)
            back_edge_pass(g_, data, n, d, cfg_.alpha);

        entries_ = diverse_entries(data, n, d,
                                   cfg_.k_entry, cfg_.entry_sample, cfg_.seed);
    }

    // ── Query ────────────────────────────────────────────────────────────────
    // out_ids   [qi*k .. +k-1]: neighbour IDs, descending similarity
    // out_scores[qi*k .. +k-1]: dot-product scores, descending
    void query(const float* queries, int nq, int k,
               std::vector<id_t>&   out_ids,
               std::vector<float>&  out_scores,
               int bw = 0) const {
        if (bw <= 0) bw = cfg_.beam_width;
        bw = std::max(bw, k);
        out_ids   .assign((size_t)nq * k, NO_ID);
        out_scores.assign((size_t)nq * k, -INF_D);
#pragma omp parallel for schedule(dynamic, 1)
        for (int qi = 0; qi < nq; ++qi)
            beam_search(queries + (size_t)qi*cfg_.dim, k, bw,
                        out_ids.data()    + (size_t)qi*k,
                        out_scores.data() + (size_t)qi*k);
    }

    // ── Serialisation ────────────────────────────────────────────────────────
    static constexpr uint32_t MAGIC   = 0x544F4450u; // "PDOT"
    static constexpr uint32_t VERSION = 1u;

    bool save(const std::string& path) const {
        FILE* fp = std::fopen(path.c_str(), "wb");
        if (!fp) return false;
        auto w32 = [&](uint32_t v){ std::fwrite(&v,4,1,fp); };
        auto w16 = [&](uint16_t v){ std::fwrite(&v,2,1,fp); };
        w32(MAGIC); w32(VERSION);
        w32((uint32_t)n_); w32((uint32_t)cfg_.dim); w32((uint32_t)cfg_.max_degree);
        w32((uint32_t)entries_.size());
        for (id_t e : entries_) w32(e);
        for (int i = 0; i < n_; ++i) {
            w16((uint16_t)g_.degree(i));
            std::fwrite(g_.row(i), sizeof(id_t), g_.degree(i), fp);
        }
        std::fclose(fp); return true;
    }

    bool load(const std::string& path) {
        FILE* fp = std::fopen(path.c_str(), "rb");
        if (!fp) return false;
        auto r32 = [&](uint32_t& v){ return std::fread(&v,4,1,fp)==1; };
        auto r16 = [&](uint16_t& v){ return std::fread(&v,2,1,fp)==1; };
        uint32_t mag,ver; r32(mag); r32(ver);
        if (mag != MAGIC || ver != VERSION) { std::fclose(fp); return false; }
        uint32_t n32,dim32,deg32,ne; r32(n32); r32(dim32); r32(deg32); r32(ne);
        n_ = (int)n32; cfg_.dim = (int)dim32; cfg_.max_degree = (int)deg32;
        entries_.resize(ne);
        for (auto& e : entries_) { uint32_t v; r32(v); e=v; }
        g_.init(n_, cfg_.max_degree);
        for (int i = 0; i < n_; ++i) {
            uint16_t deg; r16(deg);
            std::vector<id_t> nbrs(deg);
            if (deg) { auto r=std::fread(nbrs.data(),sizeof(id_t),deg,fp);(void)r; }
            g_.set(i, nbrs);
        }
        std::fclose(fp); return true;
    }

    void set_data(const float* d) { data_ = d; }

    // ── Diagnostics ──────────────────────────────────────────────────────────
    int  num_points()         const { return n_; }
    int  out_degree(int i)    const { return g_.degree(i); }
    const std::vector<id_t>& entry_points() const { return entries_; }

    struct Stats { double avg_deg, frac_bidir; };
    Stats stats() const {
        size_t total = 0, bidir = 0;
        std::vector<std::vector<id_t>> adj(n_);
        for (int i = 0; i < n_; ++i) adj[i] = g_.get(i);
        std::vector<std::unordered_set<id_t>> sets(n_);
        for (int i = 0; i < n_; ++i) for (id_t v : adj[i]) sets[i].insert(v);
        for (int u = 0; u < n_; ++u)
            for (id_t v : adj[u]) { ++total; if (sets[v].count((id_t)u)) ++bidir; }
        return { (double)total/n_, total ? (double)bidir/total : 0.0 };
    }

private:
    Config       cfg_;
    int          n_    = 0;
    const float* data_ = nullptr;
    FlatGraph    g_;
    LSHSketcher  lsh_;
    std::vector<float>  sketches_;
    std::vector<id_t>   entries_;

    void beam_search(const float* q, int k, int L,
                     id_t* ids, float* scores) const {
        using P    = std::pair<dist_t, id_t>;  // (query_dist, id)
        using MaxQ = std::priority_queue<P>;   // max on top = worst in beam
        using MinQ = std::priority_queue<P, std::vector<P>, std::greater<P>>;

        EpochVisited vis; vis.reset(n_);
        MaxQ best; MinQ frontier;

        auto try_push = [&](id_t u) __attribute__((always_inline)) {
            if (vis.test_and_mark(u)) return;
            dist_t dv = query_dist(q, data_ + (size_t)u * cfg_.dim, cfg_.dim);
            if ((int)best.size() < L || dv < best.top().first) {
                best.push({dv,u}); frontier.push({dv,u});
                if ((int)best.size() > L) best.pop();
            }
        };

        // Seed from closest diverse entry point
        { dist_t bd = INF_D; id_t be = entries_[0];
          for (id_t e : entries_) {
              dist_t d = query_dist(q, data_+(size_t)e*cfg_.dim, cfg_.dim);
              if (d < bd) { bd=d; be=e; }
          }
          try_push(be); }

        while (!frontier.empty()) {
            auto [dc,c] = frontier.top(); frontier.pop();
            if ((int)best.size() >= L && dc > best.top().first) break;
            const id_t* nbs = g_.row(c);
            int          deg = g_.degree(c);
            for (int j = 0; j < deg; ++j) {
                if (j+1 < deg)
                    __builtin_prefetch(data_+(size_t)nbs[j+1]*cfg_.dim, 0, 1);
                try_push(nbs[j]);
            }
        }

        // Extract top-k: convert query_dist → dot-product score
        std::vector<P> res;
        res.reserve(best.size());
        while (!best.empty()) { res.push_back(best.top()); best.pop(); }
        std::sort(res.begin(), res.end());  // ascending query_dist = desc similarity
        int ok = std::min((int)res.size(), k);
        for (int i = 0; i < ok;  ++i) { ids[i]=res[i].second; scores[i]=-res[i].first; }
        for (int i = ok; i < k; ++i) { ids[i]=NO_ID;          scores[i]=-INF_D; }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Recall@k utility and default config factory
// ─────────────────────────────────────────────────────────────────────────────
inline float recall_at_k(const std::vector<id_t>& approx,
                          const std::vector<id_t>& exact,
                          int k, int nq) {
    int hits = 0;
    for (int qi = 0; qi < nq; ++qi) {
        const id_t* a = approx.data() + (size_t)qi*k;
        const id_t* e = exact .data() + (size_t)qi*k;
        for (int i = 0; i < k; ++i)
            for (int j = 0; j < k; ++j)
                if (a[i]==e[j]) { ++hits; break; }
    }
    return (float)hits / (float)(nq*k);
}

/// Build a Config with good defaults for the given dataset size.
/// Smaller leaf_size → better cache utilisation, purer clusters, lower latency.
inline Config make_config(int dim, int n) {
    Config cfg;
    cfg.dim = dim;
    if      (n <   100'000) { cfg.leaf_size=128; cfg.min_leaf_size=16;  }
    else if (n <   500'000) { cfg.leaf_size=256; cfg.min_leaf_size=32;  }
    else if (n < 5'000'000) { cfg.leaf_size=512; cfg.min_leaf_size=64;  }
    else                    { cfg.leaf_size=1024; cfg.min_leaf_size=128; }
    cfg.max_degree    = 64;
    cfg.alpha         = 1.2f;
    cfg.leader_frac   = 0.05f;
    cfg.max_leaders   = 1000;
    cfg.top_fanout    = 10;
    cfg.second_fanout = 3;
    cfg.knn_k         = 2;
    cfg.hash_bits     = 12;
    cfg.reservoir_cap = 128;
    cfg.num_replicas  = 1;
    cfg.final_prune   = true;
    cfg.back_edge     = true;
    cfg.k_entry       = std::min(n, 12);
    cfg.entry_sample  = std::min(n, 3000);
    cfg.beam_width    = 128;
    cfg.seed          = 42;
    return cfg;
}

} // namespace pipnn

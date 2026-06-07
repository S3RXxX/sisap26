#pragma once
/**
 * pipnn_opt.hpp  –  Query-path and build-path optimizations
 * Include AFTER pipnn_fast.hpp.
 *
 * Optimizations (on top of pipnn_fast.hpp fixes):
 *
 *  Opt-A  nth_element instead of partial_sort in RBC batch GEMM.
 *         partial_sort is O(N log k); nth_element is O(N) average.
 *         For N=1000, k=10: ~3x faster for the selection step.
 *
 *  Opt-B  Flat contiguous graph array.
 *         vector<vector<id_t>> scatters each node's neighbour list
 *         across the heap → cache miss per node in beam search.
 *         Flat layout packs all neighbours of node i at offset i*R:
 *         accessing 36 neighbours costs 1 cache line instead of 1+36.
 *
 *  Opt-C  AVX2 / FMA distance kernel.
 *         l2_sq inner loop processes 8 floats/cycle; for D=128 this
 *         is 16 FMA instructions instead of 128 scalar FP ops.
 *         Falls back to scalar if AVX2 is not available.
 *
 *  Opt-D  Multiple diverse entry points.
 *         A single medoid entry point performs poorly for queries
 *         in parts of the space far from the medoid.  Storing
 *         k_entry ≥ 8 diverse starting nodes and selecting the
 *         closest for each query improves recall with negligible cost.
 *
 *  Opt-E  Prefetch hints in the beam search inner loop.
 *         __builtin_prefetch on the next neighbour's vector brings
 *         it into L1 before the distance computation.
 */
#include "pipnn_fast.hpp"
#include <cstring>
#include <iostream>

// ── AVX2 availability ────────────────────────────────────────────────────────
#if defined(__AVX2__) && defined(__FMA__)
#  include <immintrin.h>
#  define PIPNN_AVX2 1
#else
#  define PIPNN_AVX2 0
#endif

namespace pipnn {

// ─────────────────────────────────────────────────────────────────────────────
// Opt-C  SIMD distance kernels
// ─────────────────────────────────────────────────────────────────────────────
inline dist_t l2_sq_fast(const float* __restrict__ a,
                          const float* __restrict__ b, int d) noexcept {
#if PIPNN_AVX2
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= d; i += 8) {
        __m256 va   = _mm256_loadu_ps(a + i);
        __m256 vb   = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(diff, diff, acc);
    }
    // Horizontal reduce
    __m128 lo  = _mm256_castps256_ps128(acc);
    __m128 hi  = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float s = _mm_cvtss_f32(lo);
    for (; i < d; ++i) { float t = a[i]-b[i]; s += t*t; }
    return s;
#else
    return l2_sq(a, b, d);
#endif
}

inline dist_t neg_dot_fast(const float* __restrict__ a,
                            const float* __restrict__ b, int d) noexcept {
// #if PIPNN_AVX2
//     __m256 acc = _mm256_setzero_ps();
//     int i = 0;
//     for (; i + 8 <= d; i += 8)
//         acc = _mm256_fmadd_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i), acc);
//     __m128 lo  = _mm256_castps256_ps128(acc);
//     __m128 hi  = _mm256_extractf128_ps(acc, 1);
//     lo = _mm_add_ps(lo, hi);
//     lo = _mm_hadd_ps(lo, lo);
//     lo = _mm_hadd_ps(lo, lo);
//     float s = _mm_cvtss_f32(lo);
//     float a_norm = _mm_cvtss_f32(lo);
//     float b_norm = _mm_cvtss_f32(lo);
//     for (; i < d; ++i) 
//     {
//         s += a[i]*b[i];
//         a_norm += a[i]*a[i];
//         b_norm += b[i]*b[i];
//     }
//     a_norm = std::sqrt(a_norm);
//     b_norm = std::sqrt(b_norm);
//     s = (a_norm > 0.0f && b_norm > 0.0f) ? s / (a_norm * b_norm): 0.0f;
//     return -s;
// #else
    return neg_dot(a, b, d);
// #endif
}

// ─────────────────────────────────────────────────────────────────────────────
// Opt-A  RBC with nth_element (O(N) average vs O(N log k) partial_sort)
// ─────────────────────────────────────────────────────────────────────────────
static void rbc_recurse_opt(
        const float* data, int d,
        const std::vector<id_t>& pts,
        const Config& cfg, int depth,
        std::mt19937_64& rng,
        Partition& out_leaves) {

    int n = (int)pts.size();
    if (n <= cfg.leaf_size) { out_leaves.push_back(pts); return; }

    int nl = std::min((int)(cfg.leader_frac * n), cfg.max_leaders);
    nl     = std::max(nl, 2);

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

    // Pre-transpose + norms for leaders (once per level)
    std::vector<float> BT((size_t)d * nl), nb(nl, 0.f);
    for (int j = 0; j < nl; ++j) {
        const float* lj = data + (size_t)leaders[j] * d;
        for (int k = 0; k < d; ++k) {
            BT[(size_t)k * nl + j]  = lj[k];
            nb[j]                  += lj[k] * lj[k];
        }
    }

    std::vector<int> p2l((size_t)n * fanout);

    const int BATCH = 512;
#pragma omp parallel for schedule(dynamic, 4)
    for (int i0 = 0; i0 < n; i0 += BATCH) {
        int ie = std::min(i0 + BATCH, n), bn = ie - i0;

        static thread_local std::vector<float> tl_na, tl_D;
        static thread_local std::vector<int>   tl_idx;
        tl_na .resize(bn);
        tl_D  .resize((size_t)bn * nl);
        tl_idx.resize(nl);
        for (int i = 0; i < bn; ++i) {
                const float* pi = data + (size_t)pts[i0+i] * d;
                float s = 0;
                for (int k = 0; k < d; ++k) s += pi[k]*pi[k];
                tl_na[i] = s;
            }
        if (cfg.use_mips)
        {
            
            for (int i = 0; i < bn; ++i) {
                float*       Di  = tl_D.data() + (size_t)i * nl;
                const float* pi  = data + (size_t)pts[i0+i] * d;
                for (int k = 0; k < d; ++k) {
                    float pik        = pi[k];
                    const float* BTk = BT.data() + (size_t)k * nl;
                    for (int j = 0; j < nl; ++j) {
                        Di[j] -= pik * BTk[j];
                        // #pragma omp critical
                        // std::cout << "D" << i << j << " == "<<Di[j] << std::endl;
                    }
                }
            }
        }
        else
        {
            for (int i = 0; i < bn; ++i) {
                float* Di = tl_D.data() + (size_t)i * nl;
                for (int j = 0; j < nl; ++j) Di[j] = tl_na[i] + nb[j];
            }
            for (int i = 0; i < bn; ++i) {
                float*       Di  = tl_D.data() + (size_t)i * nl;
                const float* pi  = data + (size_t)pts[i0+i] * d;
                for (int k = 0; k < d; ++k) {
                    float pik        = pi[k];
                    const float* BTk = BT.data() + (size_t)k * nl;
                    for (int j = 0; j < nl; ++j) Di[j] -= 2.f * pik * BTk[j];
                }
            }
        }
        

        // Opt-A: nth_element instead of partial_sort
        for (int i = 0; i < bn; ++i) {
            const float* row = tl_D.data() + (size_t)i * nl;
            std::iota(tl_idx.begin(), tl_idx.end(), 0);
            // nth_element: O(nl) average vs partial_sort O(nl*log fanout)
            std::nth_element(tl_idx.begin(), tl_idx.begin() + fanout,
                             tl_idx.end(),
                             [&](int a, int b){ return row[a] < row[b]; });
            // No need to sort within the top-fanout; bucket assignment is unordered
            int* out = p2l.data() + (i0 + i) * fanout;
            for (int f = 0; f < fanout; ++f) out[f] = tl_idx[f];
        }
    }

    std::vector<std::vector<id_t>> buckets(nl);
    for (int i = 0; i < n; ++i)
        for (int f = 0; f < fanout; ++f)
            buckets[p2l[(size_t)i * fanout + f]].push_back(pts[i]);

    std::vector<id_t> orphans;
    for (auto& b : buckets) {
        if ((int)b.size() < cfg.min_leaf_size) {
            for (id_t p : b) orphans.push_back(p);
            b.clear();
        }
    }
    if (!orphans.empty()) {
        int bi = -1;
        for (int i = 0; i < nl; ++i)
            if (!buckets[i].empty() && (bi<0 || buckets[i].size()>buckets[bi].size()))
                bi = i;
        if (bi >= 0) for (id_t p : orphans) buckets[bi].push_back(p);
    }
    for (auto& b : buckets)
        if (!b.empty())
            rbc_recurse_opt(data, d, b, cfg, depth + 1, rng, out_leaves);
}

// ─────────────────────────────────────────────────────────────────────────────
// Opt-B  Flat graph storage
// Neighbours of node i live at flat[i*R .. i*R+deg[i]-1] — contiguous,
// predictable offsets → one cache-line prefetch per visited node.
// ─────────────────────────────────────────────────────────────────────────────
struct FlatGraph {
    int            n = 0, R = 0;
    std::vector<id_t>    flat;  // n * R  (row-major, NO_ID padding)
    std::vector<uint8_t> deg;   // actual out-degree per node  (≤ 255)

    void init(int n_, int R_) {
        n = n_; R = R_;
        flat.assign((size_t)n * R, NO_ID);
        deg .assign(n, 0);
    }

    void set(int i, const std::vector<id_t>& nbrs) {
        int d = (int)std::min((int)nbrs.size(), R);
        deg[i] = (uint8_t)d;
        id_t* row = flat.data() + (size_t)i * R;
        for (int j = 0; j < d; ++j) row[j] = nbrs[j];
    }

    // Used by consolidate_back_edges_flat below
    std::vector<id_t> get(int i) const {
        const id_t* row = flat.data() + (size_t)i * R;
        return {row, row + deg[i]};
    }

    const id_t* row(int i)  const { return flat.data() + (size_t)i * R; }
    int          degree(int i) const { return deg[i]; }
};

// Back-edge pass adapted for FlatGraph
static void back_edges_flat(FlatGraph& g,
                              const float* data, int n, int d,
                              float alpha, bool use_mips) {
    std::vector<std::vector<std::pair<dist_t,id_t>>> back(n);
    for (int u = 0; u < n; ++u) {
        const float* uv = data + (size_t)u * d;
        const id_t* row = g.row(u);
        for (int j = 0; j < g.degree(u); ++j) {
            id_t v = row[j];
            if (v == NO_ID) continue;
            dist_t dvu = use_mips ? neg_dot_fast(data+(size_t)v*d, uv, d)
                                  : l2_sq_fast (data+(size_t)v*d, uv, d);
            back[v].emplace_back(dvu, (id_t)u);
        }
    }
#pragma omp parallel for schedule(dynamic, 128)
    for (int v = 0; v < n; ++v) {
        if (back[v].empty()) continue;
        std::vector<std::pair<dist_t,id_t>> cands;
        const id_t* rw = g.row(v);
        const float* vv = data + (size_t)v * d;
        for (int j = 0; j < g.degree(v); ++j) {
            id_t nb = rw[j]; if (nb == NO_ID) continue;
            dist_t dv = use_mips ? neg_dot_fast(vv, data+(size_t)nb*d, d)
                                 : l2_sq_fast (vv, data+(size_t)nb*d, d);
            cands.emplace_back(dv, nb);
        }
        for (auto& e : back[v]) cands.push_back(e);
        std::sort(cands.begin(), cands.end());
        cands.erase(
            std::unique(cands.begin(), cands.end(),
                        [](const auto& a, const auto& b){return a.second==b.second;}),
            cands.end());
        std::vector<id_t> out;
        robust_prune((id_t)v, cands, data, d, alpha, g.R, use_mips, out);
        g.set(v, out);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Opt-D  Diverse entry points via greedy k-centers on a random sample
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<id_t> diverse_entries(
        const float* data, int n, int d,
        int k_entry, bool use_mips, int sample_sz, uint64_t seed) {

    sample_sz = std::min(sample_sz, n);
    std::mt19937_64 rng(seed);
    std::vector<id_t> sample(n);
    std::iota(sample.begin(), sample.end(), 0);
    std::shuffle(sample.begin(), sample.end(), rng);
    sample.resize(sample_sz);

    auto df = [&](id_t a, id_t b) -> float {
        return use_mips ? neg_dot_fast(data+(size_t)a*d, data+(size_t)b*d, d)
                        : l2_sq_fast (data+(size_t)a*d, data+(size_t)b*d, d);
    };

    // Seed: point with minimum total distance to sample (medoid estimate)
    std::vector<float> cost(sample_sz, 0.f);
    for (int si = 0; si < sample_sz; ++si)
        for (int sj = 0; sj < sample_sz; ++sj)
            cost[si] += df(sample[si], sample[sj]);
    int best_si = (int)(std::min_element(cost.begin(), cost.end()) - cost.begin());

    std::vector<id_t> entries;
    entries.push_back(sample[best_si]);

    // Greedy: each new entry maximises min-distance to existing entries
    std::vector<float> min_d(sample_sz, INF_D);
    for (int e = 1; e < k_entry && e < n; ++e) {
        id_t prev = entries.back();
        float  best_val = -1;
        id_t   best_id  = sample[0];
        for (int si = 0; si < sample_sz; ++si) {
            float d2prev = df(sample[si], prev);
            min_d[si]    = std::min(min_d[si], d2prev);
            if (min_d[si] > best_val) { best_val = min_d[si]; best_id = sample[si]; }
        }
        entries.push_back(best_id);
    }
    return entries;
}

// ─────────────────────────────────────────────────────────────────────────────
// IndexOpt  –  all four optimisations applied
// ─────────────────────────────────────────────────────────────────────────────
class IndexOpt {
public:
    struct ConfigOpt : Config {
        bool back_edge_pass = true;
        int  k_entry        = 8;    // Opt-D: number of diverse entry points
        int  entry_sample   = 3000; // sample size for entry point selection
        int  num_restarts   = 1;
    };

    explicit IndexOpt(ConfigOpt cfg) : cfg_(std::move(cfg)) {
        assert(cfg_.dim > 0 && cfg_.hash_bits <= 16);
    }

    void build(const float* data, int n) {
        data_ = data; n_ = n;
        const int d = cfg_.dim, m = cfg_.hash_bits;
        const int R = cfg_.max_degree;

#ifdef _OPENMP
        if (cfg_.num_threads > 0) omp_set_num_threads(cfg_.num_threads);
#endif

        // ── LSH sketches ─────────────────────────────────────────────────
        lsh_.init(m, d, cfg_.seed);
        sketches_.resize((size_t)n * m);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i)
            lsh_.compute_sketch(data + (size_t)i * d,
                                sketches_.data() + (size_t)i * m);

        // ── RBC + leaf edges ──────────────────────────────────────────────
        std::vector<std::vector<std::pair<dist_t,id_t>>> cands(n);
        std::mt19937_64 rng(cfg_.seed);

        for (int rep = 0; rep < cfg_.num_replicas; ++rep) {
            std::vector<id_t> all(n);
            std::iota(all.begin(), all.end(), 0);
            Partition leaves;
            // Opt-A: rbc with nth_element
            rbc_recurse_opt(data, d, all, cfg_, 0, rng, leaves);

            int nl = (int)leaves.size();
            using E4 = std::tuple<id_t,id_t,dist_t,dist_t>;
            std::vector<std::vector<E4>> leaf_edges(nl);

#pragma omp parallel for schedule(dynamic, 1)
            for (int li = 0; li < nl; ++li) {
                const Leaf& leaf = leaves[li];
                int ln = (int)leaf.size(); if (ln < 2) continue;
                std::vector<float> ld((size_t)ln * d);
                for (int i = 0; i < ln; ++i)
                    std::memcpy(ld.data()+(size_t)i*d,
                                data+(size_t)leaf[i]*d, (size_t)d*4);
                // Opt-C+thread_local distance matrix
                std::vector<dist_t> dmat;
                all_pairs_tl(ld.data(),ln,ld.data(),ln,d,cfg_.use_mips,dmat);

                int k = std::min(cfg_.knn_k, ln-1);
                auto& elist = leaf_edges[li]; elist.reserve((size_t)ln*k);
                std::vector<int> idx(ln-1);
                for (int i = 0; i < ln; ++i) {
                    const dist_t* row = dmat.data()+(size_t)i*ln;
                    int cnt=0; for(int j=0;j<ln;j++) if(j!=i) idx[cnt++]=j;
                    // Opt-A: nth_element for leaf kNN selection too
                    std::nth_element(idx.begin(), idx.begin()+k,
                                     idx.begin()+cnt,
                                     [&](int a,int b){return row[a]<row[b];});
                    for(int ki=0;ki<k;ki++){
                        int j=idx[ki];
                        elist.emplace_back(leaf[i],leaf[j],
                                           row[j], dmat[(size_t)j*ln+i]);
                    }
                }
            }
            for (int li=0;li<nl;li++)
                for (auto& [s,t,df,db]:leaf_edges[li]){
                    cands[s].emplace_back(df,t);
                    cands[t].emplace_back(db,s);
                }
        }

        // ── HashPrune ────────────────────────────────────────────────────
        std::vector<HashReservoir> res(n);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            res[i].init(cfg_.reservoir_cap);
            const float* si = sketches_.data()+(size_t)i*m;
            for (auto& [dist,j]:cands[i]) {
                if (j==(id_t)i) continue;
                uint16_t h=lsh_.hash_pair(si,sketches_.data()+(size_t)j*m);
                res[i].insert(h,j,dist);
            }
        }
        { decltype(cands)().swap(cands); }
        { std::vector<float>().swap(sketches_); }

        // ── Opt-B: build flat graph ──────────────────────────────────────
        g_.init(n, R);
        if (cfg_.final_prune) {
#pragma omp parallel for schedule(dynamic,64)
            for (int i = 0; i < n; ++i) {
                std::vector<std::pair<dist_t,id_t>> c; res[i].flush(c);
                std::vector<id_t> out;
                robust_prune(i,c,data,d,cfg_.alpha,R,cfg_.use_mips,out);
                g_.set(i, out);
            }
        } else {
#pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) {
                std::vector<std::pair<dist_t,id_t>> c; res[i].flush(c);
                std::sort(c.begin(),c.end());
                std::vector<id_t> out;
                for (int j=0,r=std::min((int)c.size(),R);j<r;j++)
                    out.push_back(c[j].second);
                g_.set(i, out);
            }
        }

        // ── Back-edge pass using flat graph ──────────────────────────────
        if (cfg_.back_edge_pass)
            back_edges_flat(g_, data, n, d, cfg_.alpha, cfg_.use_mips);

        // ── Opt-D: diverse entry points ──────────────────────────────────
        entries_ = diverse_entries(data, n, d,
                                   cfg_.k_entry, cfg_.use_mips,
                                   cfg_.entry_sample, cfg_.seed);
    }

    // ── Query ────────────────────────────────────────────────────────────────
    void query(const float* queries, int nq, int k,
               std::vector<id_t>&   out_ids,
               std::vector<dist_t>& out_dists,
               int bw = 0) const {
        if (bw <= 0) bw = cfg_.beam_width;
        bw = std::max(bw, k);
        out_ids  .assign((size_t)nq * k, NO_ID);
        out_dists.assign((size_t)nq * k, INF_D);
#pragma omp parallel for schedule(dynamic, 1)
        for (int qi = 0; qi < nq; ++qi)
            beam_search(queries+(size_t)qi*cfg_.dim, k, bw,
                        out_ids.data()+(size_t)qi*k,
                        out_dists.data()+(size_t)qi*k);
    }

    bool save(const std::string& path) const {
        // Serialise flat graph as GraphBundle (convert back to vec<vec>)
        GraphBundle gb;
        gb.n=n_; gb.dim=cfg_.dim; gb.max_degree=cfg_.max_degree;
        gb.entry=entries_.empty() ? 0 : entries_[0];
        gb.graph.resize(n_);
        for (int i=0;i<n_;i++) gb.graph[i]=g_.get(i);
        return save_graph(path, gb);
    }
    bool load(const std::string& path) {
        GraphBundle gb; if (!load_graph(path,gb)) return false;
        n_=gb.n; cfg_.dim=gb.dim; cfg_.max_degree=gb.max_degree;
        g_.init(n_, cfg_.max_degree);
        for (int i=0;i<n_;i++) g_.set(i, gb.graph[i]);
        entries_={gb.entry}; return true;
    }
    void set_data(const float* d) { data_=d; }

    // Diagnostics
    int  num_points()      const { return n_; }
    int  out_degree(int i) const { return g_.degree(i); }
    const std::vector<id_t>& entry_points() const { return entries_; }

    GraphStats stats() const {
        // Build temporary vec<vec> for graph_stats
        std::vector<std::vector<id_t>> tmp(n_);
        for (int i=0;i<n_;i++) tmp[i]=g_.get(i);
        return graph_stats(tmp, n_);
    }

private:
    ConfigOpt    cfg_;
    int          n_    = 0;
    const float* data_ = nullptr;
    FlatGraph    g_;
    LSHSketcher  lsh_;
    std::vector<float>  sketches_;
    std::vector<id_t>   entries_;

    dist_t dist_fn(const float* a, const float* b) const noexcept {
        return cfg_.use_mips ? neg_dot_fast(a, b, cfg_.dim)
                             : l2_sq_fast (a, b, cfg_.dim);
    }

    void beam_search(const float* q, int k, int L,
                     id_t* ids, dist_t* dists) const {
        using P    = std::pair<dist_t, id_t>;
        using MaxQ = std::priority_queue<P>;
        using MinQ = std::priority_queue<P, std::vector<P>, std::greater<P>>;

        // Opt-C: epoch visited (no per-query alloc)
        EpochVisited vis; vis.reset(n_);

        MaxQ best;
        MinQ frontier;

        auto try_push = [&](id_t u) __attribute__((always_inline)) {
            if (vis.test_and_mark(u)) return;
            dist_t dv = dist_fn(q, data_ + (size_t)u * cfg_.dim);
            if ((int)best.size() < L || dv < best.top().first) {
                best.push({dv,u}); frontier.push({dv,u});
                if ((int)best.size() > L) best.pop();
            }
        };

        // Opt-D: seed beam from the closest diverse entry point
        {
            float best_d = INF_D; id_t  best_e = entries_[0];
            for (id_t e : entries_) {
                float d = dist_fn(q, data_ + (size_t)e * cfg_.dim);
                if (d < best_d) { best_d = d; best_e = e; }
            }
            try_push(best_e);
        }

        while (!frontier.empty()) {
            auto [dc,c] = frontier.top(); frontier.pop();
            if ((int)best.size() >= L && dc > best.top().first) break;

            // Opt-B+E: flat graph + prefetch next neighbour's data
            const id_t* nbs = g_.row(c);
            int          deg = g_.degree(c);
            for (int j = 0; j < deg; ++j) {
                // Opt-E: prefetch data vector of next neighbour
                if (j + 1 < deg)
                    __builtin_prefetch(
                        data_ + (size_t)nbs[j+1] * cfg_.dim, 0, 1);
                try_push(nbs[j]);
            }
        }

        std::vector<P> res;
        res.reserve(best.size());
        while (!best.empty()) { res.push_back(best.top()); best.pop(); }
        std::sort(res.begin(), res.end());
        int ok = std::min((int)res.size(), k);
        for (int i=0;i<ok;i++) { ids[i]=res[i].second; dists[i]=res[i].first; }
        for (int i=ok;i<k;i++) { ids[i]=NO_ID; dists[i]=INF_D; }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Factory helper: best defaults with all optimisations
// ─────────────────────────────────────────────────────────────────────────────
inline IndexOpt::ConfigOpt opt_config(int dim, int n, bool use_mips = false) {
    IndexOpt::ConfigOpt cfg;
    cfg.dim      = dim;
    cfg.use_mips = use_mips;

    // leaf_size: small leaves are cache-resident and more cluster-pure
    if      (n <   100'000) { cfg.leaf_size=128; cfg.min_leaf_size=16;  }
    else if (n <   500'000) { cfg.leaf_size=256; cfg.min_leaf_size=32;  }
    else if (n < 5'000'000) { cfg.leaf_size=512; cfg.min_leaf_size=64;  }
    else                    { cfg.leaf_size=1024; cfg.min_leaf_size=128; }

    cfg.max_degree    = 64;
    cfg.leader_frac   = 0.05f;
    cfg.max_leaders   = 1000;
    cfg.top_fanout    = 10;
    cfg.second_fanout = 3;
    cfg.knn_k         = 2;
    cfg.hash_bits     = 12;
    cfg.reservoir_cap = 128;
    cfg.alpha         = 1.2f;
    cfg.num_replicas  = 1;
    cfg.final_prune   = true;
    cfg.back_edge_pass= true;
    cfg.k_entry       = std::min(n, 12);  // 12 diverse starting nodes
    cfg.entry_sample  = std::min(n, 3000);
    cfg.beam_width    = 128;
    cfg.num_restarts  = 1;
    cfg.seed          = 42;
    return cfg;
}

} // namespace pipnn

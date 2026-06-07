#pragma once
/**
 * pipnn_fast.hpp  –  Performance fixes for PiPNN
 * Include AFTER pipnn_ext.hpp.
 *
 * Root causes of the slow build (diagnosed via micro-benchmarks):
 *
 *  Bug 1 – rbc_recurse does O(N × NL) SCALAR distance calls, single-threaded.
 *           For N=200K, NL=1000, D=128 this is ~23 seconds on one core.
 *           Fix: pre-transpose leader matrix, compute distances as one batched
 *           GEMM per chunk of points.  5× faster (more with multiple cores).
 *
 *  Bug 2 – leaf_size=1024 causes 5 859 leaves of 1024×1024 distance matrices.
 *           Leaf work is O(leaf_size² × D), so halving leaf_size cuts leaf
 *           GEMM time by ~4×, and smaller matrices fit in L1/L2 → higher
 *           GFLOPS.  Benchmarks show leaf_size=256 is 3.4× faster AND gives
 *           6% better recall because smaller leaves are purer clusters.
 *
 *  Bug 3 – all_pairs_l2_tiled re-allocates BT (D×leaf_size floats) on every
 *           leaf call.  thread_local workspace eliminates this overhead (~10%).
 *
 * Usage:
 *   #include "pipnn_fast.hpp"          // after pipnn_ext.hpp
 *   pipnn::IndexFast idx(cfg);         // same API as IndexV2
 *   idx.build(data, n);
 */
#include "pipnn_ext.hpp"
#include <algorithm>
#include <numeric>
#include <cstring>

namespace pipnn {

// ─────────────────────────────────────────────────────────────────────────────
// Fast all-pairs distance: uses thread_local BT to avoid re-allocation.
// ─────────────────────────────────────────────────────────────────────────────
static void all_pairs_tl(const float* A, int nA,
                          const float* B, int nB,
                          int d, bool use_mips,
                          std::vector<dist_t>& D) {
    // Thread-local transposed-B workspace: resize once per thread, reused
    static thread_local std::vector<float> tl_BT;
    static thread_local std::vector<float> tl_na, tl_nb;

    tl_BT.resize((size_t)d * nB);
    tl_na.assign(nA, 0.f);
    tl_nb.assign(nB, 0.f);

    for (int j = 0; j < nB; ++j)
        for (int k = 0; k < d; ++k)
            tl_BT[(size_t)k * nB + j] = B[(size_t)j * d + k];

    D.resize((size_t)nA * nB);
    std::fill(D.begin(), D.end(), 0.f);
    if (use_mips) {
        for (int i = 0; i < nA; ++i) {
            dist_t* Di      = D.data() + (size_t)i * nB;
            const float* Ai = A + (size_t)i * d;
            for (int k = 0; k < d; ++k) {
                float aik        = Ai[k];
                const float* Bk  = tl_BT.data() + (size_t)k * nB;
                for (int j = 0; j < nB; ++j) Di[j] -= aik * Bk[j];
            }
        }
    } else {
        for (int i = 0; i < nA; ++i)
            for (int k = 0; k < d; ++k)
                tl_na[i] += A[(size_t)i*d+k] * A[(size_t)i*d+k];
        for (int j = 0; j < nB; ++j)
            for (int k = 0; k < d; ++k)
                tl_nb[j] += B[(size_t)j*d+k] * B[(size_t)j*d+k];

        for (int i = 0; i < nA; ++i) {
            dist_t* Di      = D.data() + (size_t)i * nB;
            const float* Ai = A + (size_t)i * d;
            for (int j = 0; j < nB; ++j) Di[j] = tl_na[i] + tl_nb[j];
            for (int k = 0; k < d; ++k) {
                float aik       = Ai[k];
                const float* Bk = tl_BT.data() + (size_t)k * nB;
                for (int j = 0; j < nB; ++j) Di[j] -= 2.f * aik * Bk[j];
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fast RBC: batch GEMM for point-to-leader distances.
//
// Key changes vs rbc_recurse:
//   • Transposes leader matrix ONCE per recursion level.
//   • Processes points in chunks of BATCH (512) with a dense GEMM.
//   • Writes assignments to a flat p2l[] array (race-free parallel fill).
//   • Only the final bucket-fill loop is serial (fast: N×fanout ops).
// ─────────────────────────────────────────────────────────────────────────────
static void rbc_recurse_fast(
        const float* data, int d,
        const std::vector<id_t>& pts,
        const Config& cfg, int depth,
        std::mt19937_64& rng,
        Partition& out_leaves) {

    int n = (int)pts.size();
    if (n <= cfg.leaf_size) { out_leaves.push_back(pts); return; }

    // ── Sample leaders ───────────────────────────────────────────────────
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

    // ── Pre-transpose leaders + compute their norms (done ONCE) ──────────
    // BT[k*nl+j] = leader_j[k]  →  inner loop over j is contiguous
    std::vector<float> BT((size_t)d * nl), nb(nl, 0.f);
    for (int j = 0; j < nl; ++j) {
        const float* lj = data + (size_t)leaders[j] * d;
        for (int k = 0; k < d; ++k) {
            BT[(size_t)k * nl + j]  = lj[k];
            nb[j]                  += lj[k] * lj[k];
        }
    }

    // ── Parallel batch GEMM point-to-leader assignment ───────────────────
    // p2l[i*fanout .. i*fanout+fanout-1] = top-fanout leader indices for pt i
    std::vector<int> p2l((size_t)n * fanout);

    const int BATCH = 512;
#pragma omp parallel for schedule(dynamic, 4)
    for (int i0 = 0; i0 < n; i0 += BATCH) {
        int ie = std::min(i0 + BATCH, n), bn = ie - i0;

        // Thread-local buffers – allocated once per thread, grown as needed
        static thread_local std::vector<float> tl_na, tl_D;
        static thread_local std::vector<int>   tl_idx;
        tl_na .resize(bn);
        tl_D  .resize((size_t)bn * nl);
        tl_idx.resize(nl);

        // Squared norms for this batch of points
        for (int i = 0; i < bn; ++i) {
            const float* pi = data + (size_t)pts[i0+i] * d;
            float s = 0;
            for (int k = 0; k < d; ++k) s += pi[k] * pi[k];
            tl_na[i] = s;
        }

        // Initialise D with norms, then subtract 2·A·BT
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

        // Top-fanout leaders per point (partial sort within thread)
        for (int i = 0; i < bn; ++i) {
            const float* row = tl_D.data() + (size_t)i * nl;
            std::iota(tl_idx.begin(), tl_idx.end(), 0);
            std::partial_sort(tl_idx.begin(), tl_idx.begin() + fanout,
                              tl_idx.end(),
                              [&](int a, int b){ return row[a] < row[b]; });
            int* out = p2l.data() + (i0 + i) * fanout;
            for (int f = 0; f < fanout; ++f) out[f] = tl_idx[f];
        }
    }

    // ── Serial bucket fill (N×fanout ops – very fast) ─────────────────────
    std::vector<std::vector<id_t>> buckets(nl);
    for (int i = 0; i < n; ++i)
        for (int f = 0; f < fanout; ++f)
            buckets[p2l[(size_t)i * fanout + f]].push_back(pts[i]);

    // ── Merge tiny buckets ────────────────────────────────────────────────
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

    // ── Recurse ───────────────────────────────────────────────────────────
    for (auto& b : buckets)
        if (!b.empty())
            rbc_recurse_fast(data, d, b, cfg, depth + 1, rng, out_leaves);
}

// ─────────────────────────────────────────────────────────────────────────────
// IndexFast – same API as IndexV2 but with all three fixes applied.
// Default parameters are tuned for single- and few-core machines.
// ─────────────────────────────────────────────────────────────────────────────
class IndexFast {
public:
    struct Config2 : Config {
        bool back_edge_pass = true;
        int  medoid_sample  = 2000;
        int  num_restarts   = 1;
    };

    explicit IndexFast(Config2 cfg) : cfg_(std::move(cfg)) {
        assert(cfg_.dim > 0 && cfg_.hash_bits <= 16);
    }

    void build(const float* data, int n) {
        data_ = data; n_ = n;
        const int d = cfg_.dim, m = cfg_.hash_bits;

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
        std::vector<std::vector<std::pair<dist_t, id_t>>> cands(n);
        std::mt19937_64 rng(cfg_.seed);

        for (int rep = 0; rep < cfg_.num_replicas; ++rep) {
            std::vector<id_t> all(n);
            std::iota(all.begin(), all.end(), 0);
            Partition leaves;
            leaves.reserve(n / std::max(cfg_.leaf_size, 1) * 2);

            // Fix 1: use batch-GEMM RBC
            rbc_recurse_fast(data, d, all, cfg_, 0, rng, leaves);

            int nl = (int)leaves.size();
            using Edge4 = std::tuple<id_t, id_t, dist_t, dist_t>;
            std::vector<std::vector<Edge4>> leaf_edges(nl);

#pragma omp parallel for schedule(dynamic, 1)
            for (int li = 0; li < nl; ++li) {
                const Leaf& leaf = leaves[li];
                int ln = (int)leaf.size();
                if (ln < 2) continue;

                // Pack leaf data
                std::vector<float> ld((size_t)ln * d);
                for (int i = 0; i < ln; ++i)
                    std::memcpy(ld.data() + (size_t)i * d,
                                data + (size_t)leaf[i] * d,
                                (size_t)d * sizeof(float));

                // Fix 3: thread_local BT workspace inside all_pairs_tl
                std::vector<dist_t> dmat;
                all_pairs_tl(ld.data(), ln, ld.data(), ln, d,
                             cfg_.use_mips, dmat);

                int k = std::min(cfg_.knn_k, ln - 1);
                auto& elist = leaf_edges[li];
                elist.reserve((size_t)ln * k);
                std::vector<int> idx(ln - 1);

                for (int i = 0; i < ln; ++i) {
                    const dist_t* row = dmat.data() + (size_t)i * ln;
                    int cnt = 0;
                    for (int j = 0; j < ln; ++j) if (j != i) idx[cnt++] = j;
                    std::partial_sort(idx.begin(), idx.begin() + k,
                                     idx.begin() + cnt,
                                     [&](int a, int b){ return row[a] < row[b]; });
                    for (int ki = 0; ki < k; ++ki) {
                        int j = idx[ki];
                        elist.emplace_back(leaf[i], leaf[j],
                                           row[j], dmat[(size_t)j*ln+i]);
                    }
                }
            }
            for (int li = 0; li < nl; ++li)
                for (auto& [src,dst,df,db] : leaf_edges[li]) {
                    cands[src].emplace_back(df, dst);
                    cands[dst].emplace_back(db, src);
                }
        }

        // ── HashPrune ────────────────────────────────────────────────────
        std::vector<HashReservoir> res(n);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            res[i].init(cfg_.reservoir_cap);
            const float* si = sketches_.data() + (size_t)i * m;
            for (auto& [dist, j] : cands[i]) {
                if (j == (id_t)i) continue;
                uint16_t h = lsh_.hash_pair(si,
                             sketches_.data() + (size_t)j * m);
                res[i].insert(h, j, dist);
            }
        }
        { decltype(cands)().swap(cands); }
        { std::vector<float>().swap(sketches_); }

        // ── Final graph ───────────────────────────────────────────────────
        graph_.resize(n);
        if (cfg_.final_prune) {
#pragma omp parallel for schedule(dynamic, 64)
            for (int i = 0; i < n; ++i) {
                std::vector<std::pair<dist_t, id_t>> c; res[i].flush(c);
                robust_prune(i, c, data, d, cfg_.alpha, cfg_.max_degree,
                             cfg_.use_mips, graph_[i]);
            }
        } else {
#pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) {
                std::vector<std::pair<dist_t,id_t>> c; res[i].flush(c);
                std::sort(c.begin(),c.end()); graph_[i].clear();
                int r=std::min((int)c.size(),cfg_.max_degree);
                for(int j=0;j<r;j++) graph_[i].push_back(c[j].second);
            }
        }

        // Fix 4: back-edge pass
        if (cfg_.back_edge_pass)
            consolidate_back_edges(graph_, data, n, d,
                                   cfg_.alpha, cfg_.max_degree, cfg_.use_mips);

        entry_ = compute_medoid(data, n, d,
                                cfg_.use_mips, cfg_.medoid_sample, cfg_.seed);
    }

    // ── Query (identical to IndexV2) ──────────────────────────────────────
    void query(const float* queries, int nq, int k,
               std::vector<id_t>&   out_ids,
               std::vector<dist_t>& out_dists,
               int bw = 0, int restarts = 0) const {
        if (bw <= 0) bw = cfg_.beam_width;
        if (restarts <= 0) restarts = cfg_.num_restarts;
        bw = std::max(bw, k);
        out_ids  .assign((size_t)nq * k, NO_ID);
        out_dists.assign((size_t)nq * k, INF_D);
#pragma omp parallel for schedule(dynamic, 1)
        for (int qi = 0; qi < nq; ++qi) {
            const float* q = queries + (size_t)qi * cfg_.dim;
            multi_restart(q, k, bw, restarts,
                          out_ids.data()   + (size_t)qi * k,
                          out_dists.data() + (size_t)qi * k);
        }
    }

    bool save(const std::string& p) const {
        GraphBundle gb{graph_, n_, cfg_.dim, cfg_.max_degree, entry_};
        return save_graph(p, gb);
    }
    bool load(const std::string& p) {
        GraphBundle gb;
        if (!load_graph(p, gb)) return false;
        n_=gb.n; cfg_.dim=gb.dim; cfg_.max_degree=gb.max_degree;
        graph_=std::move(gb.graph); entry_=gb.entry; return true;
    }
    void set_data(const float* d) { data_=d; }

    GraphStats stats()        const { return graph_stats(graph_, n_); }
    int  num_points()         const { return n_; }
    id_t entry_point()        const { return entry_; }
    int  out_degree(int i)    const { return (int)graph_[i].size(); }

private:
    Config2      cfg_;
    int          n_    = 0;
    id_t         entry_= 0;
    const float* data_ = nullptr;
    LSHSketcher  lsh_;
    std::vector<float>             sketches_;
    std::vector<std::vector<id_t>> graph_;

    dist_t dist_fn(const float* a, const float* b) const noexcept {
        return cfg_.use_mips ? neg_dot(a, b, cfg_.dim) : l2_sq(a, b, cfg_.dim);
    }

    std::vector<std::pair<dist_t,id_t>>
    beam_from(const float* q, int L, id_t start) const {
        using P = std::pair<dist_t, id_t>;
        EpochVisited vis; vis.reset(n_);
        std::priority_queue<P> best;
        std::priority_queue<P, std::vector<P>, std::greater<P>> frontier;
        auto try_push = [&](id_t u) {
            if (vis.test_and_mark(u)) return;
            dist_t dv = dist_fn(q, data_ + (size_t)u * cfg_.dim);
            if ((int)best.size() < L || dv < best.top().first) {
                best.push({dv,u}); frontier.push({dv,u});
                if ((int)best.size() > L) best.pop();
            }
        };
        try_push(start);
        while (!frontier.empty()) {
            auto [dc,c] = frontier.top(); frontier.pop();
            if ((int)best.size() >= L && dc > best.top().first) break;
            for (id_t nb : graph_[c]) try_push(nb);
        }
        std::vector<P> res;
        while (!best.empty()) { res.push_back(best.top()); best.pop(); }
        std::sort(res.begin(), res.end());
        return res;
    }

    void multi_restart(const float* q, int k, int bw, int R,
                       id_t* ids, dist_t* dists) const {
        using P = std::pair<dist_t, id_t>;
        std::vector<P> merged = beam_from(q, bw, entry_);
        if (R > 1) {
            uint64_t seed = 0;
            for (int i = 0; i < std::min(cfg_.dim, 8); ++i)
                seed ^= *reinterpret_cast<const uint32_t*>(q+i) * 2654435761ULL;
            std::mt19937_64 rng(seed);
            std::uniform_int_distribution<id_t> uid(0, (id_t)(n_-1));
            for (int r = 1; r < R; ++r) {
                auto res = beam_from(q, bw, uid(rng));
                for (auto& p : res) merged.push_back(p);
            }
            std::sort(merged.begin(), merged.end());
            merged.erase(std::unique(merged.begin(), merged.end(),
                [](const P& a,const P& b){return a.second==b.second;}),
                merged.end());
        }
        int ok = std::min((int)merged.size(), k);
        for (int i=0;i<ok;i++)  { ids[i]=merged[i].second; dists[i]=merged[i].first; }
        for (int i=ok;i<k;i++) { ids[i]=NO_ID; dists[i]=INF_D; }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Recommended defaults for common dataset sizes
// ─────────────────────────────────────────────────────────────────────────────
inline IndexFast::Config2 default_config(int dim, int n,
                                          bool use_mips = false) {
    IndexFast::Config2 cfg;
    cfg.dim      = dim;
    cfg.use_mips = use_mips;

    // Fix 2: leaf_size tuned by dataset size.
    // Smaller leaves: better cache efficiency AND better recall per benchmark.
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
    cfg.medoid_sample = std::min(n, 2000);
    cfg.beam_width    = 128;
    cfg.num_restarts  = 1;
    cfg.seed          = 42;
    return cfg;
}

} // namespace pipnn

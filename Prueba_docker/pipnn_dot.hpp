#pragma once
/**
 * pipnn_dot.hpp  –  PiPNN for normalised dot-product similarity.
 *
 * Data storage: float32 (default) OR float16 (uint16_t).
 * Pass float16 via IndexDot::build_f16().  Internally all GEMM and
 * distance kernels operate in float32; float16->float32 conversion
 * happens in small per-batch / per-node scratch buffers so the full
 * dataset never needs to be in RAM as float32.
 *
 * Build distance : bd(a,b) = 1 - dot(a,b) in [0,2]
 * Query distance : qd(a,b) = -dot(a,b)
 * query() returns dot-product similarity scores (higher = more similar).
 */
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
#include <unordered_set>
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

using id_t   = uint32_t;
using dist_t = float;
static constexpr id_t   NO_ID = id_t(-1);
static constexpr dist_t INF_D = std::numeric_limits<dist_t>::infinity();

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────
struct Config {
    int      dim           = 0;
    int      max_degree    = 64;
    float    alpha         = 1.2f;
    int      leaf_size     = 256;
    int      min_leaf_size = 32;
    float    leader_frac   = 0.05f;
    int      max_leaders   = 1000;
    int      top_fanout    = 10;
    int      second_fanout = 3;
    int      knn_k         = 2;
    int      hash_bits     = 12;
    int      reservoir_cap = 128;
    int      num_replicas  = 1;
    bool     final_prune   = true;
    bool     back_edge     = true;
    bool  randomness       = false;
    bool  coocked          = false;
    int      k_entry       = 12;
    int      entry_sample  = 3000;
    int      beam_width    = 128;
    int      num_threads   = 0;
    uint64_t seed          = 42;
};

// ─────────────────────────────────────────────────────────────────────────────
// Float16 <-> Float32 conversion
// ─────────────────────────────────────────────────────────────────────────────
inline uint16_t f32_to_f16(float f) noexcept {
    uint32_t x; std::memcpy(&x, &f, 4);
    uint16_t sign = (uint16_t)((x >> 16) & 0x8000u);
    int32_t  exp  = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (x >> 13) & 0x3ff;
    if (exp <= 0)  return sign;                 // underflow -> 0
    if (exp >= 31) return sign | 0x7c00u;        // overflow -> inf
    return (uint16_t)(sign | (exp << 10) | mant);
}

inline float f16_to_f32_scalar(uint16_t h) noexcept {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = (h & 0x3ff) << 13;
    uint32_t f;
    if (exp == 0)       f = sign | (mant ? ((113u<<23)|mant) : 0);
    else if (exp == 31) f = sign | 0x7f800000u | mant;
    else                f = sign | ((exp+112u)<<23) | mant;
    float out; std::memcpy(&out,&f,4); return out;
}

inline void f16_to_f32(const uint16_t* __restrict__ s,
                        float*          __restrict__ d, int n) noexcept {
#if PIPNN_AVX2 && defined(__F16C__)
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(d + i,
            _mm256_cvtph_ps(_mm_loadu_si128(
                reinterpret_cast<const __m128i*>(s + i))));
    for (; i < n; ++i) d[i] = f16_to_f32_scalar(s[i]);
#else
    for (int i = 0; i < n; ++i) d[i] = f16_to_f32_scalar(s[i]);
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// DataView – thin wrapper over float32 or float16 data.
// ─────────────────────────────────────────────────────────────────────────────
struct DataView {
    const float*    f32 = nullptr;
    const uint16_t* f16 = nullptr;
    int dim = 0;

    DataView() = default;
    DataView(const float*    p, int d) : f32(p), dim(d) {}
    DataView(const uint16_t* p, int d) : f16(p), dim(d) {}

    void pack(const id_t* rows, int n, float* dst) const {
        if (f32) {
            for (int i = 0; i < n; ++i)
                std::memcpy(dst + (size_t)i*dim, f32 + (size_t)rows[i]*dim, dim*4);
        } else {
            for (int i = 0; i < n; ++i)
                f16_to_f32(f16 + (size_t)rows[i]*dim, dst + (size_t)i*dim, dim);
        }
    }
    const float* row(id_t i, float* buf) const {
        if (f32) return f32 + (size_t)i*dim;
        f16_to_f32(f16 + (size_t)i*dim, buf, dim);
        return buf;
    }
    const float* row_a(id_t i) const {
        static thread_local std::vector<float> b;
        if ((int)b.size() < dim) b.resize(dim);
        return row(i, b.data());
    }
    const float* row_b(id_t i) const {
        static thread_local std::vector<float> b;
        if ((int)b.size() < dim) b.resize(dim);
        return row(i, b.data());
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// SIMD distance kernels
// ─────────────────────────────────────────────────────────────────────────────
inline float dot_avx(const float* __restrict__ a,
                     const float* __restrict__ b, int d) noexcept {
#if PIPNN_AVX2
    __m256 acc = _mm256_setzero_ps(); int i = 0;
    for (; i + 8 <= d; i += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i), acc);
    __m128 lo = _mm256_castps256_ps128(acc), hi = _mm256_extractf128_ps(acc,1);
    lo = _mm_add_ps(lo, hi); lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo);
    float s = _mm_cvtss_f32(lo);
    for (; i < d; ++i) s += a[i]*b[i];
    return s;
#else
    float s = 0; for (int i = 0; i < d; ++i) s += a[i]*b[i]; return s;
#endif
}
inline dist_t build_dist(const float* a, const float* b, int d) noexcept {
    return 1.0f - dot_avx(a, b, d);
}
inline dist_t query_dist(const float* a, const float* b, int d) noexcept {
    return -dot_avx(a, b, d);
}

// ─────────────────────────────────────────────────────────────────────────────
// All-pairs build-distance  D[i*nB+j] = 1 - dot(A[i], B[j])
// ─────────────────────────────────────────────────────────────────────────────
static void all_pairs_bd(const float* A, int nA,
                          const float* B, int nB, int d,
                          std::vector<dist_t>& D) {
    static thread_local std::vector<float> tl_BT;
    tl_BT.resize((size_t)d * nB);
    for (int j = 0; j < nB; ++j)
        for (int k = 0; k < d; ++k)
            tl_BT[(size_t)k*nB+j] = B[(size_t)j*d+k];

    D.resize((size_t)nA * nB);
    std::fill(D.begin(), D.end(), 1.0f);
    for (int i = 0; i < nA; ++i) {
        dist_t* Di = D.data() + (size_t)i*nB;
        const float* Ai = A + (size_t)i*d;
        for (int k = 0; k < d; ++k) {
            float aik = Ai[k];
            const float* BTk = tl_BT.data() + (size_t)k*nB;
            for (int j = 0; j < nB; ++j) Di[j] -= aik * BTk[j];
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RobustPrune
// ─────────────────────────────────────────────────────────────────────────────
static void robust_prune(id_t p,
        std::vector<std::pair<dist_t,id_t>>& cands,
        const DataView& dv, float alpha, int R,
        std::vector<id_t>& out) {
    static thread_local std::vector<float> tl_y, tl_z;
    if ((int)tl_y.size() < dv.dim) { tl_y.resize(dv.dim); tl_z.resize(dv.dim); }

    std::sort(cands.begin(), cands.end());
    out.clear();

    for (auto& [dp, y] : cands) {
        if (y == NO_ID || y == p) continue;
        if ((int)out.size() >= R) break;
        id_t yid = y; y = NO_ID;
        out.push_back(yid);
        const float* yv = dv.row(yid, tl_y.data());
        for (auto& [dz, z] : cands) {
            if (z == NO_ID) continue;
            const float* zv = dv.row(z, tl_z.data());
            if (alpha * build_dist(yv, zv, dv.dim) < dz) z = NO_ID;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HashPrune reservoir
// ─────────────────────────────────────────────────────────────────────────────
struct HashReservoir {
    struct Slot { uint16_t hash; id_t id; dist_t dist; };
    int cap = 0, sz = 0;
    std::vector<Slot> buf;
    void init(int c) { cap = c; sz = 0; buf.resize(c); }
    void insert(uint16_t h, id_t id, dist_t d) {
        for (int i = 0; i < sz; ++i)
            if (buf[i].hash == h) { if (d < buf[i].dist) buf[i]={h,id,d}; return; }
        if (sz < cap) { buf[sz++] = {h,id,d}; return; }
        int fi = 0;
        for (int i = 1; i < sz; ++i) if (buf[i].dist > buf[fi].dist) fi = i;
        if (d < buf[fi].dist) buf[fi] = {h,id,d};
    }
    void flush(std::vector<std::pair<dist_t,id_t>>& o) const {
        o.clear(); for (int i = 0; i < sz; ++i) o.emplace_back(buf[i].dist, buf[i].id);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// LSH sketcher
// ─────────────────────────────────────────────────────────────────────────────
struct LSHSketcher {
    int m=0, d=0; std::vector<float> H;
    void init(int m_, int d_, uint64_t seed) {
        m=m_; d=d_; H.resize((size_t)m*d);
        std::mt19937_64 rng(seed); std::normal_distribution<float> nd;
        for (auto& v:H) v=nd(rng);
        for (int i=0;i<m;++i){
            float n=0; float*hi=H.data()+(size_t)i*d;
            for(int k=0;k<d;++k)n+=hi[k]*hi[k]; n=std::sqrt(n)+1e-9f;
            for(int k=0;k<d;++k)hi[k]/=n;
        }
    }
    void sketch(const float* v, float* s) const {
        for (int i=0;i<m;++i){
            float a=0; const float*hi=H.data()+(size_t)i*d;
            for(int k=0;k<d;++k)a+=hi[k]*v[k]; s[i]=a;
        }
    }
    uint16_t hash_pair(const float* sp, const float* sc) const {
        uint16_t h=0;
        for(int i=0;i<m;++i) if(sc[i]>=sp[i]) h|=uint16_t(1u<<i);
        return h;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Flat contiguous graph
// ─────────────────────────────────────────────────────────────────────────────
struct FlatGraph {
    int n=0, R=0;
    std::vector<id_t>    flat;
    std::vector<uint8_t> deg;
    void init(int n_, int R_) {
        n=n_; R=R_;
        flat.assign((size_t)n*R, NO_ID); deg.assign(n,0);
    }
    void set(int i, const std::vector<id_t>& nb) {
        int d_=(int)std::min((int)nb.size(),R); deg[i]=(uint8_t)d_;
        id_t* row=flat.data()+(size_t)i*R;
        for(int j=0;j<d_;++j) row[j]=nb[j];
    }
    std::vector<id_t> get(int i) const {
        const id_t* r=flat.data()+(size_t)i*R; return{r,r+deg[i]};
    }
    const id_t* row(int i)    const { return flat.data()+(size_t)i*R; }
    int         degree(int i) const { return deg[i]; }
};

// ─────────────────────────────────────────────────────────────────────────────
// Back-edge consolidation
// ─────────────────────────────────────────────────────────────────────────────
static void back_edge_pass(FlatGraph& g, const DataView& dv,
                            int n, float alpha) {
    std::vector<std::vector<std::pair<dist_t,id_t>>> back(n);
    for (int u=0;u<n;++u) {
        const float* uv = dv.row_a(u);
        const id_t* row = g.row(u);
        for (int j=0;j<g.degree(u);++j) {
            id_t v=row[j]; if(v==NO_ID) continue;
            back[v].emplace_back(build_dist(dv.row_b(v), uv, dv.dim), (id_t)u);
        }
    }
#pragma omp parallel for schedule(dynamic,128)
    for (int v=0;v<n;++v) {
        if (back[v].empty()) continue;
        const float* vv = dv.row_a(v);
        std::vector<std::pair<dist_t,id_t>> cands;
        cands.reserve(g.degree(v)+back[v].size());
        const id_t* rw=g.row(v);
        for (int j=0;j<g.degree(v);++j) {
            id_t nb=rw[j]; if(nb==NO_ID) continue;
            cands.emplace_back(build_dist(vv, dv.row_b(nb), dv.dim), nb);
        }
        for (auto& e:back[v]) cands.push_back(e);
        std::sort(cands.begin(),cands.end());
        cands.erase(std::unique(cands.begin(),cands.end(),
            [](const auto& a,const auto& b){return a.second==b.second;}),
            cands.end());
        std::vector<id_t> out;
        robust_prune((id_t)v, cands, dv, alpha, g.R, out);
        g.set(v, out);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Epoch visited
// ─────────────────────────────────────────────────────────────────────────────
struct EpochVisited {
    static std::vector<uint32_t>& tl_buf(){static thread_local std::vector<uint32_t> b;return b;}
    static uint32_t& tl_ep(){static thread_local uint32_t e=0;return e;}
    uint32_t ep_=0;
    void reset(int n){
        auto&b=tl_buf(); if((int)b.size()<n) b.assign(n,0);
        auto&e=tl_ep(); if(++e==0){std::fill(b.begin(),b.end(),0);e=1;} ep_=e;
    }
    bool test_and_mark(id_t u){
        auto&b=tl_buf(); if(b[u]==ep_)return true; b[u]=ep_;return false;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Diverse entry points
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<id_t> diverse_entries(const DataView& dv, int n,
        int k, int sample_sz, uint64_t seed) {
    const int d = dv.dim;
    sample_sz = std::min(sample_sz, n);
    std::mt19937_64 rng(seed);
    std::vector<id_t> samp(n); std::iota(samp.begin(),samp.end(),0);
    std::shuffle(samp.begin(),samp.end(),rng); samp.resize(sample_sz);

    std::vector<float> sf32((size_t)sample_sz * d);
    for (int i=0;i<sample_sz;++i)
        dv.pack(&samp[i], 1, sf32.data()+(size_t)i*d);

    auto bd=[&](int i,int j){
        return build_dist(sf32.data()+(size_t)i*d, sf32.data()+(size_t)j*d, d);
    };

    std::vector<float> cost(sample_sz,0.f);
    for(int i=0;i<sample_sz;++i) for(int j=0;j<sample_sz;++j) cost[i]+=bd(i,j);
    int si=(int)(std::min_element(cost.begin(),cost.end())-cost.begin());

    std::vector<id_t> entries; entries.push_back(samp[si]);
    std::vector<float> min_d(sample_sz,INF_D);
    int prev_idx = si;
    for(int e=1;e<k&&e<n;++e){
        float best=-1; int bidx=0;
        for(int i=0;i<sample_sz;++i){
            min_d[i]=std::min(min_d[i], bd(i,prev_idx));
            if(min_d[i]>best){best=min_d[i];bidx=i;}
        }
        entries.push_back(samp[bidx]);
        prev_idx = bidx;
    }
    return entries;
}

// ─────────────────────────────────────────────────────────────────────────────
// RBC partitioning – batch GEMM over DataView
// ─────────────────────────────────────────────────────────────────────────────
using Leaf = std::vector<id_t>; using Partition = std::vector<Leaf>;

static int rbc_recurse(const DataView& dv,
        const std::vector<id_t>& pts, const Config& cfg,
        int depth, std::mt19937_64& rng, Partition& out) {
    int n=(int)pts.size(), d=dv.dim;
    if(n<=cfg.leaf_size){out.push_back(pts);return depth-1;}
    
    int nl=std::min((int)(cfg.leader_frac*n),cfg.max_leaders); nl=std::max(nl,2);
    std::vector<int> perm(n); std::iota(perm.begin(),perm.end(),0);
    std::shuffle(perm.begin(),perm.end(),rng); perm.resize(nl);
    std::vector<id_t> leaders(nl);
    for(int i=0;i<nl;++i) leaders[i]=pts[perm[i]];

    int fanout=1;
    if(depth==0) fanout=cfg.top_fanout;
    else if(depth==1) fanout=cfg.second_fanout;
    fanout=std::min(fanout,nl);

    std::vector<float> BT((size_t)d*nl);
    {
        std::vector<float> lbuf((size_t)nl*d);
        for(int j=0;j<nl;++j) dv.pack(&leaders[j], 1, lbuf.data()+(size_t)j*d);
        for(int j=0;j<nl;++j) for(int k=0;k<d;++k) BT[(size_t)k*nl+j]=lbuf[(size_t)j*d+k];
    }

    std::vector<int> p2l((size_t)n*fanout);
    const int BATCH=512;
#pragma omp parallel for schedule(dynamic,4)
    for(int i0=0;i0<n;i0+=BATCH){
        int ie=std::min(i0+BATCH,n),bn=ie-i0;
        static thread_local std::vector<float> tl_pts, tl_D;
        static thread_local std::vector<int>   tl_idx;
        tl_pts.resize((size_t)bn*d);
        tl_D  .resize((size_t)bn*nl);
        tl_idx.resize(nl);

        for(int i=0;i<bn;++i) dv.pack(&pts[i0+i], 1, tl_pts.data()+(size_t)i*d);

        std::fill(tl_D.begin(), tl_D.begin()+(size_t)bn*nl, 1.0f);
        for(int i=0;i<bn;++i){
            float*Di=tl_D.data()+(size_t)i*nl;
            const float*pi=tl_pts.data()+(size_t)i*d;
            for(int k=0;k<d;++k){
                float pik=pi[k]; const float*BTk=BT.data()+(size_t)k*nl;
                for(int j=0;j<nl;++j) Di[j]-=pik*BTk[j];
            }
        }
        for(int i=0;i<bn;++i){
            const float*row=tl_D.data()+(size_t)i*nl;
            std::iota(tl_idx.begin(),tl_idx.end(),0);
            std::nth_element(tl_idx.begin(),tl_idx.begin()+fanout,tl_idx.end(),
                [&](int a,int b){return row[a]<row[b];});
            int*out2=p2l.data()+(i0+i)*fanout;
            for(int f=0;f<fanout;++f) out2[f]=tl_idx[f];
        }
    }

    std::vector<std::vector<id_t>> buckets(nl);
    for(int i=0;i<n;++i)
        for(int f=0;f<fanout;++f) buckets[p2l[(size_t)i*fanout+f]].push_back(pts[i]);

    std::vector<id_t> orphans;
    for(auto&b:buckets){
        if((int)b.size()<cfg.min_leaf_size){for(id_t p:b)orphans.push_back(p);b.clear();}
    }
    if (cfg.coocked)
    {
        if (!orphans.empty()) {
            int bi = -1;
            for (int i = 0; i < nl; ++i)
                if (!buckets[i].empty() && (bi < 0 || buckets[i].size() > buckets[bi].size()))
                    bi = i;
            if (bi >= 0) for (id_t p : orphans) buckets[bi].push_back(p);
            // (if all buckets empty this depth, orphans are just dropped; shouldn't happen)
        }
    }
    else if (!cfg.randomness)
    {

        if (!orphans.empty()) {
            std::vector<int> valid;
            for (int i = 0; i < nl; ++i) if (!buckets[i].empty()) valid.push_back(i);
            if (!valid.empty())
                for (int oi = 0; oi < (int)orphans.size(); ++oi)
                    buckets[valid[oi % valid.size()]].push_back(orphans[oi]);
        }
    }
    else
    {
        if (!orphans.empty()) {
            // Build weight vector: weight[i] = 1/size(bucket_i) so smaller buckets
            // are preferred — inverse-proportional to size.
            std::vector<double> weights(nl);
            for (int i = 0; i < nl; ++i)
                weights[i] = buckets[i].empty() ? 0.0 : 1.0 / buckets[i].size();

            // Guard: if every bucket was orphaned (degenerate case), spread evenly.
            double total = 0.0;
            for (double w : weights) total += w;
            if (total == 0.0)
                std::fill(weights.begin(), weights.end(), 1.0);

            std::discrete_distribution<int> pick(weights.begin(), weights.end());
            for (id_t p : orphans)
                buckets[pick(rng)].push_back(p);
        }
    }
    int max_depth = 0;
    int children_depth;
    for(auto&b:buckets)
    {
         if(!b.empty()) 
         {
            children_depth=rbc_recurse(dv,b,cfg,depth+1,rng,out);
            if (children_depth > max_depth)
                max_depth = children_depth;
         }
    }
    return max_depth;
}

// ─────────────────────────────────────────────────────────────────────────────
// IndexDot
// ─────────────────────────────────────────────────────────────────────────────
class IndexDot {
public:
    explicit IndexDot(Config cfg) : cfg_(std::move(cfg)) {
        assert(cfg_.dim>0 && cfg_.hash_bits<=16);
    }

    void build(const float* data, int n) {
        build_impl(DataView(data, cfg_.dim), n);
        data_   = data;
        data16_ = nullptr;
    }
    void build_f16(const uint16_t* data, int n) {
        build_impl(DataView(data, cfg_.dim), n);
        data_   = nullptr;
        data16_ = data;
    }

    void query(const float* queries, int nq, int k,
               std::vector<id_t>& out_ids, std::vector<float>& out_scores,
               int bw=0) const {
        if(bw<=0) bw=cfg_.beam_width; bw=std::max(bw,k);
        out_ids   .assign((size_t)nq*k, NO_ID);
        out_scores.assign((size_t)nq*k, -INF_D);
#pragma omp parallel for schedule(dynamic,1)
        for(int qi=0;qi<nq;++qi)
            beam_search(queries+(size_t)qi*cfg_.dim, k, bw,
                        out_ids.data()+(size_t)qi*k,
                        out_scores.data()+(size_t)qi*k);
    }

    static constexpr uint32_t MAGIC=0x544F4450u, VER=1u;
    bool save(const std::string& p) const {
        FILE* fp=std::fopen(p.c_str(),"wb"); if(!fp)return false;
        auto w32=[&](uint32_t v){std::fwrite(&v,4,1,fp);};
        auto w16=[&](uint16_t v){std::fwrite(&v,2,1,fp);};
        w32(MAGIC);w32(VER);
        w32(n_);w32(cfg_.dim);w32(cfg_.max_degree);
        w32((uint32_t)entries_.size());
        for(id_t e:entries_) w32(e);
        for(int i=0;i<n_;++i){w16((uint16_t)g_.degree(i));
            std::fwrite(g_.row(i),4,g_.degree(i),fp);}
        std::fclose(fp);return true;
    }
    bool load(const std::string& p){
        FILE*fp=std::fopen(p.c_str(),"rb");if(!fp)return false;
        auto r32=[&](uint32_t&v){return std::fread(&v,4,1,fp)==1;};
        auto r16=[&](uint16_t&v){return std::fread(&v,2,1,fp)==1;};
        uint32_t mag,ver; r32(mag);r32(ver);
        if(mag!=MAGIC||ver!=VER){std::fclose(fp);return false;}
        uint32_t n32,d32,R32,ne; r32(n32);r32(d32);r32(R32);r32(ne);
        n_=n32;cfg_.dim=d32;cfg_.max_degree=R32;
        entries_.resize(ne);
        for(auto&e:entries_){uint32_t v;r32(v);e=v;}
        g_.init(n_,cfg_.max_degree);
        for(int i=0;i<n_;++i){
            uint16_t dg; r16(dg);
            std::vector<id_t> nb(dg);
            if(dg){auto r=std::fread(nb.data(),4,dg,fp);(void)r;}
            g_.set(i,nb);
        }
        std::fclose(fp);return true;
    }
    void set_data    (const float*    d){data_=d;   data16_=nullptr;}
    void set_data_f16(const uint16_t* d){data16_=d; data_=nullptr;}

    int num_points() const {return n_;}
    const std::vector<id_t>& entry_points() const {return entries_;}
    struct Stats{double avg_deg,frac_bidir;};
    Stats stats() const {
        size_t tot=0,bi=0;
        std::vector<std::unordered_set<id_t>> sets(n_);
        for(int i=0;i<n_;++i) for(id_t v:g_.get(i)) sets[i].insert(v);
        for(int u=0;u<n_;++u)
            for(id_t v:g_.get(u)){++tot;if(sets[v].count(u))++bi;}
        return{(double)tot/n_,tot?(double)bi/tot:0.};
    }

private:
    Config            cfg_;
    int               n_     = 0;
    const float*      data_  = nullptr;
    const uint16_t*   data16_= nullptr;
    FlatGraph         g_;
    LSHSketcher       lsh_;
    std::vector<float>   sketches_;
    std::vector<id_t>    entries_;

    DataView active_dv() const {
        return data16_ ? DataView(data16_, cfg_.dim)
                       : DataView(data_,   cfg_.dim);
    }

    void build_impl(const DataView& dv, int n) {
        n_=n; const int d=cfg_.dim, m=cfg_.hash_bits, R=cfg_.max_degree;
#ifdef _OPENMP
        if(cfg_.num_threads>0) omp_set_num_threads(cfg_.num_threads);
#endif
        lsh_.init(m,d,cfg_.seed);
        sketches_.resize((size_t)n*m);
#pragma omp parallel for schedule(static)
        for(int i=0;i<n;++i){
            static thread_local std::vector<float> buf;
            if((int)buf.size()<d) buf.resize(d);
            lsh_.sketch(dv.row(i,buf.data()), sketches_.data()+(size_t)i*m);
        }

        std::vector<std::vector<std::pair<dist_t,id_t>>> cands(n);
        std::mt19937_64 rng(cfg_.seed);

        for(int rep=0;rep<cfg_.num_replicas;++rep){
            std::vector<id_t> all(n); std::iota(all.begin(),all.end(),0);
            Partition leaves; leaves.reserve(n/std::max(cfg_.leaf_size,1)*2);
            int max_depth =rbc_recurse(dv, all, cfg_, 0, rng, leaves);
            printf("max depth: %i\n", max_depth);
            int nl=(int)leaves.size();

            using E4=std::tuple<id_t,id_t,dist_t,dist_t>;
            std::vector<std::vector<E4>> leaf_edges(nl);

#pragma omp parallel for schedule(dynamic,1)
            for(int li=0;li<nl;++li){
                const Leaf&leaf=leaves[li]; int ln=(int)leaf.size(); if(ln<2)continue;
                static thread_local std::vector<float> ld;
                ld.resize((size_t)ln*d);
                for(int i=0;i<ln;++i) dv.pack(&leaf[i], 1, ld.data()+(size_t)i*d);

                std::vector<dist_t> dmat;
                all_pairs_bd(ld.data(),ln,ld.data(),ln,d,dmat);
                int k2=std::min(cfg_.knn_k,ln-1);
                auto&elist=leaf_edges[li]; elist.reserve((size_t)ln*k2);
                std::vector<int> idx(ln-1);
                for(int i=0;i<ln;++i){
                    const dist_t*row=dmat.data()+(size_t)i*ln;
                    int cnt=0; for(int j=0;j<ln;++j) if(j!=i) idx[cnt++]=j;
                    std::nth_element(idx.begin(),idx.begin()+k2,idx.begin()+cnt,
                        [&](int a,int b){return row[a]<row[b];});
                    for(int ki=0;ki<k2;++ki){
                        int j=idx[ki];
                        elist.emplace_back(leaf[i],leaf[j],row[j],dmat[(size_t)j*ln+i]);
                    }
                }
            }
            for(int li=0;li<nl;++li)
                for(auto&[s,t,df,db]:leaf_edges[li]){
                    cands[s].emplace_back(df,t); cands[t].emplace_back(db,s);
                }
        }

        std::vector<HashReservoir> res(n);
#pragma omp parallel for schedule(static)
        for(int i=0;i<n;++i){
            res[i].init(cfg_.reservoir_cap);
            const float*si=sketches_.data()+(size_t)i*m;
            for(auto&[dist,j]:cands[i]){
                if(j==(id_t)i)continue;
                uint16_t h=lsh_.hash_pair(si,sketches_.data()+(size_t)j*m);
                res[i].insert(h,j,dist);
            }
        }
        {decltype(cands)().swap(cands);}
        {std::vector<float>().swap(sketches_);}

        g_.init(n,R);
        if(cfg_.final_prune){
#pragma omp parallel for schedule(dynamic,64)
            for(int i=0;i<n;++i){
                std::vector<std::pair<dist_t,id_t>> c; res[i].flush(c);
                std::vector<id_t> out2;
                robust_prune(i,c,dv,cfg_.alpha,R,out2);
                g_.set(i,out2);
            }
        } else {
#pragma omp parallel for schedule(static)
            for(int i=0;i<n;++i){
                std::vector<std::pair<dist_t,id_t>> c; res[i].flush(c);
                std::sort(c.begin(),c.end());
                std::vector<id_t> out2;
                for(int j=0,r=std::min((int)c.size(),R);j<r;++j)out2.push_back(c[j].second);
                g_.set(i,out2);
            }
        }

        if(cfg_.back_edge) back_edge_pass(g_, dv, n, cfg_.alpha);

        entries_=diverse_entries(dv, n, cfg_.k_entry, cfg_.entry_sample, cfg_.seed);
    }

    void beam_search(const float* q, int k, int L,
                     id_t* ids, float* scores) const {
        using P=std::pair<dist_t,id_t>;
        using MaxQ=std::priority_queue<P>;
        using MinQ=std::priority_queue<P,std::vector<P>,std::greater<P>>;

        EpochVisited vis; vis.reset(n_);
        MaxQ best; MinQ frontier;

        static thread_local std::vector<float> tl_node;
        if((int)tl_node.size()<cfg_.dim) tl_node.resize(cfg_.dim);
        const DataView dv = active_dv();

        auto try_push=[&](id_t u) __attribute__((always_inline)){
            if(vis.test_and_mark(u))return;
            const float* uv = dv.row(u, tl_node.data());
            dist_t dv2=query_dist(q,uv,cfg_.dim);
            if((int)best.size()<L||dv2<best.top().first){
                best.push({dv2,u}); frontier.push({dv2,u});
                if((int)best.size()>L) best.pop();
            }
        };

        {dist_t bd=INF_D;id_t be=entries_[0];
         for(id_t e:entries_){
             dist_t dv2=query_dist(q,dv.row(e,tl_node.data()),cfg_.dim);
             if(dv2<bd){bd=dv2;be=e;}
         }
         try_push(be);}

        while(!frontier.empty()){
            auto[dc,c]=frontier.top(); frontier.pop();
            if((int)best.size()>=L&&dc>best.top().first)break;
            const id_t*nbs=g_.row(c); int deg=g_.degree(c);
            for(int j=0;j<deg;++j){
                if(j+1<deg){
                    const void* pf = data16_
                        ? (const void*)(data16_+(size_t)nbs[j+1]*cfg_.dim)
                        : (const void*)(data_  +(size_t)nbs[j+1]*cfg_.dim);
                    __builtin_prefetch(pf,0,1);
                }
                try_push(nbs[j]);
            }
        }

        std::vector<P> res; res.reserve(best.size());
        while(!best.empty()){res.push_back(best.top());best.pop();}
        std::sort(res.begin(),res.end());
        int ok=std::min((int)res.size(),k);
        for(int i=0;i<ok;++i){ids[i]=res[i].second; scores[i]=-res[i].first;}
        for(int i=ok;i<k;++i){ids[i]=NO_ID; scores[i]=-INF_D;}
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────
inline float recall_at_k(const std::vector<id_t>& approx,
                          const std::vector<id_t>& exact, int k, int nq){
    int hits=0;
    for(int qi=0;qi<nq;++qi){
        const id_t*a=approx.data()+(size_t)qi*k;
        const id_t*e=exact .data()+(size_t)qi*k;
        for(int i=0;i<k;++i) for(int j=0;j<k;++j)
            if(a[i]==e[j]){++hits;break;}
    }
    return (float)hits/(float)(nq*k);
}

inline Config make_config(int dim, int n){
    Config cfg; cfg.dim=dim;
    if     (n<  100'000){cfg.leaf_size=128;cfg.min_leaf_size=16;}
    else if(n<  500'000){cfg.leaf_size=256;cfg.min_leaf_size=32;}
    else if(n<5'000'000){cfg.leaf_size=512;cfg.min_leaf_size=64;}
    else                {cfg.leaf_size=1024;cfg.min_leaf_size=128;}
    cfg.max_degree=64;cfg.alpha=1.2f;cfg.leader_frac=0.05f;
    cfg.max_leaders=1000;cfg.top_fanout=10;cfg.second_fanout=3;
    cfg.knn_k=2;cfg.hash_bits=12;cfg.reservoir_cap=128;
    cfg.num_replicas=1;cfg.final_prune=true;cfg.back_edge=true;
    cfg.k_entry=std::min(n,12);cfg.entry_sample=std::min(n,3000);
    cfg.beam_width=128;cfg.seed=42;
    cfg.randomness    = false;
    cfg.coocked          = false;
    return cfg;
}

} // namespace pipnn
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
#include <H5Cpp.h>
#include <sstream>

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
// Precision enum (detected from file size)
// ─────────────────────────────────────────────────────────────────────────────
enum class DType { F32, F16, I8 };
static const char* dtype_name(DType dt){
    return dt==DType::F32?"float32":dt==DType::F16?"float16":"int8";}
static int dtype_bytes(DType dt){
    return dt==DType::F32?4:dt==DType::F16?2:1;}

// ─────────────────────────────────────────────────────────────────────────────
// Detect precision from file size
// ─────────────────────────────────────────────────────────────────────────────
static DType detect_dtype(const std::string& path, int n, int d){
    FILE* f = std::fopen(path.c_str(),"rb");
    if (!f) throw std::runtime_error("Cannot open: "+path);
    std::fseek(f, 0, SEEK_END);
    long fsize = std::ftell(f);
    std::fclose(f);

    long data_bytes = fsize - 8;
    long nd = (long)n * d;
    if (data_bytes == nd*4) return DType::F32;
    if (data_bytes == nd*2) return DType::F16;
    if (data_bytes == nd*1) return DType::I8;
    throw std::runtime_error(
        "Cannot detect dtype for "+path+
        " (file="+std::to_string(fsize)+
        " expected "+std::to_string(8+nd*4)+"/"+
        std::to_string(8+nd*2)+"/"+std::to_string(8+nd)+")");
}

// ─────────────────────────────────────────────────────────────────────────────
// Memory-mapped float16 vector file
// ─────────────────────────────────────────────────────────────────────────────
struct MmapVecs {
    int              n=0, d=0;
    DType            dtype=DType::F16;
    size_t           map_size=0;
    void*            map_ptr=nullptr;
    int              fd_=-1;
    std::vector<uint8_t> buf_;         // used when mmap is unavailable
    const uint8_t*   data_ptr_=nullptr; // always the first data byte

    void open(const std::string& path){
        // ── Read header ───────────────────────────────────────────────
        FILE* f = std::fopen(path.c_str(),"rb");
        if (!f) throw std::runtime_error("Cannot open: "+path);
        std::fread(&n,4,1,f);
        std::fread(&d,4,1,f);
        std::fclose(f);

        dtype             = detect_dtype(path,n,d);
        size_t data_size  = (size_t)n * d * dtype_bytes(dtype);
        map_size          = 8 + data_size;

        printf("  (%s) %.2f GB  (%d x %d)\n",
               dtype_name(dtype), map_size/1e9, n, d);

#ifdef HAVE_MMAP
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ >= 0){
            map_ptr = ::mmap(nullptr, map_size, PROT_READ, MAP_SHARED, fd_, 0);
            if (map_ptr != MAP_FAILED){
                ::madvise(map_ptr, map_size, MADV_WILLNEED);
                // data starts after the 8-byte header; set once here
                data_ptr_ = static_cast<const uint8_t*>(map_ptr) + 8;
                return;
            }
            map_ptr = nullptr;
            ::close(fd_); fd_ = -1;
        }
#endif
        // ── RAM fallback: read data bytes only (skip header) ──────────
        printf("  (mmap unavailable, reading into RAM)\n");
        buf_.resize(data_size);
        FILE* f2 = std::fopen(path.c_str(),"rb");
        std::fseek(f2, 8, SEEK_SET);
        std::fread(buf_.data(), 1, data_size, f2);
        std::fclose(f2);
        data_ptr_ = buf_.data();
    }

    // First data byte, correctly set by open() regardless of path taken
    const void* raw() const { return data_ptr_; }

    void build_index(pipnn::IndexDot& idx) const {
        switch(dtype){
        case DType::F32: idx.build    (static_cast<const float*   >(raw()),n); break;
        case DType::F16: idx.build_f16(static_cast<const uint16_t*>(raw()),n); break;
        case DType::I8:  idx.build_i8 (static_cast<const int8_t*  >(raw()),n); break;
        }
    }

    void set_data(pipnn::IndexDot& idx) const {
        switch(dtype){
        case DType::F32: idx.set_data    (static_cast<const float*   >(raw())); break;
        case DType::F16: idx.set_data_f16(static_cast<const uint16_t*>(raw())); break;
        case DType::I8:  idx.set_data_i8 (static_cast<const int8_t*  >(raw())); break;
        }
    }

    ~MmapVecs(){
#ifdef HAVE_MMAP
        if (map_ptr && map_ptr!=MAP_FAILED) ::munmap(map_ptr, map_size);
        if (fd_ >= 0) ::close(fd_);
#endif
    }
    MmapVecs()=default;
    MmapVecs(const MmapVecs&)=delete;
    MmapVecs& operator=(const MmapVecs&)=delete;
};

// ─────────────────────────────────────────────────────────────────────────────
// Query vectors – small enough to load fully and convert to float32 once
// ─────────────────────────────────────────────────────────────────────────────
struct QueryVecs {
    int n=0, d=0;
    std::vector<float> data;

    void open(const std::string& path){
        FILE* f = std::fopen(path.c_str(),"rb");
        if (!f) throw std::runtime_error("Cannot open: "+path);
        std::fread(&n,4,1,f); std::fread(&d,4,1,f);

        DType dt = detect_dtype(path,n,d);
        size_t elems = (size_t)n * d;
        data.resize(elems);

        if (dt == DType::F32){
            std::fread(data.data(), 4, elems, f);
        } else if (dt == DType::F16){
            std::vector<uint16_t> raw(elems);
            std::fread(raw.data(), 2, elems, f);
            pipnn::f16_to_f32(raw.data(), data.data(), (int)elems);
        } else {
            std::vector<int8_t> raw(elems);
            std::fread(raw.data(), 1, elems, f);
            pipnn::i8_to_f32(raw.data(), data.data(), (int)elems);
        }
        std::fclose(f);
        printf("  (%s -> f32) %.1f MB  (%d x %d)\n",
               dtype_name(dt), elems*4/1e6, n, d);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Ground-truth loader
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<int> read_gt(const std::string& path, int& N, int& K){
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open GT: "+path);
    f.read(reinterpret_cast<char*>(&N),4);
    f.read(reinterpret_cast<char*>(&K),4);
    std::vector<int> g((size_t)N*K);
    f.read(reinterpret_cast<char*>(g.data()),(size_t)N*K*4);
    return g;
}

// ─────────────────────────────────────────────────────────────────────────────
// Chunked query + recall@k  (only checks against the top-k GT entries)
// ─────────────────────────────────────────────────────────────────────────────

void resultH5(std::vector<pipnn::id_t> knns, std::vector<float> dists, int k, int n,
     double buildT, double preproT, double queryT,  pipnn::Config cfg, const char* output)
{
    H5::H5File file(output, H5F_ACC_TRUNC);

    // Dataset dimensions
    hsize_t dims[2] = {static_cast<hsize_t>(n),
                       static_cast<hsize_t>(k)};

    H5::DataSpace dataspace(2, dims);

    // Create knns dataset
    H5::DataSet knnDataset =
        file.createDataSet("knns",
                           H5::PredType::NATIVE_INT,
                           dataspace);

    knnDataset.write(knns.data(), H5::PredType::NATIVE_INT);

    // Create dists dataset
    H5::DataSet distDataset =
        file.createDataSet("dists",
                           H5::PredType::NATIVE_FLOAT,
                           dataspace);

    distDataset.write(dists.data(), H5::PredType::NATIVE_FLOAT);

    // ---------- Attributes ----------

    // String datatype
    H5::StrType strdatatype(H5::PredType::C_S1, H5T_VARIABLE);
    H5::DataSpace scalarSpace(H5S_SCALAR);

    // algo
    {
        H5::Attribute attr = file.createAttribute(
            "algo", strdatatype, scalarSpace);
        std::string value = "PIPNN";
        attr.write(strdatatype, value);
    }

    // task
    {
        H5::Attribute attr = file.createAttribute(
            "task", strdatatype, scalarSpace);
        std::string value = "task1";
        attr.write(strdatatype, value);
    }

    // params
    {
        H5::Attribute attr = file.createAttribute(
            "params", strdatatype, scalarSpace);

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

    // buildtime
    {
        H5::Attribute attr = file.createAttribute(
            "buildtime",
            H5::PredType::NATIVE_DOUBLE,
            scalarSpace);
        double bpT = buildT + preproT;
        attr.write(H5::PredType::NATIVE_DOUBLE, &bpT);
    }

    // querytime
    {
        H5::Attribute attr = file.createAttribute(
            "querytime",
            H5::PredType::NATIVE_DOUBLE,
            scalarSpace);

        attr.write(H5::PredType::NATIVE_DOUBLE, &queryT);
    }

    file.close();
}

static void run_split(const char* label,
                      pipnn::IndexDot& idx,
                      const float* queries, int nq, int D,
                      const std::vector<int>& gt, int gt_k,
                      int k, const std::vector<int>& bws,
                      double preproT, double buildT, pipnn::Config cfg,
                      const char* output,
                      int chunk = 10000) {
    printf("\n-- %s (%d queries, recall@%d) ----\n",label,nq,k);
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
                        if ((int)r[i] + 1 == g[j]) { ++total_hits; break; }
            }
        }
        for (int i; i < k*chunk; i++)
        {
            ids[i]++;
        }
        double rec = (double)total_hits / (double)((long long)nq * k);
        printf("  %-6d  %.4f     %.0f\n", bw, rec, nq / total_s);
        resultH5(ids, scores, k, chunk, buildT, preproT, total_s,  cfg, output);
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
    const int   K       = std::stoi(argv[4]);
    double Prepro_time = std::stod(argv[5]);
    int         BW      = (argc >= 10 && argv[7][0] != '-') ? std::stoi(argv[6]) : 256;
    const int         MAX_DEG  = (argc >= 8) ? std::stoi(argv[7]) : 64;
    const float       ALPHA    = (argc >= 9) ? std::stof(argv[8]) : 1.2f;
    const int         LEAF_SZ  = (argc >= 10) ? std::stoi(argv[9]) : 512;
    const int         MIN_LEAF = (argc >= 11) ? std::stoi(argv[10]) : 32;
    const int         K_ENTRY  = (argc >= 12) ? std::stoi(argv[11]) : 12;
    const int         ENT_SAMP = (argc >= 13) ? std::stoi(argv[12]) : 3000;
    const int         HBITS    = (argc >= 14) ? std::stoi(argv[13]) : 12;
    const int         RES_CAP  = (argc >= 15) ? std::stoi(argv[14]) : 128;
    const int         REPLICAS = (argc >= 16) ? std::stoi(argv[15]) : 1;
    const bool        F_PRUNE  = (argc >= 17) ? (std::stoi(argv[16]) != 0) : true;
    const bool        B_EDGE   = (argc >= 18) ? (std::stoi(argv[17]) != 0) : true;
    const int         N_THREADS= (argc >= 19) ? std::stoi(argv[18]) : 0;
    const uint64_t    SEED     = (argc >= 20) ? (uint64_t)std::stoull(argv[19]) : 42;
    const bool        RAND     = (argc >= 21) ? std::stoi(argv[20]) : false;
    const bool        COOCKED     = (argc >= 22) ? std::stoi(argv[21]) : false;
    const char* output = argv[22];
    std::string save_idx, load_idx;
    for (int i = 5; i < argc - 1; ++i) { //starts in 5 because train_f ... K are necessary.
        if (!std::strcmp(argv[i], "--save-index")) save_idx = argv[++i];
        if (!std::strcmp(argv[i], "--load-index")) load_idx = argv[++i];
    }

#if PIPNN_AVX2
    printf("AVX2+FMA  enabled\n");
#else
    printf("AVX2      disabled\n");
#endif
    auto cont_prepro = clk::now();
    printf("\nLoading train ...\n");
    MmapVecs train; train.open(train_f);
    const int Nt=train.n, D=train.d;

    printf("Loading allknn queries ...\n");
    QueryVecs allq; allq.open(allq_f);
    int Nallgt,Kallgt; printf("Loading allknn GT ...\n");
    auto allgt = read_gt(allgt_f, Nallgt, Kallgt);

    if (D!=allq.d){
        fprintf(stderr,"Dimension mismatch: train=%d allknn=%d\n",
                D,allq.d); return 1;}

    auto cfg = pipnn::make_config(D, Nt); cfg.beam_width = BW;
    printf("Config : leaf_size=%-4d  max_degree=%-3d  alpha=%.1f  "
           "k_entry=%d  bw=%d\n\n",
           cfg.leaf_size,cfg.max_degree,cfg.alpha,cfg.k_entry,BW);
    Prepro_time += sec(cont_prepro);
    printf("Prepro time is %f\n", Prepro_time);
    pipnn::IndexDot idx(cfg);
    double build_s = 0.0;
    if (!load_idx.empty() && std::ifstream(load_idx).good()){
        printf("Loading pre-built index from %s ...\n", load_idx.c_str());
        auto t = clk::now();
        if (!idx.load(load_idx)){ fprintf(stderr,"Load failed\n"); return 1; }
        train.set_data(idx);
        printf("  Loaded in %.2f s\n", sec(t));
    } else {
        printf("Building index on %d x %d vectors (%s) ...\n",
               Nt, D, dtype_name(train.dtype));
        auto t = clk::now();
        train.build_index(idx);
        build_s = sec(t);
        auto st = idx.stats();
        printf("  Build time : %.2f s\n",   build_s);
        printf("  Avg degree : %.1f\n",      st.avg_deg);
        printf("  Bidir      : %.1f%%\n\n",  100.0*st.frac_bidir);
        if (!save_idx.empty()){
            printf("Saving index to %s ...\n", save_idx.c_str());
            idx.save(save_idx);}
    }

    std::vector<int> bws = {BW};
    run_split("allknn", idx, allq.data.data(), allq.n, D, allgt, Kallgt, K, bws, Prepro_time, build_s, cfg, output);
    printf("\nDone.\n");
    return 0;
}


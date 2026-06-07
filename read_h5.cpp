#include <iostream>
#include <vector>
#include <string>
#include "pipnn_opt.hpp"
#include <chrono>
#include <cstdio>
#include <hdf5/serial/H5Cpp.h>   // Cambia a <H5Cpp.h> si da error

// g++ -O3 -std=c++17 -fopenmp -march=native read_h5.cpp -o read_h5 $(pkg-config --cflags --libs hdf5) -lhdf5_cpp

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

int main() {
    const std::string filePath = 
        "/home/ivan/.cache/huggingface/hub/datasets--SISAP-Challenges--SISAP2026/snapshots/ebc9e713a296f0aff6b5109c8c1961b3b75caadf/wikipedia-small/benchmark-dev-wikipedia-bge-m3-small.h5"; //Ejecutar primero download_data y mirar el path donde se han descargado los datos!

    try {
        H5::H5File file(filePath, H5F_ACC_RDONLY);
        H5::DataSet dataset = file.openDataSet("train");

        H5::DataSpace dataspace = dataset.getSpace();

        int rank = dataspace.getSimpleExtentNdims();
        std::vector<hsize_t> dims(rank);
        dataspace.getSimpleExtentDims(dims.data());

        int num_vectors = dims[0];
        int dim = dims[1];
        hsize_t total_elements = num_vectors * dim;

        // std::cout << "Cargando " << num_vectors << " vectores de dimensión " << dim << " (float)..." << std::endl;

        // Leer datos como uint16_t (float16 en HDF5)
        std::vector<float> data(total_elements);
        dataset.read(data.data(), H5::PredType::NATIVE_FLOAT);

        /////////// Query //////////////
        H5::DataSet ds_q = file.openDataSet("itest/queries");
        H5::DataSpace space_q = ds_q.getSpace();

        std::vector<hsize_t> dims_q(2);
        space_q.getSimpleExtentDims(dims_q.data());

        int n_queries = dims_q[0];
        int j_queries = dims_q[1];
        hsize_t total_q = n_queries * j_queries;

        std::vector<float> queries(total_q);
        ds_q.read(queries.data(), H5::PredType::NATIVE_FLOAT);


    ////////// ground truth /////////////////
    H5::DataSet ds_gr = file.openDataSet("itest/knns");
    H5::DataSpace space_gr = ds_gr.getSpace();

    std::vector<hsize_t> dims_gr(2);
    space_gr.getSimpleExtentDims(dims_gr.data());

    int n_gr = dims_gr[0];
    int j_gr = dims_gr[1];
    hsize_t total_gr = n_gr * j_gr;

    std::vector<pipnn::id_t> gt(total_gr);
    ds_gr.read(gt.data(), H5::PredType::NATIVE_UINT32);




    //////////// pipnn ////////////
    const int K   = 15;
    // printf("Computing ground truth...\n");
    auto t0  = clk::now();
    // auto gt  = brute_force_knn(data, num_vectors, queries, n_queries, dim, K);
    // auto gt  = brute_force_mip(data, num_vectors, queries, n_queries, dim, K);
    // printf("  exact k-NN:  %ld ms\n\n", ms_since(t0));


     printf("── opt) pipnn::IndexOpt (back-edges + medoid) ──\n");
    {
        pipnn::IndexOpt::ConfigOpt cfg;
        cfg.dim            = dim;
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
        idx.build(data.data(), num_vectors);
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
        idx.query(queries.data(), n_queries, K, ids, dists);
        long qms = ms_since(t0);
        printf("  query:   %ld ms  (%.0f QPS)\n", qms, qms>0?1000.0*n_queries/qms:0.);
        printf("  recall@%d: %.4f\n\n", K, pipnn::recall_at_k(ids, gt, K, n_queries));

        // ── Save & reload ────────────────────────────────
        printf("  Saving index to /tmp/pipnn_test.bin ...\n");
        if (idx.save("/tmp/pipnn_test.bin")) {
            pipnn::IndexOpt idx2(cfg);
            idx2.load("/tmp/pipnn_test.bin");
            idx2.set_data(data.data());

            ids.clear(); dists.clear();
            idx2.query(queries.data(), n_queries, K, ids, dists);
            printf("  recall after reload: %.4f\n\n",
                   pipnn::recall_at_k(ids, gt, K, n_queries));
        }
    }

    // ─────────────────────────────────────────────────────
    // A)  Basic Index  (pipnn.hpp)
    // ─────────────────────────────────────────────────────
    printf("── A) pipnn::Index (v1) ─────────────────────\n");
    {
        pipnn::Config cfg;
        cfg.dim          = dim;
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
        idx.build(data.data(), num_vectors);
        printf("  build:   %ld ms\n", ms_since(t0));

        std::vector<pipnn::id_t>   ids;
        std::vector<pipnn::dist_t> dists;
        t0 = clk::now();
        idx.query(queries.data(), n_queries, K, ids, dists);
        long qms = ms_since(t0);
        printf("  query:   %ld ms  (%.0f QPS)\n", qms, qms>0?1000.0*n_queries/qms:0.);
        printf("  recall@%d: %.4f\n\n", K, pipnn::recall_at_k(ids, gt, K, n_queries));
    }

    // ─────────────────────────────────────────────────────
    // B)  IndexV2 with back-edge pass + medoid entry
    // ─────────────────────────────────────────────────────
    printf("── B) pipnn::IndexV2 (back-edges + medoid) ──\n");
    {
        pipnn::IndexV2::ConfigV2 cfg;
        cfg.dim            = dim;
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
        idx.build(data.data(), num_vectors);
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
        idx.query(queries.data(), n_queries, K, ids, dists);
        long qms = ms_since(t0);
        printf("  query:   %ld ms  (%.0f QPS)\n", qms, qms>0?1000.0*n_queries/qms:0.);
        printf("  recall@%d: %.4f\n\n", K, pipnn::recall_at_k(ids, gt, K, n_queries));

        // ── Save & reload ────────────────────────────────
        printf("  Saving index to /tmp/pipnn_test.bin ...\n");
        if (idx.save("/tmp/pipnn_test.bin")) {
            pipnn::IndexV2 idx2(cfg);
            idx2.load("/tmp/pipnn_test.bin");
            idx2.set_data(data.data());

            ids.clear(); dists.clear();
            idx2.query(queries.data(), n_queries, K, ids, dists);
            printf("  recall after reload: %.4f\n\n",
                   pipnn::recall_at_k(ids, gt, K, n_queries));
        }
    }

    // ─────────────────────────────────────────────────────
    // C)  Recall vs. beam-width sweep
    // ─────────────────────────────────────────────────────
    printf("── C) Recall vs. beam width (IndexV2) ───────\n");
    {
        pipnn::IndexV2::ConfigV2 cfg;
        cfg.dim            = dim;
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
        idx.build(data.data(), num_vectors);

        printf("  beam_width  recall@%d   QPS\n", K);
        for (int bw : {32, 64, 128, 256, 512}) {
            std::vector<pipnn::id_t>   ids;
            std::vector<pipnn::dist_t> dists;
            auto t = clk::now();
            idx.query(queries.data(), n_queries, K, ids, dists, bw);
            long ms = ms_since(t);
            printf("  %10d  %.4f     %.0f\n",
                   bw,
                   pipnn::recall_at_k(ids, gt, K, n_queries),
                   ms > 0 ? 1000.0*n_queries/ms : 0.);
        }
        printf("\n");
    }

    } catch (H5::Exception& err) {
        std::cerr << "❌ Error HDF5: " << err.getDetailMsg() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }



    return 0;
}
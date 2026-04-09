import h5py
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import hnswlib
import pickle


def print_structure(name, obj):
    print(name)

def read_train_h5(filename):
    #TODO: make lazy or do not read until needed
    with h5py.File(filename, "r") as f:
    # Print top-level structure
        f.visititems(print_structure)
        train = f["train"][:]   # shape (200000, 1024), dtype=float16
        print("Train", train.shape, train.dtype)

    print()
    return train

def read_test_h5(filename):
    #TODO: make lazy or do not read until needed
    with h5py.File(filename, "r") as f:
        
        # print(type(train))
        iquery = f["itest/queries"][:]
        print("Query ", iquery.shape, iquery.dtype)
        iquery_knn = f["itest/knns"][:]
        print("Query knn ", iquery_knn.shape, iquery_knn.dtype)

        oquery = f["otest/queries"][:]
        print("Query ", oquery.shape, oquery.dtype)
        oquery_knn = f["otest/knns"][:]
        print("Query knn ", oquery_knn.shape, oquery_knn.dtype)
        
        # print(oquery_knn[1,:5])  ## itself not included
        # oquery_dists = f["itest/dists"][:]  ## already sorted
        # print(oquery_dists[1,:5])
    print()
    return iquery, iquery_knn, oquery, oquery_knn

def do_pca(data, n_dim=256, sub_size=50000, plot=False):
    
    # sample subset to fit PCA eq
    X_sub = data[np.random.choice(len(data), size=sub_size, replace=False)]
    print("subsampled X dim: ", X_sub.shape)


    pca = PCA(n_components=n_dim, svd_solver="randomized")
    # pca = PCA()
    pca.fit(X_sub)

    

    explained_variance = pca.explained_variance_ratio_
    # print(explained_variance)
    print("Explained variance in subsample: ", np.sum(explained_variance))
    

    if plot:
        cumulative_variance = np.cumsum(explained_variance)
        # Plot
        plt.figure(figsize=(8, 5))

        # Bar plot for individual variance
        plt.bar(range(1, len(explained_variance) + 1),
                explained_variance,
                alpha=0.6,
                label='Individual explained variance')

        # Line plot for cumulative variance
        plt.plot(range(1, len(cumulative_variance) + 1),
                cumulative_variance,
                marker='o',
                color='red',
                label='Cumulative explained variance')

        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('PCA Explained Variance')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
    
    # X_reduced = pca.transform(data)
    return pca

def construct_hnswg(X, ef_construction=200, M=16, ef=50, n_threads=1):
    n_obs, n_dim = X.shape
    ids = np.arange(n_obs) # temporal indices?

    p = hnswlib.Index(space = 'ip', dim = n_dim)
    p.init_index(max_elements = n_obs, ef_construction = ef_construction, M = M)
    
    # tunning params
    p.set_ef(ef=ef)
    p.set_num_threads(num_threads=n_threads)
    
    # we add the data
    p.add_items(X, ids)


    return p

def query_hnswg(Y, p, k=15):
    labels, distances = p.knn_query(Y, k=k)
    return labels, distances

def recallk(computed, gt):
    num_queries, k = computed.shape
    kgt = gt[:,:k]-1
    print("Ground Truth shape ", kgt.shape)
    print("Computed shape ", computed.shape)
    

    # Measure recall
    correct = 0
    for i in range(num_queries):
        for label in computed[i]:
            for correct_label in kgt[i]:
                if label == correct_label:
                    correct += 1
                    break

    # average recall@k
    recall = float(correct)/(k*num_queries)
    print(recall, correct)
    print()
    return recall


if __name__=="__main__":
    # script just to check ideas, will make C++ optimized code with possible changes in each section
    # TODO: 
    ## make it into classes
    ## use faiss ; hnswlib ; scann
    
    n_threads = 8 # idk
    k=15
    n_dim = 512; m = 128; ef = 256; ef_construction=512;
    file_path = "../benchmark-dev-wikipedia-bge-m3-small.h5"

    train = read_train_h5(filename=file_path)

    pca = do_pca(data=train, n_dim=n_dim, sub_size=50000, plot=False)
    
    train_red = pca.transform(train)
    p = construct_hnswg(X=train_red, ef_construction=ef_construction, M=m, ef=ef, n_threads=n_threads) # com son embeddings la M hauria de ser gran ig

    iquery, iquery_knn, oquery, oquery_knn = read_test_h5(filename=file_path)

    iquery_red = pca.transform(iquery)
    ilabels, _ = query_hnswg(Y=iquery_red, p=p, k=k)

    oquery_red = pca.transform(oquery)
    olabels, _ = query_hnswg(Y=oquery_red, p=p, k=k)

    irecall = recallk(computed=ilabels, gt=iquery_knn)
    orecall = recallk(computed=olabels, gt=oquery_knn)

    # print(ilabels[:10,:k])
    # print(iquery_knn[:10,:k])



    

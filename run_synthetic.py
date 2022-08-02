import numpy as np
import sys
import time

from cluster.selfrepresentation import ElasticNetSubspaceClustering, SparseSubspaceClusteringOMP, LowRankRepresentation\
    , BlockDiagonalRepresentation
from gen_union_of_subspaces import gen_union_of_subspaces
from metrics.cluster.accuracy import clustering_accuracy
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn import cluster
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# =================================================
# Generate dataset where data is drawn from a union of subspaces
# =================================================
# ambient_dim = 20
# # subspace_dim = 4
# subspace_dims = [2, 4, 6, 8, 10, 12, 14, 16]
# num_subspaces = 4
# num_points_per_subspace = 200
# noises = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
# algorithm_nmi = list()
# algorithm_ari = list()
# algorithm_time = list()

# for subspace_dim in subspace_dims:
#     # data, label = gen_union_of_subspaces(ambient_dim, subspace_dim, num_subspaces, num_points_per_subspace, 0.01)
#
#     # =================================================
#     # Create cluster objects
#     # =================================================
#
#     # Baseline: non-subspace clustering methods
#     model_kmeans = cluster.KMeans(n_clusters=num_subspaces)  # k-means as baseline
#     model_spectral = cluster.SpectralClustering(n_clusters=num_subspaces, affinity='nearest_neighbors',
#                                                 n_neighbors=6)  # spectral clustering as baseline
#
#     # Elastic net subspace clustering with a scalable active support elastic net solver
#     # You et al., Oracle Based Active Set Algorithm for Scalable Elastic Net Subspace Clustering, CVPR 2016
#     model_ensc = ElasticNetSubspaceClustering(n_clusters=num_subspaces, algorithm='spams', gamma=500)
#
#     # Sparse subspace clusterign by orthogonal matching pursuit (SSC-OMP)
#     # You et al., Scalable Sparse Subspace Clustering by Orthogonal Matching Pursuit, CVPR 2016
#     model_ssc_omp = SparseSubspaceClusteringOMP(n_clusters=num_subspaces, n_nonzero=subspace_dim, thr=1e-5)
#
#     model_lrr = LowRankRepresentation(n_clusters=num_subspaces)
#
#     model_bdr = BlockDiagonalRepresentation(n_clusters=num_subspaces)
#
#     clustering_algorithms = (
#         # ('KMeans', model_kmeans),
#         # ('Spectral Clustering', model_spectral),
#         # ('EnSC', model_ensc),
#         ('SSC-OMP', model_ssc_omp),
#         # ('LRR', model_lrr),
#         # ('BDR', model_bdr)
#     )
#     # print(data)
#     # from sklearn.cluster import DBSCAN
#     #
#     # # print(len(data[0]))
#     # clustering = DBSCAN(eps=0.01, min_samples=5).fit(data[:, [0, 19]])
#     # print(clustering.labels_)
#
#     names = ['KMeans', 'Spectral Clustering', 'EnSC', 'SSC-OMP', 'LRR']
#
#     for name, algorithm in clustering_algorithms:
#         t_begin = time.time()
#         algorithm.fit(data)
#         t_end = time.time()
#         acc = clustering_accuracy(label, algorithm.labels_)
#         nmi = normalized_mutual_info_score(label, algorithm.labels_, average_method='geometric')
#         ari = adjusted_rand_score(label, algorithm.labels_)
#
#         print('Algorithm: {}. acc: {}, nmi: {}, ari: {}, Running time: {}'.format(name, acc, nmi, ari, t_end - t_begin))
#         algorithm_nmi.append(nmi)
#         algorithm_ari.append(ari)
#         algorithm_time.append(t_end - t_begin)
angles = [10, 20, 30, 40, 50, 60]
algorithm_nmi = list()
algorithm_ari = list()
algorithm_time = list()
for angle in angles:
    raw_data = pd.read_csv("data/"+str(angle)+".csv")
    raw_data = np.array(raw_data)
    data = []
    label = []
    for i, j, k, l in raw_data:
        data.append([i, j, k])
        label.append(int(l))
    # =================================================
    # Create cluster objects
    # =================================================

    # Baseline: non-subspace clustering methods
    model_kmeans = cluster.KMeans(n_clusters=2)  # k-means as baseline
    model_spectral = cluster.SpectralClustering(n_clusters=2, affinity='nearest_neighbors',
                                                n_neighbors=6)  # spectral clustering as baseline

    # Elastic net subspace clustering with a scalable active support elastic net solver
    # You et al., Oracle Based Active Set Algorithm for Scalable Elastic Net Subspace Clustering, CVPR 2016
    model_ensc = ElasticNetSubspaceClustering(n_clusters=2, algorithm='spams', gamma=500)

    # Sparse subspace clusterign by orthogonal matching pursuit (SSC-OMP)
    # You et al., Scalable Sparse Subspace Clustering by Orthogonal Matching Pursuit, CVPR 2016
    model_ssc_omp = SparseSubspaceClusteringOMP(n_clusters=2, n_nonzero=3, thr=1e-5)

    model_lrr = LowRankRepresentation(n_clusters=2)

    model_bdr = BlockDiagonalRepresentation(n_clusters=2)

    clustering_algorithms = (
        # ('KMeans', model_kmeans),
        # ('Spectral Clustering', model_spectral),
        ('EnSC', model_ensc),
        # ('SSC-OMP', model_ssc_omp),
        # ('LRR', model_lrr),
    )

    names = ['KMeans', 'Spectral Clustering', 'EnSC', 'SSC-OMP', 'LRR']

    for name, algorithm in clustering_algorithms:
        t_begin = time.time()
        algorithm.fit(data)
        t_end = time.time()
        acc = clustering_accuracy(label, algorithm.labels_)
        nmi = normalized_mutual_info_score(label, algorithm.labels_, average_method='geometric')
        ari = adjusted_rand_score(label, algorithm.labels_)

        print('Algorithm: {}. acc: {}, nmi: {}, ari: {}, Running time: {}'.format(name, acc, nmi, ari, t_end - t_begin))
        algorithm_nmi.append(nmi)
        algorithm_ari.append(ari)
        algorithm_time.append(t_end - t_begin)

# Adjusted Random Index
# Normalized Mutual Information
plt.plot(angles, algorithm_time, linewidth=2)
plt.title('EnSC - Time', fontsize=20)
plt.xlabel("Angle", fontsize=14)
plt.ylabel("Time", fontsize=14)
plt.tick_params(axis='both', labelsize=10)
plt.savefig('image/angle/EnSC1.png')
plt.show()

plt.plot(angles, algorithm_nmi, linewidth=2)
plt.title('EnSC - NMI', fontsize=20)
plt.xlabel("Angle", fontsize=14)
plt.ylabel("Normalized Mutual Information", fontsize=14)
plt.tick_params(axis='both', labelsize=10)
plt.savefig('image/angle/EnSC2.png')
plt.show()

plt.plot(angles, algorithm_ari, linewidth=2)
plt.title('EnSC - ARI', fontsize=20)
plt.xlabel("Angle", fontsize=14)
plt.ylabel("Adjusted Random Index", fontsize=14)
plt.tick_params(axis='both', labelsize=10)
plt.savefig('image/angle/EnSC3.png')
plt.show()




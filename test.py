# -*- coding = utf-8 -*-
# @Time : 29/05/2022 18:23
# @Author : Yan Zhu
# @File : test.py
# @Software : PyCharm

import time
from scipy.io import loadmat
from cluster.selfrepresentation import ElasticNetSubspaceClustering, SparseSubspaceClusteringOMP, LowRankRepresentation
from gen_union_of_subspaces import gen_union_of_subspaces
from metrics.cluster.accuracy import clustering_accuracy
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn import cluster
import matplotlib.pyplot as plt

data = loadmat('./data/ORL_32x32.mat')
img = data['fea']
raw_label = data['gnd']
label = list()
# print(len(img))
for l in raw_label:
    label.append(l[0])
# print(label)
num_clusters = 10
# Baseline: non-subspace clustering methods
model_kmeans = cluster.KMeans(n_clusters=num_clusters)  # k-means as baseline

model_spectral = cluster.SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors',
                                            n_neighbors=5)  # spectral clustering as baseline

# Our work: elastic net subspace clustering (EnSC)
# You et al., Oracle Based Active Set Algorithm for Scalable Elastic Net Subspace Clustering, CVPR 2016
model_ensc = ElasticNetSubspaceClustering(n_clusters=num_clusters, affinity='nearest_neighbors', algorithm='spams',
                                          active_support=True, gamma=200, tau=0.9)

# Our work: sparse subspace clusterign by orthogonal matching pursuit (SSC-OMP)
# You et al., Scalable Sparse Subspace Clustering by Orthogonal Matching Pursuit, CVPR 2016
model_ssc_omp = SparseSubspaceClusteringOMP(n_clusters=num_clusters, affinity='symmetrize', n_nonzero=5, thr=1.0e-5)

model_lrr = LowRankRepresentation(n_clusters=num_clusters)

clustering_algorithms = (
    ('KMeans', model_kmeans),
    ('Spectral Clustering', model_spectral),
    ('EnSC', model_ensc),
    ('SSC-OMP', model_ssc_omp),
    ('LRR', model_lrr)
)
# print(img)
# from sklearn.cluster import DBSCAN
# print(len(img[0]))
# clustering = DBSCAN(eps=10, min_samples=5).fit(img[:,[0, 1023]])
# print(clustering.labels_)
#
# print(clustering)
names = ['KMeans', 'Spectral Clustering', 'EnSC', 'SSC-OMP', 'LRR']
algorithm_nmi = list()
algorithm_ari = list()
algorithm_time = list()
for name, algorithm in clustering_algorithms:
    t_begin = time.time()
    algorithm.fit(img)
    t_end = time.time()
    acc = clustering_accuracy(label, algorithm.labels_)
    nmi = normalized_mutual_info_score(label, algorithm.labels_, average_method='geometric')
    ari = adjusted_rand_score(label, algorithm.labels_)

    print('Algorithm: {}. acc: {}, nmi: {}, ari: {}, Running time: {}'.format(name, acc, nmi, ari, t_end - t_begin))
    algorithm_time.append(t_end - t_begin)
    algorithm_ari.append(ari)
    algorithm_nmi.append(nmi)
plt.plot(names, algorithm_time, linewidth=2)
plt.title('Algorithm - Time', fontsize=20)
plt.xlabel("Algorithm", fontsize=14)
plt.ylabel("Time", fontsize=14)
plt.tick_params(axis='both', labelsize=10)
plt.savefig('image/hopkins/hopkins1.png')
plt.show()

plt.plot(names, algorithm_nmi, linewidth=2)
plt.title('Algorithm - NMI', fontsize=20)
plt.xlabel("Algorithm", fontsize=14)
plt.ylabel("Normalized Mutual Information", fontsize=14)
plt.tick_params(axis='both', labelsize=10)
plt.savefig('image/hopkins/hopkins2.png')
plt.show()

plt.plot(names, algorithm_ari, linewidth=2)
plt.title('Algorithm - ARI', fontsize=20)
plt.xlabel("Algorithm", fontsize=14)
plt.ylabel("Adjusted Random Index", fontsize=14)
plt.tick_params(axis='both', labelsize=10)
plt.savefig('image/hopkins/hopkins3.png')
plt.show()

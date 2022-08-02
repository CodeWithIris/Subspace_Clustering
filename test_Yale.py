# -*- coding = utf-8 -*-
# @Time : 29/05/2022 20:24
# @Author : Yan Zhu
# @File : test_Yale.py
# @Software : PyCharm
import time

import numpy as np
import scipy
from scipy.io import loadmat
from cluster.selfrepresentation import ElasticNetSubspaceClustering, SparseSubspaceClusteringOMP,LowRankRepresentation
from gen_union_of_subspaces import gen_union_of_subspaces
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from metrics.cluster.accuracy import clustering_accuracy
from sklearn import cluster
import matplotlib.pyplot as plt


def getPhoto(fname):
    data = scipy.io.loadmat(fname)

    X = data['fea']

    row, col = X.shape
    images = []
    for i in range(row):
        img = np.array(col)
        img = X[i]
        img = img.reshape(32, 32).T
        images.append(img)

    images = np.array(images)

    # 15,11为创建子图的个数，根据数据集包含的信息填写
    # 例如Yale数据集包含15个人的11张图像，因此可以写15,11，每一行是一个样本的信息
    fig, axes = plt.subplots(5, 10
                             , subplot_kw={"xticks": [], "yticks": []}
                             )

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i]
                  , cmap="gray"
                  )

    return fig
# fig = getPhoto('./data/YaleBCrop025.mat')
# fig = getPhoto('./data/YaleB_32x32.mat')
# fig.show()
# fig.savefig('image/yale/yale_face.png')
raw_data = loadmat('./data/YaleB_32x32.mat')
# nsamples, nx, ny = raw_data['Y'].shape
# data = np.array(raw_data['Y']).reshape((nsamples, nx*ny))
# label = list()
# for raw_label in np.array(raw_data["s"][0]):
#     for i in raw_label:
#         label.append(i[0])
# label[:] = label[:2016]
data = raw_data['fea']
label = list()
for i in raw_data['gnd']:
    label.append(i[0])
num_clusters = 10
# Baseline: non-subspace clustering methods
model_kmeans = cluster.KMeans(n_clusters=num_clusters)  # k-means as baseline

model_spectral = cluster.SpectralClustering(n_clusters=num_clusters,affinity='nearest_neighbors',n_neighbors=5)  # spectral clustering as baseline

# Our work: elastic net subspace clustering (EnSC)
# You et al., Oracle Based Active Set Algorithm for Scalable Elastic Net Subspace Clustering, CVPR 2016
model_ensc = ElasticNetSubspaceClustering(n_clusters=num_clusters,affinity='nearest_neighbors',algorithm='spams',active_support=True,gamma=200,tau=0.9)

# Our work: sparse subspace clusterign by orthogonal matching pursuit (SSC-OMP)
# You et al., Scalable Sparse Subspace Clustering by Orthogonal Matching Pursuit, CVPR 2016
model_ssc_omp = SparseSubspaceClusteringOMP(n_clusters=num_clusters,affinity='symmetrize',n_nonzero=5,thr=1.0e-5)

model_lrr = LowRankRepresentation(n_clusters=num_clusters)

clustering_algorithms = (
    ('KMeans', model_kmeans),
    ('Spectral Clustering', model_spectral),
    ('EnSC', model_ensc),
    ('SSC-OMP', model_ssc_omp),
    ('LRR', model_lrr)
)
algorithm_nmi = list()
algorithm_ari = list()
algorithm_time = list()
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
plt.plot(names, algorithm_time, linewidth=2)
plt.title('Algorithm - Time', fontsize=20)
plt.xlabel("Algorithm", fontsize=14)
plt.ylabel("Time", fontsize=14)
plt.tick_params(axis='both', labelsize=10)
plt.savefig('image/yale/yale1.png')
plt.show()

plt.plot(names, algorithm_nmi, linewidth=2)
plt.title('Algorithm - NMI', fontsize=20)
plt.xlabel("Algorithm", fontsize=14)
plt.ylabel("Normalized Mutual Information", fontsize=14)
plt.tick_params(axis='both', labelsize=10)
plt.savefig('image/yale/yale2.png')
plt.show()

plt.plot(names, algorithm_ari, linewidth=2)
plt.title('Algorithm - ARI', fontsize=20)
plt.xlabel("Algorithm", fontsize=14)
plt.ylabel("Adjusted Random Index", fontsize=14)
plt.tick_params(axis='both', labelsize=10)
plt.savefig('image/yale/yale3.png')
plt.show()
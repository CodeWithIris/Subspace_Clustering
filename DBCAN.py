# -*- coding = utf-8 -*-
# @Time : 19/06/2022 10:32
# @Author : Yan Zhu
# @File : DBCAN.py
# @Software : PyCharm
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from scipy.io import loadmat
from gen_union_of_subspaces import gen_union_of_subspaces

angles = [10, 20, 30, 40, 50, 60]
# data, label = gen_union_of_subspaces(3, 2, 2, 200, 0.01)
# X = np.array([[1, 2, 3], [2, 2, 4], [2, 3, 5],
#               [8, 7, 12], [8, 8, 14], [25, 80, 47]])
# raw_data = loadmat('./data/YaleB_32x32.mat')
# data = raw_data['fea']
# row, col = data.shape
# images = []
# for i in range(row):
#     img = np.array(col)
#     img = data[i]
#     img = img.reshape(32, 32).T
#     images.append(img)
#
# images = np.array(images)
# print(images)
x, y = make_moons(n_samples=200, noise=0.09, random_state=0)

# K-means
from sklearn.cluster import KMeans

km = KMeans(n_clusters=2, init="k-means++", n_init=10, max_iter=300)
y_km = km.fit_predict(x)
plt.scatter(x[y_km == 0, 0], x[y_km == 0, 1], c="green", marker="o", label="Cluster 1")
plt.scatter(x[y_km == 1, 0], x[y_km == 1, 1], c="red", marker="s", label="Cluster 2")
plt.title("K-Means")
plt.legend()
plt.savefig('image/DBSCAN/Kmeans.png')
plt.show()

# DBSCAN
dbscan = DBSCAN(eps=0.2,min_samples=5,metric="euclidean")
dbscan_y = dbscan.fit_predict(x)
print(dbscan_y)
plt.scatter(x[dbscan_y==0,0],x[dbscan_y==0,1],c="green",marker="o",label="Cluster 1")
plt.scatter(x[dbscan_y==1,0],x[dbscan_y==1,1],c="red",marker="s",label="Cluster 2")
plt.scatter(x[dbscan_y==-1,0],x[dbscan_y==-1,1],c="blue",marker="^",label="Noise point")
plt.title("DBSCAN")
plt.legend()
plt.savefig('image/DBSCAN/DBSCAN.png')
plt.show()

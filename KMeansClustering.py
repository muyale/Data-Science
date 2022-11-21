"""This is A K means clustering project"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

data = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1.8, random_state=101)
# plt.scatter(data[0][:, 0],data[0][:,1],c=data[1],cmap='rainbow')
# plt.title('Make Blobs Clusters')
# plt.show()
"""Now I create my clusters"""
from sklearn.cluster import KMeans

k_cluster = KMeans(n_clusters=4)
clusters = k_cluster.fit(data)

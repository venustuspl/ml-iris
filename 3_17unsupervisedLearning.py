import numpy as np
from matplotlib import pyplot as plt
#%matplotlib inline

from sklearn.cluster import KMeans
from sklearn.datasets._samples_generator import make_blobs

X, y = make_blobs(n_samples=100, centers=4, cluster_std=0.60, random_state=0)

plt.scatter(X[:, 0], X[:, 1])

WCSS = []

for k in range(1, 15):
    kmeans = KMeans(n_clusters=k, n_init='auto')
    kmeans.fit(X)
    WCSS.append(kmeans.inertia_)

plt.plot(range(1, 15), WCSS)
plt.xlabel("Number of K Value(Cluster)")
plt.ylabel("WCSS")
plt.grid()
plt.show()

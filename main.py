from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt 
from k_means import KMeans

centroids = [(-5, -5), (1, 4), (-2, 3), (4, -2)]
cluster_std = [1, 1, 1, 1]

X, y = make_blobs(n_samples = 100, cluster_std=cluster_std, centers=centroids,
                   n_features= 2, random_state = 2)

# plt.scatter(X[:, 0], X[:, 1])

km = KMeans(n_clusters=4, max_iter=100)
y_means = km.fit(X)

plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], color = 'yellow')
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], color = 'green')
plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1], color = 'red')
plt.scatter(X[y_means == 3, 0], X[y_means == 3, 1], color = 'blue')

# plt.scatter(X[:, 0], X[:, 1])
plt.show()
# print(X)
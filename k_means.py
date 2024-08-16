import random
import numpy as np
import math
class KMeans:
    def __init__(self, n_clusters=3, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def fit(self, X):
        # Initialize centroids randomly
        random_index = random.sample(range(0, X.shape[0]), self.n_clusters)
        self.centroids = X[random_index]
        
        for i in range(self.max_iter):
            # Assign each data point to the closest centroid
            cluster_group = self.assign_cluster(X)
            # print(cluster_group)
            old_centroids = self.centroids
            # calculating new centroids
            # move the centroid
            self.centroids = self.move_centroids(X, cluster_group)
            # check finish 
            if (old_centroids == self.centroids).all():
                break

        return cluster_group  

    def assign_cluster(self, X):
        cluster_group = []
        distance = []

        for row in X:
            for centroids in self.centroids:
                distance.append(np.sqrt(np.dot(row-centroids, row-centroids)))
            min_distance = min(distance)
            # print(min_distance)
            index_pos = distance.index(min_distance)
            cluster_group.append(index_pos)
            # print(index_pos)
            distance.clear()

        return np.array(cluster_group)
    
    def move_centroids(self, X, cluster_group):
        new_centroids = []
        cluster_type =  np.unique(cluster_group) # type = 0, 1

        for type in cluster_type:
             new_centroids.append(X[cluster_group == type].mean(axis= 0))

        return np.array(new_centroids)
      
# Euclidean Distance = (Xo - Xc) + (Yo - Yc)


# (2 , 5) randomly generated centroid
#  ( 1, 2 ) new centroid 
# (2, 0) new centroid 
# ( 2, 0 ) same new centroid

# (1, 2) (4, 9) = [ 0, 1, 0, 0, 1, 2, 3, 3, 2] [ a, b, a, d] 

# np.unique() = 1, 0, 2, 3

# [ 0, 0, 0 ] , [1, 1, 1] positions 
# [ 0.48368310811401793, 0.19160075381080863, 0.17153874554036228 ] group of 0 position
# [ 1.7776103506004959, 1.17153874554036228, 1.8775088639520303 ] group of 1 position

# (2.0, 3.5)

import numpy as np

class KMean():
	def __init__(self, n_clusters):
		self.n_clusters = n_clusters
		self.centroids = None
		self.max_iter = 10

	def fit(self, data):
        # Initialize the centroids randomly.
		self.centroids = np.random.randint(0, 255, (self.n_clusters, data.shape[-1]))

        # Repeat until convergence or max_iter times:
		for i in range(self.max_iter):
			# Assign each data point to the closest centroid.
			distances = np.linalg.norm(data - self.centroids[:, np.newaxis], axis=2)
			labels = np.argmin(distances, axis=0)

            # Update the centroids.
			for i in range(self.n_clusters):
				if np.any(labels == i):
 					new_centroid = np.mean(data[labels == i], axis=0)
				self.centroids[i] = new_centroid

            # If the centroids have not changed, then we have converged.
			if np.allclose(new_centroid, self.centroids):
				break

	def predict(self, data):
		distances = np.linalg.norm(data - self.centroids[:, np.newaxis], axis=2)
		return np.argmin(distances, axis=0)
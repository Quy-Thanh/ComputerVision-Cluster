import numpy as np

class KMean():
	def __init__(self, n_clusters):
		self.n_clusters = n_clusters
		self.centroids = None

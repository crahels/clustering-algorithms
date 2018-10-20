import numpy as np
from scipy.spatial.distance import euclidean

class DBSCAN:

	def __init__(self, eps=0.5, min_points=5):
		self.eps = eps
		self.min_points = min_points

	def fit_predict(self, X, y=None):
		self.labels = np.full(len(X), -2)
		c = -1
		for p in range(len(X)):
			if not self.labels[p] == -2:
				continue
			neighbor_points = self.region_query(X, p)
			if len(neighbor_points) < self.min_points:
				self.labels[p] = -1
			else:
				c += 1
				self.expand_cluster(X, p, neighbor_points, c)
		return self.labels

	def expand_cluster(self, X, p, neighbor_points, c):
		self.labels[p] = c
		i = 0
		while i < len(neighbor_points):
			q = neighbor_points[i]
			if self.labels[q] == -1:
				self.labels[q] = c
			elif self.labels[q] == -2:
				self.labels[q] = c
				q_neighbor_points = self.region_query(X, q)
				if len(q_neighbor_points) >= self.min_points:
					neighbor_points = neighbor_points + q_neighbor_points
			i += 1

	def region_query(self, X, p):
		neighbor_points = []
		for q in range(len(X)):
			if euclidean(X[p], X[q]) < self.eps:
				neighbor_points.append(q)
		return neighbor_points

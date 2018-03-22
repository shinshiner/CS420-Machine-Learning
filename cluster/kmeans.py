from sklearn.cluster import KMeans
from dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from copy import deepcopy
from pyCompatible import *

pyPatch()

class kmeans(object):
	def __init__(self, n_clusters = 1, verbose = 2):
		self.model = KMeans(
			n_clusters = n_clusters,
			verbose = verbose)
		self.n_clusters = n_clusters
		self.dataset = Dataset(class_num = 4)
		self.dataset.generate()
		self.data = self.dataset.data

	def train(self):
		self.model.fit(self.data)

	def select(self):
		scores = []
		low = 99999
		for n in xrange(1, self.n_clusters + 1):
			kmean = KMeans(n_clusters = n)
			kmean.fit(self.data)
			scores.append(-kmean.score(self.data))
			if scores[-1] < low:
				low = scores[-1]
				self.model = deepcopy(kmean)
		print '------scores-------\n', scores
		print 'selected clusters:', scores.index(low) + 1, '\n'

	def show(self):
		plt.figure()
		x = np.linspace(-0.5, 3, 10)
		y = np.linspace(0.3, 3, 10)
		X, Y = np.meshgrid(x, y)
		#Z = (2-self.model.score_samples(np.array([X.ravel(), Y.ravel()]).T)).reshape(X.shape)
		labels = self.model.predict(self.data)

		#plt.contour(X, Y, Z, norm=LogNorm(vmin=0.1, vmax=1000.0), levels=np.logspace(0, 2, 15))
		plt.scatter(self.data[:, 0], self.data[:, 1], c = labels, s = 15)

		plt.show()
from sklearn.mixture import GaussianMixture
from dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

class GMM(object):
	def __init__(self, n_components = 1, verbose = 2, verbose_interval = 1):
		self.model = GaussianMixture(
			n_components = n_components,
			verbose = verbose,
			verbose_interval = verbose_interval)
		self.dataset = Dataset()
		self.dataset.generate()
		self.data = self.dataset.data

	def train(self):
		self.model.fit(self.data)

	def show(self):
		plt.figure()
		x = np.linspace(-0.5, 3, 10)
		y = np.linspace(0.3, 3, 10)
		X, Y = np.meshgrid(x, y)
		Z = (1-self.model.score_samples(np.array([X.ravel(), Y.ravel()]).T)).reshape(X.shape)

		plt.contour(X, Y, Z, norm=LogNorm(vmin=0.1, vmax=1000.0), levels=np.logspace(0, 2, 15))
		plt.scatter(self.data[:, 0], self.data[:, 1], s = 15)

		plt.show()

if __name__ == '__main__':
	gmm = GMM(2)
	gmm.train()
	gmm.show()
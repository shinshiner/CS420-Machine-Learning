from sklearn.mixture import GaussianMixture
from dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from copy import deepcopy
from pyCompatible import *

pyPatch()

class GMM_EM(object):
    def __init__(self, n_components = 1, verbose = 2, verbose_interval = 1,
                 Data = None):
        self.model = GaussianMixture(
            n_components = n_components,
            verbose = verbose,
            verbose_interval = verbose_interval)
        self.n_components = n_components
        if Data == None:
            self.dataset = Dataset()
            self.dataset.generate()
        else:
            self.dataset = Data
        self.data = self.dataset.data
        self.aic = []
        self.bic = []
        self.aic_b = None

    def train(self):
        self.model.fit(self.data)

    def aic_select(self):
        self.aic_b = True
        low = 99999
        for n in range(1, self.n_components + 1):
            gmm = GaussianMixture(n_components = n)
            gmm.fit(self.data)
            self.aic.append(gmm.aic(self.data))
            if self.aic[-1] < low:
                low = self.aic[-1]
                self.model = deepcopy(gmm)
        print('------aic-------\n', self.aic)
        self.res_n = self.aic.index(low) + 1
        print('selected components:', self.res_n, '\n')

    def bic_select(self):
        self.aic_b = False
        low = 99999
        for n in range(1, self.n_components + 1):
            gmm = GaussianMixture(n_components = n)
            gmm.fit(self.data)
            self.bic.append(gmm.bic(self.data))
            if self.bic[-1] < low:
                low = self.bic[-1]
                self.model = deepcopy(gmm)
        print('------bic-------\n', self.bic)
        self.res_n = self.bic.index(low) + 1
        print('selected components:', self.res_n, '\n')

    def show(self, n = None):
        plt.figure()
        labels = self.model.predict(self.data)
        plt.scatter(self.data[:, 0], self.data[:, 1], c = labels, s = 15)

        if n == None:
            plt.show()
        else:
            if self.aic_b:
                plt.savefig('report/demo/aic_%d_%d' % (n, self.res_n))
            else:
                plt.savefig('report/demo/bic_%d_%d' % (n, self.res_n))

    def show_dis(self, dis = None):
        plt.figure()
        labels = self.model.predict(self.data)

        plt.scatter(self.data[:, 0], self.data[:, 1], c = labels, s = 15)

        if dis == None:
            plt.show()
        else:
            if self.aic_b:
                plt.savefig('report/demo/dis_aic_%d_%d' % (dis, self.res_n))
            else:
                plt.savefig('report/demo/dis_bic_%d_%d' % (dis, self.res_n))

if __name__ == '__main__':
    gmm = GMM_EM(n_components = 4)
    gmm.train()
    gmm.show()
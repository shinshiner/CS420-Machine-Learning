from sklearn.cluster import KMeans
from dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import random
from math import exp

def dis(cor1, cor2):
    return (cor1[0] - cor2[0])**2 + (cor1[1] - cor2[1])**2

class Center(object):
    def __init__(self, x = 0, y = 0):
        self.x = x
        self.y = y
        self.data = None
        self.cnt = 0

class kmeans(object):
    def __init__(self, n_clusters = 1, verbose = 2):
        self.model = KMeans(
            n_clusters = n_clusters,
            verbose = verbose)
        self.n_clusters = n_clusters
        self.dataset = Dataset(class_num = 3)
        self.dataset.generate()
        self.data = self.dataset.data
        self.centers = []
        self.rho = []

    def train(self):
        self.model.fit(self.data)

    def select(self):
        scores = []
        low = 99999
        for n in range(1, self.n_clusters + 1):
            kmean = KMeans(n_clusters = n)
            kmean.fit(self.data)
            scores.append(-kmean.score(self.data))
            if scores[-1] < low:
                low = scores[-1]
                self.model = deepcopy(kmean)
        print('------scores-------\n', scores)
        print('selected clusters:', scores.index(low) + 1, '\n')

    def get_density(self):
        b = 0.0
        for i in self.data:
            b += dis(self.data[0], i)

        for i in self.data:
            v = 0.0
            for j in self.data:
                v += dis(i, j) / b
            self.rho.append(exp(-v))

    def RPCL(self):
        # init
        self.get_density()
        self.random_center()
        self.show('report/demo/RPCL_1')
        alpha = 0.05
        beta = 0.05

        # training weight vectors
        while True:
            for idx, i in enumerate(self.data):
                distances = []
                for c in self.centers:
                    distances.append(dis(i, (c.x, c.y)))

                winner_idx = distances.index(max(distances))
                delta_x = alpha * self.rho[idx] * (i[0] - self.centers[winner_idx].x)
                delta_y = alpha * self.rho[idx] * (i[1] - self.centers[winner_idx].y)
                self.centers[winner_idx].x += delta_x
                self.centers[winner_idx].y += delta_y
                distances[winner_idx] = -1

                rival_idx = distances.index(max(distances))
                delta_x = beta * self.rho[idx] * (i[0] - self.centers[rival_idx].x)
                delta_y = beta * self.rho[idx] * (i[1] - self.centers[rival_idx].y)
                self.centers[rival_idx].x -= delta_x
                self.centers[rival_idx].y -= delta_y

            if (delta_x + delta_y) < 0.01:
                break

        # assign each data
        for i in self.data:
            tmp = 999
            res = None
            for idx, c in enumerate(self.centers):
                if dis(i, (c.x, c.y)) < tmp:
                    tmp = dis(i, (c.x, c.y))
                    res = idx
            self.centers[res].cnt += 1

        # remove extra centers
        eta = 100
        kill_list = []
        for idx, c in enumerate(self.centers):
            print(c.cnt)
            if c.cnt < eta:
                kill_list.append(idx)

        for i in kill_list:
            self.centers[i] = None

        self.show('report/demo/RPCL_2')

    def random_center(self):
        for i in range(8):
            self.centers.append(Center(x=np.random.normal(0.5, 0.2),
                                       y=np.random.normal(0.5, 0.2)))

    def show(self, filename = None):
        plt.figure()
        x = np.linspace(-0.5, 3, 10)
        y = np.linspace(0.3, 3, 10)
        X, Y = np.meshgrid(x, y)
        #Z = (2-self.model.score_samples(np.array([X.ravel(), Y.ravel()]).T)).reshape(X.shape)
        #labels = self.model.predict(self.data)

        #plt.contour(X, Y, Z, norm=LogNorm(vmin=0.1, vmax=1000.0), levels=np.logspace(0, 2, 15))
        #plt.scatter(self.data[:, 0], self.data[:, 1], c = labels, s = 15)
        plt.scatter(self.data[:, 0], self.data[:, 1], s=15)
        for c in self.centers:
            if c != None:
                plt.scatter(c.x, c.y, c='red', s=25)

        if filename != None:
            plt.savefig(filename)
        plt.show()

if __name__ == '__main__':
    km = kmeans()
    km.RPCL()
    # km.random_center()
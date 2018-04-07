import numpy as np
import random
import matplotlib.pyplot as plt
from pyCompatible import *

pyPatch()

# seeds can be used: 4, 5
class Dataset(object):
    def __init__(self, class_num = 2, data_num = 200, seed = 4, center=None):
        self.class_num = class_num
        self.data_num = data_num
        self.data = np.zeros((data_num * class_num, 2), dtype = np.float32)
        if center == None:
            self.centers = [(0, 0), (0, 1), (1, 1), (1, 0)]
            # self.centers = [(0, 1), (0, 0.9), (0, 1.1), (0.1, 1.1)]
        else:
            self.centers = center
        self.colors = ['red', 'blue']

        random.seed(seed)

    def generate(self):
        for k in range(self.class_num):
            # self.data[:, 2] = k
            mu_x = self.centers[k][0]# + random.random()
            mu_y = self.centers[k][1]# + random.random()
            sigma = random.random() * 0.5
            for i in range(self.data_num):
                self.data[k * self.data_num + i][0] = np.random.normal(mu_x, sigma)
                self.data[k * self.data_num + i][1] = np.random.normal(mu_y, sigma)

    def show(self):
        plt.figure()
        for i in range(self.class_num):
            x = self.data[i * self.data_num: (i + 1) * self.data_num - 1, 0]
            y = self.data[i * self.data_num: (i + 1) * self.data_num - 1, 1]
            plt.scatter(x, y, s = 10, c = self.colors[i])
        plt.show()

def debug():
    arr = np.arange(12).reshape(6, 2)
    print(arr)
    print(arr[:, 0])

if __name__ == '__main__':
    dataset = Dataset()
    dataset.generate()
    dataset.show()
    # debug()
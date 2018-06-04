from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt

interval = 100
max_iter = 100

def pca(x):
    model = PCA(n_components=9)
    data = model.fit_transform(x)
    print(model.explained_variance_ratio_.sum())
    return data

def cifar_sample():
    with open('data/cifar-10-batches-py/test_batch', 'rb') as f:
        dic = pickle.load(f, encoding='bytes')
        labels = np.array(dic[b'labels'])
        print(dic[b'data'].shape)
        np.save('data/cifar-10-batches-py/cifar10-data.t.npy', dic[b'data'])
        np.save('data/cifar-10-batches-py/cifar10-labels.t.npy', labels)

def merge_batches():
    labels = []
    data = np.zeros((50000, 3072))
    for i in range(1, 6):
        with open('data/cifar-10-batches-py/data_batch_%d' % i, 'rb') as f:
            dic = pickle.load(f, encoding='bytes')
            labels += dic[b'labels']
            data[10000*(i-1):10000*i] = dic[b'data']

    labels = np.array(labels)
    #print(labels.shape, data.shape)
    np.save('data/cifar-10-batches-py/cifar10-data.npy', data)
    np.save('data/cifar-10-batches-py/cifar10-labels.npy', labels)


def svm_bonus():
    x_tr = np.load('data/cifar-10-batches-py/cifar10-data.npy')[:10000]
    y_tr = np.load('data/cifar-10-batches-py/cifar10-labels.npy')[:10000]
    x_t = np.load('data/cifar-10-batches-py/cifar10-data.t.npy')
    y_t = np.load('data/cifar-10-batches-py/cifar10-labels.t.npy')

    # scaler = StandardScaler()
    # scaler.fit(x_tr)
    # x_tr = scaler.transform(x_tr)
    # scaler.fit(x_t)
    # x_t = scaler.transform(x_t)

    res_tr = []
    res_t = []
    print('start to training')
    t = time.time()
    for i in range(interval, max_iter + 1, interval):
        model = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                    decision_function_shape='ovr', degree=3, gamma=0.000001, kernel='rbf',
                    max_iter=i, probability=False, random_state=None, shrinking=True,
                    tol=0.001, verbose=False)
        model.fit(x_tr, y_tr)
        print('finish training, spending %.4f seconds' % (time.time() - t))

        res_tr.append(round(model.score(x_tr, y_tr), 3))
        res_t.append(round(model.score(x_t, y_t), 3))
        # print(model.score(x_tr, y_tr))
        # print(model.score(x_t, y_t))
    print('train: ', res_tr)
    print('test: ', res_t)

def plot_poly_para():
    x = list(range(2, 9))
    y_tr = [0.259, 0.223, 0.200, 0.198, 0.218, 0.212, 0.176]
    y_t = [0.232, 0.209, 0.186, 0.187, 0.201, 0.189, 0.16]

    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.plot(x, y_tr, color='#9999ff', linewidth=1.7, label='Training set')
    ax.plot(x, y_t, color='#ffa07a', linewidth=1.7, label='Testing set')
    ax.scatter(x, y_tr, s=13, c='#9999ff')
    ax.scatter(x, y_t, s=13, c='#ffa07a')
    ax.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('report/img/svm_poly_para')
    plt.show()

def plot_penalty():
    x = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1.0, 3.0, 10.0]
    y_tr = [0.331, 0.328, 0.337, 0.344, 0.343, 0.356, 0.344, 0.201, 0.201]
    y_t = [0.241, 0.235, 0.241, 0.248, 0.243, 0.255, 0.242, 0.118, 0.118]

    x_ax = np.arange(9) * 0.9
    total_width, n = 0.75, 2
    width = total_width / n
    x_ax = x_ax - (total_width - width) / 2

    plt.bar(x_ax, y_tr, width=width, facecolor='#9999ff', edgecolor='white', label='Training set')
    plt.bar(x_ax + width, y_t, width=width, facecolor='#ffa07a', edgecolor='white', label='Testing set')
    for x, y1, y2 in zip(x_ax, y_tr, y_t):
        plt.text(x - 0.02, y1, '%.2f' % y1, ha='center', va='bottom')
        plt.text(x + width + 0.075, y2, '%.2f' % y2, ha='center', va='bottom')

    ax = plt.gca()
    ax.set_xticks(x_ax + width / 2)
    ax.set_xticklabels((0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1.0, 3.0, 10.0))
    plt.xlabel('Penalty parameter')
    plt.ylabel('Accuracy')
    #plt.ylim(0, 1.245)
    plt.legend()
    plt.savefig('report/img/para_penalty')
    plt.show()

if __name__ == '__main__':
    svm_bonus()
    #plot_penalty()
    #cifar_sample()
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pickle
import time

interval = 100
max_iter = 100

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
                    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
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

if __name__ == '__main__':
    svm_bonus()
    #cifar_sample()
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt

interval = 1000
max_iter = 1000

# zero mean
def zeroMean(dataMat):
    meanVal = np.mean(dataMat,axis=0)     # get the mean value of each feature
    newData = dataMat - meanVal
    return newData, meanVal

def percentage2n(eigVals,percentage):
    sortArray = np.sort(eigVals)   # ascending order
    sortArray = sortArray[-1::-1]  # descending order
    arraySum = sum(sortArray)
    tmpSum = 0
    num = 0
    for i in sortArray:
        tmpSum += i
        num += 1
        if tmpSum >= arraySum * percentage:
            return num

def pca(dataMat,percentage=0.99):
    dataMat,meanVal=zeroMean(dataMat)
    # print "datamat type :" + str(type(dataMat))

    print ("Now computing covariance matrix...")
    covMat=np.cov(dataMat,rowvar=0)    # solve covariance matrix
    # print "covmat type :" + str(type(covMat))

    print ("Finished. Now solve eigen values and vectors...")
    eigVals,eigVects=np.linalg.eig(np.mat(covMat))# solve eigen vectors & values
    # print "eigVals type :" + str(type(eigVals))
    # print "eigVects type :" + str(type(eigVects))

    print ("Finished. Now select eigen vectors...")
    n=percentage2n(eigVals,percentage)
    eigValIndice=np.argsort(eigVals)            # sorting according to eigen values
    # print "eigValIndice type :" + str(type(eigValIndice))

    n_eigValIndice=eigValIndice[-1:-(n+1):-1]   # index of max_n features
    n_eigVect=eigVects[:,n_eigValIndice]        # eigen vectors
    print ("Finished. Now generating new data...")
    # print "n_eigVect type :" + str(type(n_eigVect))

    lowDDataMat=dataMat*n_eigVect               # data in low dim
    # reconMat=(lowDDataMat*n_eigVect.T)+meanVal  # reconstruct data
    return np.array(lowDDataMat)

def skpca(x, i):
    model = PCA(n_components=i)
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
	# load data
    x_tr = np.load('data/cifar-10-batches-py/cifar10-data.npy')
    y_tr = np.load('data/cifar-10-batches-py/cifar10-labels.npy')
    x_t = np.load('data/cifar-10-batches-py/cifar10-data.t.npy')
    y_t = np.load('data/cifar-10-batches-py/cifar10-labels.t.npy')

    # preprocessing
    # scaler = StandardScaler()
    # scaler.fit(x_tr)
    # x_tr = scaler.transform(x_tr)
    # scaler.fit(x_t)
    # x_t = scaler.transform(x_t)
    
    x_tr = pca(x_tr, 0.9)
    x_t = skpca(x_t, x_tr.shape[1])

    # training stage
    res_tr = []
    res_t = []
    print('start to training')
    t = time.time()
    tmpl = [10.0]
    #tmpl = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1.0, 3.0, 10.0]
    for tmp in tmpl:
        for i in range(interval, max_iter + 1, interval):
            model = SVC(C=tmp, cache_size=200, class_weight=None, coef0=0.0,
                        decision_function_shape='ovr', degree=3, gamma=0.0000003, kernel='rbf',
                        max_iter=-1, probability=False, random_state=666, shrinking=True,
                        tol=0.001, verbose=False)
            model.fit(x_tr, y_tr)
            print('finish training, spending %.4f seconds' % (time.time() - t))

            res_tr.append(round(model.score(x_tr, y_tr), 3))
            res_t.append(round(model.score(x_t, y_t), 6))
            # print(model.score(x_tr, y_tr))
            # print(model.score(x_t, y_t))
    print('train: ', res_tr)
    print('test: ', res_t)

# ploting
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

# ploting
def plot_penalty():
    x = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1.0, 3.0, 10.0, 11.0, 12.0]
    y_tr = [0.328, 0.34, 0.392, 0.41, 0.473, 0.473, 0.464, 0.48, 0.484, 0.484, 0.484]
    y_t = [0.276, 0.283, 0.316, 0.318, 0.347, 0.334, 0.33, 0.336, 0.338, 0.338, 0.338]

    x_ax = np.arange(11) * 0.9
    total_width, n = 0.75, 2
    width = total_width / n
    x_ax = x_ax - (total_width - width) / 2

    plt.figure(figsize=(10, 8))
    plt.bar(x_ax, y_tr, width=width, facecolor='#9999ff', edgecolor='white', label='Training set')
    plt.bar(x_ax + width, y_t, width=width, facecolor='#ffa07a', edgecolor='white', label='Testing set')
    for x, y1, y2 in zip(x_ax, y_tr, y_t):
        plt.text(x - 0.02, y1, '%.2f' % y1, ha='center', va='bottom')
        plt.text(x + width + 0.03, y2, '%.2f' % y2, ha='center', va='bottom')

    ax = plt.gca()
    ax.set_xticks(x_ax + width / 2)
    ax.set_xticklabels((0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1.0, 3.0, 10.0, 11.0, 12.0))
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
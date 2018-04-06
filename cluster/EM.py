from GMM import GMM_EM
from VBEM import VBEM
from kmeans import kmeans
from dataset import Dataset
import matplotlib.pyplot as plt

def testGMM(data, n, aic=True):
    model = GMM_EM(n_components = 10, Data=data)
    if aic:
        print('------------test GMM with aic selection------------')
        model.aic_select()
    else:
        print('------------test GMM with bic selection------------')
        model.bic_select()
    model.show(n)

def testVBEM(data, n):
    print('------------test VBEM------------')
    model = VBEM(n_components = 10, Data=data)
    model.select()
    model.show(n)

def testKmeans():
    print('------------test kmeans------------')
    model = kmeans(n_clusters = 10)
    model.select()
    model.show()

def sample_size():
    samples = [5, 10, 30, 50, 100]
    for n in samples:
        data = Dataset(class_num=4, data_num=n)
        data.generate()
        testGMM(data, n)
        testGMM(data, n, False)
        testVBEM(data, n)

def sample_size_l():
    x = [5, 10, 30, 50, 100]
    y_bic = [10, 4, 4, 4, 4]
    y_aic = [10, 10, 4, 4, 4]
    y_vbem = [4, 4, 4, 4, 4]

    plt.figure()
    plt.xlabel('Sample Sizes')
    plt.xticks([5, 10, 30, 50, 100])
    plt.ylabel('The Number of Clutsets')
    plt.ylim((0, 12))

    point_size = 20
    plt.scatter(x, y_bic, s=point_size)
    plt.scatter(x, y_aic, s=point_size)
    plt.scatter(x, y_vbem, s=point_size)
    ax1 = plt.plot(x, y_bic, label='BIC', linestyle='--')
    ax2 = plt.plot(x, y_aic, label='AIC', linestyle='--')
    ax3 = plt.plot(x, y_vbem, label='VBEM', linestyle='--')

    plt.legend()
    plt.savefig('report/demo/sample_line')
    plt.show()

if __name__ == '__main__':
    # sample_size()
    sample_size_l()
    pass
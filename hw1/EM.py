from GMM import GMM_EM
from VBEM import VBEM
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
    model.show_dis(n)

def testVBEM(data, n):
    print('------------test VBEM------------')
    model = VBEM(n_components = 10, Data=data)
    model.show_dis(n)

def sample_size():
    '''
    sample size experiment
    :return: None
    '''
    samples = [5, 10, 30, 50, 100]
    for n in samples:
        data = Dataset(class_num=4, data_num=n)
        data.generate()
        testGMM(data, n)
        testGMM(data, n, False)
        testVBEM(data, n)

def sample_size_l():
    '''
    sample size experiment line chart
    :return: None
    '''
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

def cluster_distance():
    '''
    cluster distance experiment
    :return: None
    '''
    dis = [0.3, 0.7, 1.0, 3.0]
    centers = [[(0, 0), (dis[0], 0), (dis[0] / 2, 0.866 * dis[0])],
               [(0, 0), (dis[1], 0), (dis[1] / 2, 0.866 * dis[1])],
               [(0, 0), (dis[2], 0), (dis[2] / 2, 0.866 * dis[2])],
               [(0, 0), (dis[3], 0), (dis[3] / 2, 0.866 * dis[3])]]
    for i, center in enumerate(centers):
        data = Dataset(class_num=3, data_num=50, center=center)
        data.generate()
        testGMM(data, int(dis[i] * 10))
        testGMM(data, int(dis[i] * 10), False)
        testVBEM(data, int(dis[i] * 10))

if __name__ == '__main__':
    # sample_size()
    # sample_size_l()
    # cluster_distance()
    pass
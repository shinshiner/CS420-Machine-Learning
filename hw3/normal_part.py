from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, chi2, f_regression

import numpy as np
interval = 15
max_iter = 150

def svm(data_name):
    x_tr = np.load('data/' + data_name + '_feature.npy')
    y_tr = np.load('data/' + data_name + '_target.npy')
    x_t = np.load('data/' + data_name + '.t_feature.npy')
    y_t = np.load('data/' + data_name + '.t_target.npy')
    if data_name == 'madelon':
        scaler = StandardScaler()
        scaler.fit(x_tr)
        x_tr = scaler.transform(x_tr)
        scaler.fit(x_t)
        x_t = scaler.transform(x_t)

    res_tr = []
    res_t = []
    for i in range(interval, max_iter + 1, interval):
        model = SVC(C=0.6, cache_size=200, class_weight=None, coef0=0.5,
                decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
                max_iter=i, probability=False, random_state=None, shrinking=True,
                tol=0.001, verbose=False)
        model.fit(x_tr, y_tr)

        res_tr.append(round(model.score(x_tr, y_tr), 3))
        res_t.append(round(model.score(x_t, y_t), 3))
        # print(model.score(x_tr, y_tr))
        # print(model.score(x_t, y_t))
        # print(model.predict(x_tr))
    print('train: ', res_tr)
    print('test: ', res_t)

def pca(x):
    model = PCA(n_components=5)
    data = model.fit_transform(x)
    print(model.explained_variance_ratio_.sum())
    return data

def select(x, y):
    selector = SelectKBest(score_func=f_classif, k=10)
    real_features = selector.fit_transform(x, y)
    #print(selector.scores_)

    return real_features

def mlp(data_name):
    x_tr = np.load('data/' + data_name + '_feature.npy')
    y_tr = np.load('data/' + data_name + '_target.npy')
    x_t = np.load('data/' + data_name + '.t_feature.npy')
    y_t = np.load('data/' + data_name + '.t_target.npy')
    if data_name == 'madelon':
        x_tr = select(x_tr, y_tr)
        x_t = select(x_t, y_t)

        # scaler = StandardScaler()
        # scaler.fit(x_tr)
        # x_tr = scaler.transform(x_tr)
        # scaler.fit(x_t)
        # x_t = scaler.transform(x_t)

    res_tr = []
    res_t = []
    for i in range(interval, max_iter + 1, interval):
        model = MLPClassifier(solver='adam', alpha=1e-3,
                    learning_rate_init=0.001, max_iter=i,
                    activation='tanh',
                    hidden_layer_sizes=(10, 10, 2), random_state=666)
        model.fit(x_tr, y_tr)

        res_tr.append(round(model.score(x_tr, y_tr), 3))
        res_t.append(round(model.score(x_t, y_t), 3))
        # print(model.score(x_tr, y_tr))
        # print(model.score(x_t, y_t))
    print('train: ', res_tr)
    print('test: ', res_t)

def plot_lr_splice():
    x = list(range(interval, max_iter + 1, interval))
    y_tr_relu = [0.698, 0.764, 0.813, 0.845, 0.865, 0.884, 0.906, 0.921, 0.921, 0.921]
    y_tr_logi = [0.517, 0.517, 0.721, 0.799, 0.819, 0.834, 0.845, 0.852, 0.857, 0.861]
    y_tr_tanh = [0.65, 0.732, 0.793, 0.832, 0.863, 0.881, 0.905, 0.915, 0.929, 0.944]
    y_t_relu = [0.68, 0.759, 0.822, 0.85, 0.857, 0.867, 0.874, 0.883, 0.883, 0.883]
    y_t_logi = [0.52, 0.52, 0.717, 0.835, 0.832, 0.827, 0.83, 0.834, 0.836, 0.839]
    y_t_tanh = [0.646, 0.721, 0.777, 0.805, 0.817, 0.822, 0.832, 0.836, 0.84, 0.84]

if __name__ == '__main__':
    # mlp('satimage.scale')
    mlp('splice')
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, chi2, f_regression

import numpy as np

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

    model = SVC(C=0.6, cache_size=200, class_weight=None, coef0=0.5,
            decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
            max_iter=1000, probability=False, random_state=None, shrinking=True,
            tol=0.001, verbose=False)
    model.fit(x_tr, y_tr)

    print(model.score(x_tr, y_tr))
    print(model.score(x_t, y_t))
    # print(model.predict(x_tr))

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

    model = MLPClassifier(solver='adam', alpha=1e-3,
                learning_rate_init=0.001, max_iter=200,
                activation='relu',
                hidden_layer_sizes=(10, 10, 2), random_state=666)
    model.fit(x_tr, y_tr)

    print(model.score(x_tr, y_tr))
    print(model.score(x_t, y_t))

if __name__ == '__main__':
    # mlp('satimage.scale')
    mlp('splice')
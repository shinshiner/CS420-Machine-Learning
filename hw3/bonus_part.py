from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

def svm_bonus():
    x_tr = np.load('data/convex/convex_train.amat_feature.npy')
    y_tr = np.load('data/convex/convex_train.amat_target.npy')
    x_t = np.load('data/convex/convex_test.amat_feature.npy')[27000:33000]
    y_t = np.load('data/convex/convex_test.amat_target.npy')[27000:33000]

    scaler = StandardScaler()
    scaler.fit(x_tr)
    x_tr = scaler.transform(x_tr)
    scaler.fit(x_t)
    x_t = scaler.transform(x_t)

    model = SVC(C=1.0, cache_size=200, class_weight=None, coef0=1.0,
                decision_function_shape='ovr', degree=5, gamma='auto', kernel='poly',
                max_iter=500, probability=False, random_state=None, shrinking=True,
                tol=0.001, verbose=False)
    model.fit(x_tr, y_tr)

    print(model.score(x_tr, y_tr))
    print(model.score(x_t, y_t))

if __name__ == '__main__':
    svm_bonus()
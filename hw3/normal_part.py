from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, chi2, f_regression

import numpy as np
import matplotlib.pyplot as plt

interval = 15
max_iter = 150

def pca(x):
    model = PCA(n_components=27)
    data = model.fit_transform(x)
    print(model.explained_variance_ratio_.sum())
    #return data
    return x

def select(x, y):
    selector = SelectKBest(score_func=f_classif, k=10)
    real_features = selector.fit_transform(x, y)
    #print(selector.scores_)

    return real_features

################################### SVM Part #####################################

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

################################### SVM Part #####################################




################################### MLP Part #####################################

def mlp(data_name):
    x_tr = np.load('data/' + data_name + '_feature.npy')
    y_tr = np.load('data/' + data_name + '_target.npy')
    x_t = np.load('data/' + data_name + '.t_feature.npy')
    y_t = np.load('data/' + data_name + '.t_target.npy')
    # if data_name == 'madelon':
    #     x_tr = select(x_tr, y_tr)
    #     x_t = select(x_t, y_t)

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
                    activation='relu',
                    hidden_layer_sizes=(10, 10, 2), random_state=666)
        model.fit(x_tr, y_tr)

        res_tr.append(round(model.score(x_tr, y_tr), 3))
        res_t.append(round(model.score(x_t, y_t), 3))
        # print(model.score(x_tr, y_tr))
        # print(model.score(x_t, y_t))
    print('train: ', res_tr)
    print('test: ', res_t)

def plot_activation_splice():
    # model = MLPClassifier(solver='adam', alpha=1e-3,
    #                       learning_rate_init=0.001, max_iter=i,
    #                       activation='relu',
    #                       hidden_layer_sizes=(10, 10, 2), random_state=666)
    x = list(range(interval, max_iter + 1, interval))
    y_tr_relu = [0.698, 0.764, 0.813, 0.845, 0.865, 0.884, 0.906, 0.921, 0.921, 0.921]
    y_tr_logi = [0.517, 0.517, 0.721, 0.799, 0.819, 0.834, 0.845, 0.852, 0.857, 0.861]
    y_tr_tanh = [0.65, 0.732, 0.793, 0.832, 0.863, 0.881, 0.905, 0.915, 0.929, 0.944]
    y_t_relu = [0.68, 0.759, 0.822, 0.85, 0.857, 0.867, 0.874, 0.883, 0.883, 0.883]
    y_t_logi = [0.52, 0.52, 0.717, 0.835, 0.832, 0.827, 0.83, 0.834, 0.836, 0.839]
    y_t_tanh = [0.646, 0.721, 0.777, 0.805, 0.817, 0.822, 0.832, 0.836, 0.84, 0.84]

    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.plot(x, y_tr_relu, color='#90EE90', linewidth=1.7, label='relu')
    ax.plot(x, y_tr_logi, color='#ffa07a', linewidth=1.7, label='sigmoid')
    ax.plot(x, y_tr_tanh, color='#9999ff', linewidth=1.7, label='tanh')
    ax.scatter(x, y_tr_relu, s=13, c='#90EE90')
    ax.scatter(x, y_tr_logi, s=13, c='#ffa07a')
    ax.scatter(x, y_tr_tanh, s=13, c='#9999ff')
    ax.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('report/img/mlp_activation_splice_tr')
    plt.show()

    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.plot(x, y_t_relu, color='#90EE90', linewidth=1.7, label='relu')
    ax.plot(x, y_t_logi, color='#ffa07a', linewidth=1.7, label='sigmoid')
    ax.plot(x, y_t_tanh, color='#9999ff', linewidth=1.7, label='tanh')
    ax.scatter(x, y_t_relu, s=13, c='#90EE90')
    ax.scatter(x, y_t_logi, s=13, c='#ffa07a')
    ax.scatter(x, y_t_tanh, s=13, c='#9999ff')
    ax.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('report/img/mlp_activation_splice_t')
    plt.show()

def plot_activation_sat():
    # model = MLPClassifier(solver='adam', alpha=1e-3,
    #                       learning_rate_init=0.001, max_iter=i,
    #                       activation='relu',
    #                       hidden_layer_sizes=(10, 10, 2), random_state=666)
    x = list(range(interval, max_iter + 1, interval))
    y_tr_relu = [0.15, 0.455, 0.816, 0.834, 0.845, 0.847, 0.854, 0.86, 0.862, 0.866]
    y_tr_logi = [0.099, 0.257, 0.419, 0.43, 0.412, 0.396, 0.394, 0.392, 0.39, 0.428]
    y_tr_tanh = [0.476, 0.59, 0.588, 0.624, 0.614, 0.621, 0.629, 0.653, 0.776, 0.803]
    y_t_relu = [0.17, 0.445, 0.777, 0.788, 0.796, 0.795, 0.806, 0.812, 0.814, 0.814]
    y_t_logi = [0.106, 0.198, 0.411, 0.421, 0.411, 0.406, 0.403, 0.401, 0.4, 0.452]
    y_t_tanh = [0.436, 0.566, 0.574, 0.618, 0.605, 0.605, 0.608, 0.622, 0.718, 0.745]

    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.plot(x, y_tr_relu, color='#90EE90', linewidth=1.7, label='relu')
    ax.plot(x, y_tr_logi, color='#ffa07a', linewidth=1.7, label='sigmoid')
    ax.plot(x, y_tr_tanh, color='#9999ff', linewidth=1.7, label='tanh')
    ax.scatter(x, y_tr_relu, s=13, c='#90EE90')
    ax.scatter(x, y_tr_logi, s=13, c='#ffa07a')
    ax.scatter(x, y_tr_tanh, s=13, c='#9999ff')
    ax.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('report/img/mlp_activation_sat_tr')
    plt.show()

    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.plot(x, y_t_relu, color='#90EE90', linewidth=1.7, label='relu')
    ax.plot(x, y_t_logi, color='#ffa07a', linewidth=1.7, label='sigmoid')
    ax.plot(x, y_t_tanh, color='#9999ff', linewidth=1.7, label='tanh')
    ax.scatter(x, y_t_relu, s=13, c='#90EE90')
    ax.scatter(x, y_t_logi, s=13, c='#ffa07a')
    ax.scatter(x, y_t_tanh, s=13, c='#9999ff')
    ax.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('report/img/mlp_activation_sat_t')
    plt.show()

def plot_optimizer_splice():
    # model = MLPClassifier(solver='adam', alpha=1e-3,
    #                       learning_rate_init=0.001, max_iter=i,
    #                       activation='relu',
    #                       hidden_layer_sizes=(10, 10, 2), random_state=666)
    x = list(range(interval, max_iter + 1, interval))
    y_tr_lbfgs = [0.779, 0.847, 0.88, 0.913, 0.925, 0.943, 0.952, 0.963, 0.964, 0.97]
    y_tr_sgd = [0.57, 0.63, 0.643, 0.673, 0.698, 0.723, 0.752, 0.761, 0.781, 0.796]
    y_tr_adam = [0.698, 0.764, 0.813, 0.845, 0.865, 0.884, 0.906, 0.921, 0.921, 0.921]
    y_t_lbfgs = [0.794, 0.849, 0.848, 0.874, 0.878, 0.875, 0.873, 0.866, 0.857, 0.855]
    y_t_sgd = [0.595, 0.637, 0.651, 0.671, 0.703, 0.724, 0.746, 0.76, 0.768, 0.788]
    y_t_adam = [0.68, 0.759, 0.822, 0.85, 0.857, 0.867, 0.874, 0.883, 0.883, 0.883]

    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.plot(x, y_tr_lbfgs, color='#90EE90', linewidth=1.7, label='lbfgs')
    ax.plot(x, y_tr_sgd, color='#ffa07a', linewidth=1.7, label='sgd')
    ax.plot(x, y_tr_adam, color='#9999ff', linewidth=1.7, label='adam')
    ax.scatter(x, y_tr_lbfgs, s=13, c='#90EE90')
    ax.scatter(x, y_tr_sgd, s=13, c='#ffa07a')
    ax.scatter(x, y_tr_adam, s=13, c='#9999ff')
    ax.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('report/img/mlp_optimizer_splice_tr')
    plt.show()

    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.plot(x, y_t_lbfgs, color='#90EE90', linewidth=1.7, label='lbfgs')
    ax.plot(x, y_t_sgd, color='#ffa07a', linewidth=1.7, label='sgd')
    ax.plot(x, y_t_adam, color='#9999ff', linewidth=1.7, label='adam')
    ax.scatter(x, y_t_lbfgs, s=13, c='#90EE90')
    ax.scatter(x, y_t_sgd, s=13, c='#ffa07a')
    ax.scatter(x, y_t_adam, s=13, c='#9999ff')
    ax.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('report/img/mlp_optimizer_splice_t')
    plt.show()

def plot_optimizer_sat():
    # model = MLPClassifier(solver='adam', alpha=1e-3,
    #                       learning_rate_init=0.001, max_iter=i,
    #                       activation='relu',
    #                       hidden_layer_sizes=(10, 10, 2), random_state=666)
    x = list(range(interval, max_iter + 1, interval))
    y_tr_lbfgs = [0.257, 0.257, 0.257, 0.257, 0.257, 0.257, 0.257, 0.257, 0.257, 0.257]
    y_tr_sgd = [0.099, 0.257, 0.257, 0.407, 0.434, 0.446, 0.458, 0.471, 0.485, 0.518]
    y_tr_adam = [0.15, 0.455, 0.816, 0.834, 0.845, 0.847, 0.854, 0.86, 0.862, 0.866]
    y_t_lbfgs = [0.198, 0.198, 0.198, 0.198, 0.198, 0.198, 0.198, 0.198, 0.198, 0.198]
    y_t_sgd = [0.106, 0.198, 0.198, 0.397, 0.43, 0.444, 0.45, 0.463, 0.477, 0.508]
    y_t_adam = [0.17, 0.445, 0.777, 0.788, 0.796, 0.795, 0.806, 0.812, 0.814, 0.814]

    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.plot(x, y_tr_lbfgs, color='#90EE90', linewidth=1.7, label='lbfgs')
    ax.plot(x, y_tr_sgd, color='#ffa07a', linewidth=1.7, label='sgd')
    ax.plot(x, y_tr_adam, color='#9999ff', linewidth=1.7, label='adam')
    ax.scatter(x, y_tr_lbfgs, s=13, c='#90EE90')
    ax.scatter(x, y_tr_sgd, s=13, c='#ffa07a')
    ax.scatter(x, y_tr_adam, s=13, c='#9999ff')
    ax.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('report/img/mlp_optimizer_sat_tr')
    plt.show()

    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.plot(x, y_t_lbfgs, color='#90EE90', linewidth=1.7, label='lbfgs')
    ax.plot(x, y_t_sgd, color='#ffa07a', linewidth=1.7, label='sgd')
    ax.plot(x, y_t_adam, color='#9999ff', linewidth=1.7, label='adam')
    ax.scatter(x, y_t_lbfgs, s=13, c='#90EE90')
    ax.scatter(x, y_t_sgd, s=13, c='#ffa07a')
    ax.scatter(x, y_t_adam, s=13, c='#9999ff')
    ax.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('report/img/mlp_optimizer_sat_t')
    plt.show()

def plot_lr_splice():
    # model = MLPClassifier(solver='adam', alpha=1e-3,
    #                       learning_rate_init=0.001, max_iter=300,
    #                       activation='relu',
    #                       hidden_layer_sizes=(10, 10, 2), random_state=666)
    x = [0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1]
    y_tr = [0.921, 0.937, 0.911, 0.832, 0.517, 0.517, 0.517]
    y_t = [0.883, 0.888, 0.876, 0.833, 0.52, 0.52, 0.52]

    x_ax = np.arange(7)
    total_width, n = 0.75, 2
    width = total_width / n
    x_ax = x_ax - (total_width - width) / 2

    plt.bar(x_ax, y_tr, width=width, facecolor='#9999ff', edgecolor='white', label='Training set')
    plt.bar(x_ax + width, y_t, width=width, facecolor='#ffa07a', edgecolor='white', label='Testing set')
    for x, y1, y2 in zip(x_ax, y_tr, y_t):
        plt.text(x - 0.02, y1, '%.2f' % y1, ha='center', va='bottom')
        plt.text(x + width + 0.05, y2, '%.2f' % y2, ha='center', va='bottom')

    ax = plt.gca()
    ax.set_xticks(x_ax + width / 2)
    ax.set_xticklabels((0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1))
    plt.xlabel('Learning rate')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('report/img/mlp_lr_splice')
    plt.show()

def plot_lr_sat():
    # model = MLPClassifier(solver='adam', alpha=1e-3,
    #                       learning_rate_init=0.001, max_iter=300,
    #                       activation='relu',
    #                       hidden_layer_sizes=(10, 10, 2), random_state=666)
    x = [0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1]
    y_tr = [0.877, 0.875, 0.87, 0.832, 0.87, 0.862, 0.715]
    y_t = [0.832, 0.818, 0.812, 0.833, 0.826, 0.817, 0.661]

    x_ax = np.arange(7)
    total_width, n = 0.75, 2
    width = total_width / n
    x_ax = x_ax - (total_width - width) / 2

    plt.bar(x_ax, y_tr, width=width, facecolor='#9999ff', edgecolor='white', label='Training set')
    plt.bar(x_ax + width, y_t, width=width, facecolor='#ffa07a', edgecolor='white', label='Testing set')
    for x, y1, y2 in zip(x_ax, y_tr, y_t):
        plt.text(x - 0.02, y1, '%.2f' % y1, ha='center', va='bottom')
        plt.text(x + width + 0.05, y2, '%.2f' % y2, ha='center', va='bottom')

    ax = plt.gca()
    ax.set_xticks(x_ax + width / 2)
    ax.set_xticklabels((0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1))
    plt.xlabel('Learning rate')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.06)
    plt.legend()
    plt.savefig('report/img/mlp_lr_sat')
    plt.show()

def plot_dim_splice():
    # model = MLPClassifier(solver='adam', alpha=1e-3,
    #                       learning_rate_init=0.001, max_iter=i,
    #                       activation='relu',
    #                       hidden_layer_sizes=(10, 10, 2), random_state=666)
    x = list(range(interval, max_iter + 1, interval))
    y_25_tr = [0.665, 0.727, 0.769, 0.792, 0.815, 0.832, 0.841, 0.853, 0.865, 0.876]
    y_5_tr = [0.541, 0.727, 0.814, 0.833, 0.854, 0.864, 0.876, 0.887, 0.899, 0.909]
    y_75_tr = [0.524, 0.712, 0.811, 0.858, 0.891, 0.914, 0.932, 0.956, 0.964, 0.974]
    y_tr = [0.698, 0.764, 0.813, 0.845, 0.865, 0.884, 0.906, 0.921, 0.921, 0.921]
    y_25_t = [0.392, 0.443, 0.486, 0.525, 0.55, 0.562, 0.573, 0.574, 0.575, 0.58]
    y_5_t = [0.541, 0.727, 0.814, 0.833, 0.854, 0.864, 0.876, 0.887, 0.899, 0.909]
    y_75_t = [0.522, 0.564, 0.56, 0.569, 0.571, 0.571, 0.576, 0.586, 0.586, 0.582]
    y_t = [0.68, 0.759, 0.822, 0.85, 0.857, 0.867, 0.874, 0.883, 0.883, 0.883]

    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.plot(x, y_25_tr, color='#90EE90', linewidth=1.7, label='25% features')
    ax.plot(x, y_5_tr, color='#ffa07a', linewidth=1.7, label='50% features')
    ax.plot(x, y_75_tr, color='#9999ff', linewidth=1.7, label='75% features')
    ax.plot(x, y_tr, color='#F0E68C', linewidth=1.7, label='100% features')
    ax.scatter(x, y_25_tr, s=13, c='#90EE90')
    ax.scatter(x, y_5_tr, s=13, c='#ffa07a')
    ax.scatter(x, y_75_tr, s=13, c='#9999ff')
    ax.scatter(x, y_tr, s=13, c='#F0E68C')
    ax.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('report/img/mlp_dim_splice_tr')
    plt.show()

    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.plot(x, y_25_t, color='#90EE90', linewidth=1.7, label='25% features')
    ax.plot(x, y_5_t, color='#ffa07a', linewidth=1.7, label='50% features')
    ax.plot(x, y_75_t, color='#9999ff', linewidth=1.7, label='75% features')
    ax.plot(x, y_t, color='#F0E68C', linewidth=1.7, label='100% features')
    ax.scatter(x, y_25_t, s=13, c='#90EE90')
    ax.scatter(x, y_5_t, s=13, c='#ffa07a')
    ax.scatter(x, y_75_t, s=13, c='#9999ff')
    ax.scatter(x, y_t, s=13, c='#F0E68C')
    ax.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('report/img/mlp_dim_splice_t')
    plt.show()

def plot_dim_sat():
    # model = MLPClassifier(solver='adam', alpha=1e-3,
    #                       learning_rate_init=0.001, max_iter=i,
    #                       activation='relu',
    #                       hidden_layer_sizes=(10, 10, 2), random_state=666)
    x = list(range(interval, max_iter + 1, interval))
    y_25_tr = [0.413, 0.44, 0.489, 0.681, 0.79, 0.8, 0.809, 0.815, 0.823, 0.826]
    y_5_tr = [0.174, 0.396, 0.477, 0.649, 0.769, 0.776, 0.792, 0.806, 0.823, 0.832]
    y_75_tr = [0.342, 0.406, 0.412, 0.416, 0.453, 0.504, 0.742, 0.754, 0.763, 0.798]
    y_tr = [0.15, 0.455, 0.816, 0.834, 0.845, 0.847, 0.854, 0.86, 0.862, 0.866]
    y_25_t = [0.384, 0.424, 0.482, 0.604, 0.683, 0.68, 0.669, 0.664, 0.664, 0.662]
    y_5_t = [0.235, 0.3, 0.354, 0.536, 0.604, 0.574, 0.57, 0.568, 0.575, 0.583]
    y_75_t = [0.322, 0.372, 0.382, 0.432, 0.452, 0.464, 0.462, 0.469, 0.487, 0.522]
    y_t = [0.17, 0.445, 0.777, 0.788, 0.796, 0.795, 0.806, 0.812, 0.814, 0.814]

    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.plot(x, y_25_tr, color='#90EE90', linewidth=1.7, label='25% features')
    ax.plot(x, y_5_tr, color='#ffa07a', linewidth=1.7, label='50% features')
    ax.plot(x, y_75_tr, color='#9999ff', linewidth=1.7, label='75% features')
    ax.plot(x, y_tr, color='#F0E68C', linewidth=1.7, label='100% features')
    ax.scatter(x, y_25_tr, s=13, c='#90EE90')
    ax.scatter(x, y_5_tr, s=13, c='#ffa07a')
    ax.scatter(x, y_75_tr, s=13, c='#9999ff')
    ax.scatter(x, y_tr, s=13, c='#F0E68C')
    ax.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('report/img/mlp_dim_sat_tr')
    plt.show()

    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.plot(x, y_25_t, color='#90EE90', linewidth=1.7, label='25% features')
    ax.plot(x, y_5_t, color='#ffa07a', linewidth=1.7, label='50% features')
    ax.plot(x, y_75_t, color='#9999ff', linewidth=1.7, label='75% features')
    ax.plot(x, y_t, color='#F0E68C', linewidth=1.7, label='100% features')
    ax.scatter(x, y_25_t, s=13, c='#90EE90')
    ax.scatter(x, y_5_t, s=13, c='#ffa07a')
    ax.scatter(x, y_75_t, s=13, c='#9999ff')
    ax.scatter(x, y_t, s=13, c='#F0E68C')
    ax.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('report/img/mlp_dim_sat_t')
    plt.show()

def plot_archi_sat():
    # model = MLPClassifier(solver='adam', alpha=1e-3,
    #                       learning_rate_init=0.001, max_iter=i,
    #                       activation='relu',
    #                       hidden_layer_sizes=(10, 10, 2), random_state=666)

    # 1: (10, 10)
    # 2: (10, 10, 10, 10)
    # 3: (50, 50)
    # 4: (50, 50, 50, 50)

    x = list(range(interval, max_iter + 1, interval))
    y_1_tr = [0.15, 0.455, 0.816, 0.834, 0.845, 0.847, 0.854, 0.86, 0.862, 0.866]
    y_2_tr = [0.257, 0.257, 0.257, 0.257, 0.257, 0.257, 0.257, 0.257, 0.257, 0.257]
    y_3_tr = [0.679, 0.729, 0.743, 0.756, 0.762, 0.769, 0.847, 0.858, 0.871, 0.901]
    y_4_tr = [0.337, 0.558, 0.558, 0.611, 0.652, 0.652, 0.652, 0.652, 0.652, 0.652]
    y_1_t = [0.17, 0.445, 0.777, 0.788, 0.796, 0.795, 0.806, 0.812, 0.814, 0.814]
    y_2_t = [0.198, 0.198, 0.198, 0.198, 0.198, 0.198, 0.198, 0.198, 0.198, 0.198]
    y_3_t = [0.652, 0.714, 0.719, 0.723, 0.726, 0.727, 0.779, 0.79, 0.814, 0.83]
    y_4_t = [0.279, 0.484, 0.476, 0.552, 0.606, 0.606, 0.606, 0.606, 0.606, 0.606]

    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.plot(x, y_1_tr, color='#90EE90', linewidth=1.7, label='(10, 10)')
    ax.plot(x, y_2_tr, color='#ffa07a', linewidth=1.7, label='(10, 10, 10, 10)')
    ax.plot(x, y_3_tr, color='#9999ff', linewidth=1.7, label='(50, 50)')
    ax.plot(x, y_4_tr, color='#F0E68C', linewidth=1.7, label='(50, 50, 50, 50)')
    ax.scatter(x, y_1_tr, s=13, c='#90EE90')
    ax.scatter(x, y_2_tr, s=13, c='#ffa07a')
    ax.scatter(x, y_3_tr, s=13, c='#9999ff')
    ax.scatter(x, y_4_tr, s=13, c='#F0E68C')
    ax.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.ylim(0, 0.97)
    plt.legend()
    plt.savefig('report/img/mlp_archi_sat_tr')
    plt.show()

    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.plot(x, y_1_t, color='#90EE90', linewidth=1.7, label='(10, 10)')
    ax.plot(x, y_2_t, color='#ffa07a', linewidth=1.7, label='(10, 10, 10, 10)')
    ax.plot(x, y_3_t, color='#9999ff', linewidth=1.7, label='(50, 50)')
    ax.plot(x, y_4_t, color='#F0E68C', linewidth=1.7, label='(50, 50, 50, 50)')
    ax.scatter(x, y_1_t, s=13, c='#90EE90')
    ax.scatter(x, y_2_t, s=13, c='#ffa07a')
    ax.scatter(x, y_3_t, s=13, c='#9999ff')
    ax.scatter(x, y_4_t, s=13, c='#F0E68C')
    ax.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.ylim(0, 0.92)
    plt.legend()
    plt.savefig('report/img/mlp_archi_sat_t')
    plt.show()

def plot_archi_splice():
    # model = MLPClassifier(solver='adam', alpha=1e-3,
    #                       learning_rate_init=0.001, max_iter=i,
    #                       activation='relu',
    #                       hidden_layer_sizes=(10, 10, 2), random_state=666)

    # 1: (10, 10)
    # 2: (10, 10, 10, 10)
    # 3: (50, 50)
    # 4: (50, 50, 50, 50)

    x = list(range(interval, max_iter + 1, interval))
    y_1_tr = [0.698, 0.764, 0.813, 0.845, 0.865, 0.884, 0.906, 0.921, 0.921, 0.921]
    y_2_tr = [0.616, 0.691, 0.779, 0.827, 0.838, 0.858, 0.86, 0.86, 0.86, 0.86]
    y_3_tr = [0.483, 0.483, 0.483, 0.483, 0.483, 0.483, 0.483, 0.483, 0.483, 0.483]
    y_4_tr = [0.785, 0.867, 0.914, 0.969, 0.992, 1.0, 1.0, 1.0, 1.0, 1.0]
    y_1_t = [0.68, 0.759, 0.822, 0.85, 0.857, 0.867, 0.874, 0.883, 0.883, 0.883]
    y_2_t = [0.608, 0.687, 0.775, 0.816, 0.835, 0.843, 0.842, 0.842, 0.842, 0.842]
    y_3_t = [0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48]
    y_4_t = [0.76, 0.83, 0.844, 0.844, 0.849, 0.842, 0.841, 0.844, 0.844, 0.844]

    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.plot(x, y_1_tr, color='#90EE90', linewidth=1.7, label='(10, 10)')
    ax.plot(x, y_2_tr, color='#ffa07a', linewidth=1.7, label='(10, 10, 10, 10)')
    ax.plot(x, y_3_tr, color='#9999ff', linewidth=1.7, label='(50, 50)')
    ax.plot(x, y_4_tr, color='#F0E68C', linewidth=1.7, label='(50, 50, 50, 50)')
    ax.scatter(x, y_1_tr, s=13, c='#90EE90')
    ax.scatter(x, y_2_tr, s=13, c='#ffa07a')
    ax.scatter(x, y_3_tr, s=13, c='#9999ff')
    ax.scatter(x, y_4_tr, s=13, c='#F0E68C')
    ax.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('report/img/mlp_archi_splice_tr')
    plt.show()

    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.plot(x, y_1_t, color='#90EE90', linewidth=1.7, label='(10, 10)')
    ax.plot(x, y_2_t, color='#ffa07a', linewidth=1.7, label='(10, 10, 10, 10)')
    ax.plot(x, y_3_t, color='#9999ff', linewidth=1.7, label='(50, 50)')
    ax.plot(x, y_4_t, color='#F0E68C', linewidth=1.7, label='(50, 50, 50, 50)')
    ax.scatter(x, y_1_t, s=13, c='#90EE90')
    ax.scatter(x, y_2_t, s=13, c='#ffa07a')
    ax.scatter(x, y_3_t, s=13, c='#9999ff')
    ax.scatter(x, y_4_t, s=13, c='#F0E68C')
    ax.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.ylim(0, 0.92)
    plt.legend()
    plt.savefig('report/img/mlp_archi_splice_t')
    plt.show()

################################### MLP Part #####################################

if __name__ == '__main__':
    # splice satimage.scale
    #mlp('satimage.scale')
    #mlp('splice')
    plot_archi_splice()
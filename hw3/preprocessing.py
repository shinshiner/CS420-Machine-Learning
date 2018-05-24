import numpy as np

def process(fname, feature_size):
    targets = []
    features = []

    with open(fname, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            items = line.split()
            tmp = int(items.pop(0))
            if tmp == -1:
                tmp = 0
            targets.append(tmp)

            feature = []
            for j, item in enumerate(items):
                feature.append(float(item.split(':')[1]))
            while len(feature) < feature_size:
                feature.append(0.)
            features.append(feature)

    # if feature_size == 500:
    #     one_hot = np.zeros((len(targets), 2))
    # elif feature_size == 36:
    #     one_hot = np.zeros((len(targets), 6))
    #
    # for i, t in enumerate(targets):
    #     if feature_size == 500 and t == -1:
    #         normal_t = 0
    #     elif feature_size == 36:
    #         normal_t = t - 1
    #     else:
    #         normal_t = t
    #     one_hot[i][normal_t] = 1

    # save as .npy
    np.save(fname + '_feature.npy', np.array(features))
    np.save(fname + '_target.npy', np.array(targets))

def check(fname):
    arr = np.load(fname)
    print(arr.shape)
    print(arr)
    # for i in range(arr.shape[0]):
    #     print(arr[i])

if __name__ == '__main__':
    process('data/splice.t', 60)
    check('data/splice.t_feature.npy')
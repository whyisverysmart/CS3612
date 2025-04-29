import numpy as np


np.random.seed(42)


def load_data():
    glass = np.loadtxt('glass.txt', delimiter=',')

    X = glass[:, 1:10]
    y = glass[:, -1]

    indices = np.random.permutation(len(glass))

    train_size = int(0.9 * len(glass))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]


    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    return X_train, X_test, y_train, y_test

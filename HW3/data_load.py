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

    # Normalize the data
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test, y_train, y_test

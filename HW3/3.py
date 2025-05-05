from data_load import load_data
import numpy as np
from plot_pca import plot_pca

def lda_train_binary(X, y):
    Omega1 = X[y == 1]
    Omega0 = X[y == 0]

    miu1 = np.mean(Omega1, axis=0)
    miu0 = np.mean(Omega0, axis=0)

    S1 = np.cov(Omega1, rowvar=False)
    S0 = np.cov(Omega0, rowvar=False)
    Sw = Omega1.shape[0] * S1 + Omega0.shape[0] * S0

    Sw_inv = np.linalg.pinv(Sw)
    beta = Sw_inv @ (miu1 - miu0)
    b = -0.5 * (miu1 + miu0) @ beta
    return beta, b


def train(X_train, y_train, class_labels):
    classifiers = {}
    for label in class_labels:
        binary_y = (y_train == label).astype(int)
        print(f"Training classifier for class {label}...")
        beta, b = lda_train_binary(X_train, binary_y)
        classifiers[label] = (beta, b)
    return classifiers


def predict(X, classifiers):
    scores = {}
    for label, (beta, b) in classifiers.items():
        scores[label] = X @ beta + b

    preds = np.array([
        max(scores, key=lambda label: scores[label][i])
        for i in range(X.shape[0])
    ])
    return preds


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    class_labels = np.unique(y_train).tolist()

    classifiers = train(X_train, y_train, class_labels)

    y_pred = predict(X_test, classifiers)
    accuracy = np.mean(y_pred == y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

    W = np.array([classifiers[label][0] for label in class_labels])
    plot_pca(X_train, y_train, W, '3.png')

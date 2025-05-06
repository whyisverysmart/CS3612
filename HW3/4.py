from data_load import load_data
import numpy as np
from plot_pca import plot_pca

def logistic_loss_train(X, y, lr, num_iter):
    m, n = X.shape
    # Add intercept term
    X = np.hstack((np.ones((m, 1)), X))
    beta = np.zeros(n + 1)
    for _ in range(num_iter):
        z = X @ beta
        grad = -(1/m) * (X.T @ (y / (1 + np.exp(y * z))))
        beta -= lr * grad
    return beta

def train(X_train, y_train, class_labels, lr, num_iter):
    classifiers = {}
    for label in class_labels:
        binary_y = np.where(y_train == label, 1, -1)
        print(f"Training classifier for class {label}...")
        beta = logistic_loss_train(X_train, binary_y, lr, num_iter)
        classifiers[label] = beta
    return classifiers

def predict(X, classifiers):
    m = X.shape[0]
    X = np.hstack((np.ones((m, 1)), X))
    scores = {}

    for label, beta in classifiers.items():
        scores[label] = X @ beta

    preds = np.array([
        max(scores, key=lambda label: scores[label][i])
        for i in range(m)
    ])
    return preds


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    class_labels = np.unique(y_train).tolist()
    classifiers = train(X_train, y_train, class_labels, lr=0.05, num_iter=1000)

    y_pred = predict(X_test, classifiers)
    accuracy = np.mean(y_pred == y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

    W = np.array([classifiers[label] for label in class_labels])
    plot_pca(X_train, y_train, W, "4.png")

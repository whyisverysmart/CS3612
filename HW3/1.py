from data_load import load_data
import numpy as np
from plot_pca import plot_pca

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression_train(X, y, lr, num_iter):
    m, n = X.shape
    # Add intercept term
    X = np.hstack((np.ones((m, 1)), X))
    beta = np.zeros(n + 1)
    for _ in range(num_iter):
        z = X @ beta
        h = sigmoid(z)
        grad = (1/m) * (X.T @ (h - y))
        beta -= lr * grad
    return beta

def train(X_train, y_train, class_labels, lr, num_iter):
    classifiers = {}
    for label in class_labels:
        binary_y = (y_train == label).astype(int)
        print(f"Training classifier for class {label}...")
        beta = logistic_regression_train(X_train, binary_y, lr, num_iter)
        classifiers[label] = beta
    return classifiers

def predict(X, classifiers):
    m = X.shape[0]
    X = np.hstack((np.ones((m, 1)), X))
    results = {}
    for label, beta in classifiers.items():
        results[label] = sigmoid(X @ beta)
    preds = np.array([max(results, key=lambda label: results[label][i]) for i in range(m)])
    return preds


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    class_labels = np.unique(y_train).tolist()
    classifiers = train(X_train, y_train, class_labels, lr=0.01, num_iter=2000)

    y_pred = predict(X_test, classifiers)
    accuracy = np.mean(y_pred == y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

    W = np.array([classifiers[label] for label in class_labels]) # [6, 10]
    plot_pca(X_train, y_train, W, "1.png")

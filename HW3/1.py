from data_load import load_data
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logistic_regression_train(X, y, lr=0.01, num_iter=1000):
    m, n = X.shape
    X_bias = np.hstack((np.ones((m, 1)), X))
    theta = np.zeros(n + 1)

    for _ in range(num_iter):
        z = X_bias @ theta
        h = sigmoid(z)
        gradient = (1/m) * (X_bias.T @ (h - y))
        theta -= lr * gradient

    return theta

def train_ovr(X_train, y_train, class_labels, lr=0.01, num_iter=1000):
    classifiers = {}
    for label in class_labels:
        binary_y = (y_train == label).astype(int)
        theta = logistic_regression_train(X_train, binary_y, lr, num_iter)
        classifiers[label] = theta
    return classifiers


def predict_ovr(X, classifiers):
    m = X.shape[0]
    X_bias = np.hstack((np.ones((m, 1)), X))  # 加偏置项
    confidences = {}

    for label, theta in classifiers.items():
        confidences[label] = sigmoid(X_bias @ theta)

    # 对每个样本，选最大概率的类别
    preds = np.array([
        max(confidences, key=lambda label: confidences[label][i])
        for i in range(m)
    ])
    return preds


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    class_labels = [1, 2, 3, 5, 6, 7]
    assert class_labels == np.unique(y_train).tolist(), "Class labels do not match the unique values in y_train"
    classifiers = train_ovr(X_train, y_train, class_labels)

    y_pred = predict_ovr(X_test, classifiers)

    accuracy = np.mean(y_pred == y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

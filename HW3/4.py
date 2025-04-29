from data_load import load_data
import numpy as np


def logistic_loss_train(X, y, lr=0.1, num_iter=1000):
    """
    X: m x n matrix
    y: m-dimensional vector, values in {+1, -1}
    """
    m, n = X.shape
    X_bias = np.hstack((np.ones((m, 1)), X))  # 加偏置项
    w = np.zeros(n + 1)  # 初始化参数

    for _ in range(num_iter):
        z = X_bias @ w
        yz = y * z
        grad = -(1/m) * (X_bias.T @ (y / (1 + np.exp(yz))))  # 梯度
        w -= lr * grad  # 梯度下降更新参数

    return w

def train_ovr_logistic_loss(X_train, y_train, class_labels, lr=0.005, num_iter=2000):
    classifiers = {}
    for label in class_labels:
        binary_y = np.where(y_train == label, 1, -1)  # 将正类设为+1，其余为-1
        w = logistic_loss_train(X_train, binary_y, lr, num_iter)
        classifiers[label] = w
    return classifiers

def predict_ovr_logistic_loss(X, classifiers):
    m = X.shape[0]
    X_bias = np.hstack((np.ones((m, 1)), X))  # 加偏置项
    scores = {}

    for label, w in classifiers.items():
        scores[label] = X_bias @ w  # 分数 (不是sigmoid输出)

    preds = np.array([
        max(scores, key=lambda label: scores[label][i])
        for i in range(m)
    ])
    return preds


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    class_labels = [1, 2, 3, 5, 6, 7]

    classifiers = train_ovr_logistic_loss(X_train, y_train, class_labels)

    y_pred = predict_ovr_logistic_loss(X_test, classifiers)

    accuracy = np.mean(y_pred == y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

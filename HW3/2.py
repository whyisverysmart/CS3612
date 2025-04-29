from data_load import load_data
import numpy as np


class_labels = [1, 2, 3, 5, 6, 7]
label_to_index = {label: idx for idx, label in enumerate([1,2,3,5,6,7])}
index_to_label = {idx: label for label, idx in label_to_index.items()}

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)  # 数值稳定：减去每行最大值
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy_loss(p, y_one_hot):
    m = y_one_hot.shape[0]  # 样本数
    loss = -np.sum(y_one_hot * np.log(p + 1e-8)) / m  # 加一个小常数防止log(0)
    return loss


def compute_gradient(X, p, y_one_hot):
    m = X.shape[0]
    grad = (X.T @ (p - y_one_hot)) / m
    return grad


def train_softmax_classifier(X_train, y_train, num_classes=6, lr=0.005, num_iter=2000):
    m, n = X_train.shape
    W = np.random.randn(n, num_classes) * 0.01  # 随机初始化权重矩阵

    # 把y_train转成one-hot
    y_one_hot = np.zeros((m, num_classes))
    for idx, label in enumerate(y_train):
        y_one_hot[idx, label_to_index[label]] = 1

    losses = []

    for i in range(num_iter):
        scores = X_train @ W
        p = softmax(scores)
        loss = cross_entropy_loss(p, y_one_hot)
        losses.append(loss)

        grad = compute_gradient(X_train, p, y_one_hot)
        W -= lr * grad

        if (i + 1) % 100 == 0:
            print(f"Iteration {i+1}/{num_iter}, Loss: {loss:.4f}")

    return W, losses


def predict(X, W):
    scores = X @ W
    p = softmax(scores)
    pred_indices = np.argmax(p, axis=1)
    preds = np.array([index_to_label[idx] for idx in pred_indices])
    return preds

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    assert class_labels == np.unique(y_train).tolist(), "Class labels do not match the unique values in y_train"

    W, losses = train_softmax_classifier(X_train, y_train, num_classes=6, lr=0.01, num_iter=2500)

    y_pred = predict(X_test, W)

    accuracy = np.mean(y_pred == y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

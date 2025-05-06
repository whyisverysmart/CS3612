from data_load import load_data
import numpy as np
from tqdm import tqdm
from plot_pca import plot_pca, plot_loss

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(p, y_one_hot):
    m = y_one_hot.shape[0]
    loss = -np.sum(y_one_hot * np.log(p + 1e-8)) / m
    return loss

def train(X_train, y_train, num_classes=6, lr=0.005, num_iter=2000):
    m, n = X_train.shape
    W = np.random.randn(n, num_classes) * 0.01

    y_one_hot = np.zeros((m, num_classes))
    for idx, label in enumerate(y_train):
        y_one_hot[idx, label_to_index[label]] = 1

    loss_history = []
    for _ in tqdm(range(num_iter)):
        scores = X_train @ W
        p = softmax(scores)

        loss = cross_entropy_loss(p, y_one_hot)
        loss_history.append(loss)

        grad = (X_train.T @ (p - y_one_hot)) / m
        W -= lr * grad

    return W, loss_history


def predict(X, W):
    scores = X @ W
    p = softmax(scores)
    pred_indices = np.argmax(p, axis=1)
    preds = np.array([index_to_label[idx] for idx in pred_indices])
    return preds

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    class_labels = np.unique(y_train).tolist()
    label_to_index = {label: idx for idx, label in enumerate(class_labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}

    W, loss_history = train(X_train, y_train, num_classes=6, lr=0.04, num_iter=3000)
    # Visualize the loss
    plot_loss(loss_history)

    y_pred = predict(X_test, W)
    accuracy = np.mean(y_pred == y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

    plot_pca(X_train, y_train, W.T, "2.png")

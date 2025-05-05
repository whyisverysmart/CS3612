import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_pca(X_train, y_train, W, file_name):
    # Remove intercept terms
    if W.shape[1] == 10:
        W = W[:, 1:]

    pca = PCA(n_components=2)
    W_2 = pca.fit_transform(W)
    P = pca.components_
    X_2 = X_train @ P.T

    plt.figure(figsize=(10, 8))
    class_labels = np.unique(y_train)
    colors = plt.cm.get_cmap('tab10', len(class_labels))

    # Sample points
    for i, label in enumerate(class_labels):
        idx = (y_train == label)
        plt.scatter(
            X_2[idx, 0], X_2[idx, 1],
            label=f'Class {int(label)}',
            alpha=0.7,
            color=colors(i),
            s=30
        )

    # W vectors
    origin = np.mean(X_2, axis=0)
    for i, label in enumerate(class_labels):
        plt.arrow(
            origin[0], origin[1],
            W_2[i, 0] - origin[0],
            W_2[i, 1] - origin[1],
            color=colors(i),
            width=0.002,
            head_width=0.05,
            length_includes_head=True,
        )
        plt.text(W_2[i, 0], W_2[i, 1], f'W{int(label)}', fontsize=12, color=colors(i))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_name, dpi=300)

def plot_loss(loss_history):
    plt.plot(loss_history)
    plt.savefig("2_loss.png")

if __name__ == "__main__":
    X_train = np.random.rand(192, 9)
    y_train = np.random.randint(0, 6, size=192)
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    W = np.random.rand(6, 10)
    plot_pca(X_train, y_train, W, "pca_plot.png")
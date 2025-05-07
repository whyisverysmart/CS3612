import numpy as np
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def plot_support(support_indices, support_weights):
    original_images = np.load('X_train_sampled.npy')
    images = original_images[support_indices]

    num_sv = len(images)
    images_per_page = 420
    cols = 21
    rows = images_per_page // cols
    pages = 2

    for page in range(pages):
        fig, axes = plt.subplots(rows, cols, figsize=(28, 28))
        # fig.suptitle(f'Support Vectors Page {page+1}', fontsize=16)

        for i in range(rows * cols):
            idx = page * images_per_page + i
            if idx >= num_sv:
                axes.flat[i].axis('off')
                continue
            axes.flat[i].imshow(images[idx], cmap='gray')
            axes.flat[i].set_title(f"{support_weights[idx]:.2e}", fontsize=8)
            axes.flat[i].axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.savefig(f"support_vectors_page_{page+1}.png", dpi=300)
        plt.close()


X_train_hog = np.load('X_train_hog.npy')
y_train = np.load('y_train_sampled.npy')
X_test_hog = np.load('X_test_hog.npy')
y_test = np.load('y_test_sampled.npy')

# Lower C makes the decision surface smoother, higher C aims for a more complex decision boundary
# max_iter = 5 results in a warning, but the prformance is slightly better?
classifier1 = SVC(kernel='linear', C=1, max_iter=5000)
classifier2 = LinearSVC(C=0.1, max_iter=100)

classifier1.fit(X_train_hog, y_train)
classifier2.fit(X_train_hog, y_train)

y_pred1 = classifier1.predict(X_test_hog)
y_pred2 = classifier2.predict(X_test_hog)
acc1 = accuracy_score(y_test, y_pred1)
acc2 = accuracy_score(y_test, y_pred2)
print(f"SVC Accuracy: {acc1:.4f}")
print(f"Linear SVC Accuracy: {acc2:.4f}")

support_vectors = classifier1.support_vectors_
support_indices = classifier1.support_
support_weights = classifier1.dual_coef_[0]
print(f"Number of support vectors: {len(support_indices)}")
print(f"Support vectors per class: {classifier1.n_support_}")

# plot_support(support_indices, support_weights)
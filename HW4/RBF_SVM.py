from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score

X_train_hog = np.load('X_train_hog.npy')
y_train = np.load('y_train_sampled.npy')
X_test_hog = np.load('X_test_hog.npy')
y_test = np.load('y_test_sampled.npy')

classifier = SVC(kernel='rbf', C=2, gamma='scale')
classifier.fit(X_train_hog, y_train)

y_pred = classifier.predict(X_test_hog)
acc = accuracy_score(y_test, y_pred)
print(f"RBF SVM Accuracy: {acc:.4f}")

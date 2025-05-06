import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X_train = np.load('X_train_sampled.npy')
y_train = np.load('y_train_sampled.npy')

hog_features = []
for image in X_train:
    _, hog_image = hog(image,
                    orientations=9,
                    pixels_per_cell=(2, 2),
                    cells_per_block=(1, 1),
                    visualize=True)
    hog_vector = hog_image.reshape(-1)
    hog_features.append(hog_vector)

X_hog = np.array(hog_features)
y_labels = y_train

# Step 3: 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_hog, y_labels, test_size=0.2, random_state=42)

# Step 4: 使用 LinearSVC 训练 SVM 模型
clf = LinearSVC(C=1.0, max_iter=10000)  # C 可调整，越小越宽容异常点
clf.fit(X_train, y_train)

# Step 5: 预测并评估
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Linear SVM Accuracy (T-shirt vs Trouser): {acc:.4f}")

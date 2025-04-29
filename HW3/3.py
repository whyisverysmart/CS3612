from data_load import load_data
import numpy as np

def lda_train_binary(X, y):
    X1 = X[y == 1]  # 正类样本
    X0 = X[y == 0]  # 负类样本

    mu1 = np.mean(X1, axis=0)  # 正类均值
    mu0 = np.mean(X0, axis=0)  # 负类均值

    S1 = np.cov(X1, rowvar=False)
    S0 = np.cov(X0, rowvar=False)

    Sw = S1 + S0  # 类内散度矩阵

    w = np.linalg.pinv(Sw) @ (mu1 - mu0)  # 注意使用广义逆防止Sw不可逆

    # 计算偏置项 b，使得决策面更合理
    b = -0.5 * (mu1 + mu0) @ w

    return w, b


def train_ovr_lda(X_train, y_train, class_labels):
    classifiers = {}
    for label in class_labels:
        binary_y = (y_train == label).astype(int)
        w, b = lda_train_binary(X_train, binary_y)
        classifiers[label] = (w, b)
    return classifiers


def predict_ovr_lda(X, classifiers):
    scores = {}
    for label, (w, b) in classifiers.items():
        scores[label] = X @ w + b  # shape = (样本数,)

    preds = np.array([
        max(scores, key=lambda label: scores[label][i])
        for i in range(X.shape[0])
    ])
    return preds


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    class_labels = [1, 2, 3, 5, 6, 7]
    assert class_labels == np.unique(y_train).tolist(), "Class labels do not match the unique values in y_train"

    classifiers = train_ovr_lda(X_train, y_train, class_labels)

    y_pred = predict_ovr_lda(X_test, classifiers)

    accuracy = np.mean(y_pred == y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

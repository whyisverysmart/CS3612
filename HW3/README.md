# 结构
    ├── 1.py                // Task 1 实现
    ├── 2.py                // Task 2 实现
    ├── 3.py                // Task 3 实现
    ├── 4.py                // Task 4 实现
    ├── data_load.py        // 数据集读取
    ├── plot_pca.py         // PCA 和 Loss 可视化
    ├── README.md           // README

# 使用

对于每个 Task 的源码，主要逻辑都相似，具体如下:
1. 读取数据集
```python
X_train, X_test, y_train, y_test = load_data()
class_labels = np.unique(y_train).tolist()
```
2. 训练函数 train()，测试函数 predict () 以及 Accuracy 计算
```python
classifiers = train(X, y, labels, lr, num_iter)

y_pred = predict(X_test, classifiers)
accuracy = np.mean(y_pred == y_test)
print(f"Test Accuracy: {accuracy:.4f}")
```
3. 对于多个二分类任务，整合得到的参数矩阵并进行 PCA 可视化
```python
W = np.array([classifiers[label] for label in class_labels])
plot_pca(X_train, y_train, W, "filename")
```

直接运行对应文件即可:
```bash
python *.py
```
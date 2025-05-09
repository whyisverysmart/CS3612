# Report of HW4

王焕宇 522030910212

---

## Accuracy

我本想用 LinearSVC(C=1, max_iter=100)，因为这个类优化更好，速度更快，但是无法得到准确的支持向量，只能近似得到结果。因此，在测试集上的分类准确率中，我将汇报四种 SVM 的准确率，但是在后续讨论支持向量时，选取 SVC(kernel='linear', C=0.1, max_iter=5000) 作为分类器。

|        Method       | Accuracy |
|---------------------|----------|
| SVC: Linear         |  99.10%  |
| LinearSVC           |  99.10%  |
| SVC: RBF            |  99.15%  |
| SVC: Polynomial     |  99.15%  |

上述结果对应参数如下:
```python
SVC(kernel='linear', C=0.1, max_iter=5000)
LinearSVC(C=1, max_iter=100)
SVC(kernel='rbf', C=2, gamma='scale')
SVC(kernel='poly', degree=3, C=1, gamma='scale', coef0=1)
```

## Linear SVM

1. 一共有 451 个支持向量参与 w 的计算，其中类别 0 包含 232 个，类别 1 包含 219 个。
    ```python
    len(classifier.support_)    # 451
    classifier.n_support_       # [232, 219]
    ```
2. 以下是上述支持向量的可视化，以及对应的权重 (原图见代码压缩包):

<div align="center">
<img src="support_vectors.png" width="98%"/>
</div>
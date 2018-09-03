import numpy as np
from math import sqrt
from collections import Counter
from comm_utils.testCapability import accuracy_score


class KNNClassfier:
    def __init__(self, k):
        """ 初始化kNN分类器 """
        assert k >= 1, "k must be valid"
        self.k = k
        self._x_train = None
        self._y_train = None

    def fit(self, x_train, y_train):
        """ 根据训练数据集x_train和y_train训练kNN分类器 """
        assert x_train.shape[0] == y_train.shape[0], "the size of x_train must be equal to the y_train"
        assert self.k <= x_train.shape[0], "the size of x_train must be as least k."

        self._x_train = x_train
        self._y_train = y_train
        return self

    def predict(self, x_predict):
        """ 给定预测数据集x_predict，返回表示x_predict的结果向量"""
        assert self._x_train is not None and self._y_train is not None, \
            "must fit before predict."
        assert x_predict.shape[1] == self._x_train.shape[1], \
            "the feature number of x_predict must be equal to x_train"

        y_predict = [self._predict(x) for x in x_predict]
        return np.array(y_predict)

    def _predict(self, x):
        """ 给定单个待预测的数据x，返回x的预测结果值"""
        assert x.shape[0] == self._x_train.shape[1], \
            "the feature number of x must be equal to x_train "

        # 计算距离
        distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in self._x_train]
        # 索引排序
        nearest = np.argsort(distances)
        # 获取索引对应的类别值list
        topk_y = [self._y_train[i] for i in nearest[:self.k]]
        # 计数，对类别进行统计
        votes = Counter(topk_y)
        # 返回数目最多的类别
        return votes.most_common(1)[0][0]

    def score(self, x_test, y_test):
        """ 根据测试数据集 x_test和y_test确定当前模型的准确度"""

        y_predict = self.predict(x_test)
        return accuracy_score(y_test, y_predict)

    def __repr__(self):
        return "KNN(k=%d)" % self.k

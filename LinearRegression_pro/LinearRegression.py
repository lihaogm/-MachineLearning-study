import numpy as np
from comm_utils.testCapability import r2_score


class LinearRegression:

    def __init__(self):
        """ 初识化 LR 模型"""
        self.coef_ = None
        self.intercept_ = None
        self._theta = None

    def fit_normal(self, x_train, y_train):
        """ 根据训练数据集x_train,y_train训练LR模型"""
        assert x_train.shape[0] == y_train.shape[0], "the size of x_train must be equal to the size of y_train"

        # 在第一列前添加一列1
        x_b = np.hstack([np.ones((len(x_train), 1)), x_train])
        self._theta = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y_train)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def predict(self, x_predict):
        """ 给定待预测数据集x_predict,返回表示x_predict的结果向量"""
        assert self.intercept_ is not None and self.coef_ is not None, "must fit before predict."
        assert x_predict.shape[1] == len(self.coef_), "the feature number of x_predict must be equal to x_train"

        x_b = np.hstack([np.ones((len(x_predict), 1)), x_predict])
        return x_b.dot(self._theta)

    def score(self, x_test, y_test):
        """ 根据测试数据集 x_test 和 y_test 确定当前模型的准确度"""
        y_predict = self.predict(x_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "LinearRegression()"

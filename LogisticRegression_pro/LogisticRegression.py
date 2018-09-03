import numpy as np
from comm_utils.testCapability import accuracy_score


class LinearRegression:

    def __init__(self):
        """ 初识化 LR 模型"""
        self.coef_ = None
        self.intercept_ = None
        self._theta = None

    def _sigmoid(self, t):
        return 1. / (1. + np.exp(-t))

    def fit(self, x_train, y_train, eta=0.01, n_iters=1e4):
        """ 根据训练数据集x_train,y_train,使用梯度下降法训练Logistic Regression模型"""
        assert x_train.shape[0] == y_train.shape[0], "the size of x_train must be equal to y_train"

        # 定义损失函数
        def j(theta, x_b, y):
            y_hat = self._sigmoid(x_b.dot(theta))
            try:
                return -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / len(y)
            except:
                return float('inf')

        # 定义梯度函数
        def dj(theta, x_b, y):
            # return (x_b.T.dot(x_b.dot(theta) - y) * 2. / len(x_b)).T
            return x_b.T.dot(self._sigmoid(x_b.dot(theta)) - y) / len(x_b)

        # 求梯度下的theta
        def gradient_descent(x_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
            theta = initial_theta
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = dj(theta, x_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                if (abs(j(theta, x_b, y) - j(last_theta, x_b, y)) < epsilon):
                    break
                cur_iter += 1

            return theta

        x_b = np.hstack([np.ones((len(x_train), 1)), x_train])
        initial_theta = np.zeros(x_b.shape[1])
        self._theta = gradient_descent(x_b, y_train, initial_theta, eta, n_iters)

        self.intercept_ = self._theta[0]  # 截距
        self.coef_ = self._theta[1:]  # theta

        return self

    def predict_proba(self, x_predict):
        """ 给定待预测数据集x_predict,返回表示x_predict的结果概率向量"""
        assert self.intercept_ is not None and self.coef_ is not None, "must fit before predict."
        assert x_predict.shape[1] == len(self.coef_), "the feature number of x_predict must be equal to x_train"

        x_b = np.hstack([np.ones((len(x_predict), 1)), x_predict])
        return self._sigmoid(x_b.dot(self._theta))

    def predict(self, x_predict):
        """ 给定待预测数据集x_predict,返回表示x_predict的结果向量"""
        assert self.intercept_ is not None and self.coef_ is not None, "must fit before predict."
        assert x_predict.shape[1] == len(self.coef_), "the feature number of x_predict must be equal to x_train"

        proba = self.predict_proba(x_predict)
        return np.array(proba >= 0.5, dtype='int')

    def score(self, x_test, y_test):
        """ 根据测试数据集 x_test 和 y_test 确定当前模型的准确度"""
        y_predict = self.predict(x_test)
        return accuracy_score(y_test, y_predict)

    def __repr__(self):
        return "LogisticRegression()"

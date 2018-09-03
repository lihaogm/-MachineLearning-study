import numpy as np
import matplotlib.pyplot as plt


class PCA:

    def __init__(self, n_components):
        """ 初始化PCA"""
        assert n_components >= 1, "n_components must be valid."
        self.n_components = n_components
        # 获取的主成分矩阵
        self.components_ = None

    def fit(self, xx, eta=0.01, n_iters=1e4):
        """ 获得数据集xx的前n个主成分"""
        assert self.n_components <= xx.shape[1], "n_components must be greater than the feature num of xx."

        def demean(xx):
            return xx - np.mean(xx, axis=0)

        def f(w, xx):
            """ 函数f """
            return np.sum((xx.dot(w) ** 2)) / len(xx)

        def df(w, xx):
            """ 函数f的梯度 """
            return xx.T.dot(xx.dot(w)) * 2. / len(xx)

        def direction(w):
            """ 求w的单位向量"""
            return w / np.linalg.norm(w)

        def first_component(xx, initial_w, eta=0.01, n_iters=1e4, epsilon=1e-8):
            w = direction(initial_w)
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = df(w, xx)
                last_w = w
                w = w + eta * gradient
                w = direction(w)  # 注意1：每次求一个单位方向
                if (abs(f(w, xx) - f(last_w, xx)) < epsilon):
                    break
                cur_iter += 1
            return w

        xx_pca = demean(xx)
        self.components_ = np.empty(shape=(self.n_components, xx.shape[1]))
        for i in range(self.n_components):
            # 注意2：初始位置不能为0，不能从0向量开始
            initial_w = np.random.random(xx_pca.shape[1])
            w = first_component(xx_pca, initial_w, eta, n_iters)
            self.components_[i, :] = w

            plt.scatter(xx_pca[:, 0], xx_pca[:, 1])
            plt.plot([0, w[0] * 30], [0, w[1] * 30], color='r')
            plt.show()

            # 去除前一个主成分
            xx_pca = xx_pca - xx_pca.dot(w).reshape(-1, 1) * w

        # 注意3：不能使用StandardScaler标准化数据（归一化）
        return self

    def transform(self, xx):
        """ 将给定的xx，映射到各个主成分分量中，降维 """
        assert xx.shape[1] == self.components_.shape[1]

        return xx.dot(self.components_.T)

    def inverse_transform(self, xx):
        """ 将给定的xx，反向映射回原来的特征空间"""
        assert xx.shape[1] == self.components_.shape[0]

        return xx.dot(self.components_)

    def __repr__(self):
        return "PCA(n_components=%d)" % self.n_components

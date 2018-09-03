import numpy as np
import matplotlib.pyplot as plt
import GradientDescent_pro.LinearRegression_gd

# 1 准备数据
m = 100000

x = np.random.normal(size=m)
y = x * 4. + 3. + np.random.normal(0, 3, size=m)
x = x.reshape(-1, 1)

# 2 随机梯度
slrgd = GradientDescent_pro.LinearRegression_gd.LinearRegressionGD()
slrgd.fit_sgd(x, y)
print(slrgd.intercept_)
print(slrgd.coef_)

import matplotlib.pyplot as plt
from sklearn import datasets
from comm_utils.model_selection import train_test_split
import LinearRegression_pro.SimpleLR

# 准备数据，波士顿房产数据
boston = datasets.load_boston()
print(boston.feature_names)
x = boston.data[:, 5]  # 只使用房间数量这个特征
print(x.shape)
y = boston.target
print(y.shape)
# 绘制图
plt.scatter(x, y)
plt.show()

# 去除 y 为50的特殊值
x = x[y < 50.0]
y = y[y < 50.0]

plt.scatter(x, y)


# 使用简单线性回归法

# 1 拆分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y)
# 2 拟合
slr = LinearRegression_pro.SimpleLR.SimpleLinearRegression()
slr.fit(x_train, y_train)
print(slr.score(x_test, y_test))
# 3 绘制出线性关系
plt.plot(x_test, slr.predict(x_test), color="red")
plt.show()

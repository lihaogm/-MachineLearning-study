from sklearn import datasets
from comm_utils.model_selection import train_test_split
import LinearRegression_pro.LinearRegression

# 准备数据，波士顿房产数据
boston = datasets.load_boston()

x = boston.data
y = boston.target

x = x[y < 50.0]
y = y[y < 50.0]

print(x.shape)
print(y.shape)


# 使用简单线性回归法

# 1 拆分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y)
# 2 拟合
slr = LinearRegression_pro.LinearRegression.LinearRegression()
slr.fit_normal(x_train, y_train)
print(slr.score(x_test, y_test))

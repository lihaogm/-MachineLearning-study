import numpy as np
import matplotlib.pyplot as plt
from PCA_pro.PCA import PCA

# 1 准备数据
xx = np.empty((100, 2))
xx[:, 0] = np.random.uniform(0., 100., size=100)
xx[:, 1] = 0.75 * xx[:, 0] + 3. + np.random.normal(0, 10., size=100)

plt.scatter(xx[:, 0], xx[:, 1])
plt.show()

# 2 求主成分和其他成分
mpca2 = PCA(2)
mpca2.fit(xx)
print(mpca2.components_)

# 3 降维
xx_reduction = mpca2.transform(xx)
print("xx_reduction.shape:", xx_reduction.shape)

# 恢复降维,损失了信息
xx_restore = mpca2.inverse_transform(xx_reduction)
print("xx_restore.shape", xx_restore.shape)

plt.scatter(xx[:, 0], xx[:, 1], color='g')
plt.scatter(xx_restore[:, 0], xx_restore[:, 1], color='r', alpha=0.5)
plt.show()

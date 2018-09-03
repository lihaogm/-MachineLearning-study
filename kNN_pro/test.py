import numpy as np
from comm_utils.model_selection import train_test_split
import kNN_pro.my_feature_scaling

if __name__ == '__main__':
    raw_data_x = [[1, 2],
                  [1, 0],
                  [1, 1],
                  [3, 1],
                  [3, 3],
                  [4, 4]]
    raw_data_y = [0, 0, 0, 1, 1, 1]

    # 转换成numpy数组
    x_src = np.array(raw_data_x)
    y_src = np.array(raw_data_y)

    # 目标值
    xs = np.array([[3, 3.5]])

    # 将原始数据集分为训练集与测试集
    x_train, x_test, y_train, y_test = train_test_split(x_src, y_src)

    # 数值归一化
    my_scaler = kNN_pro.my_feature_scaling.StandardScaler()
    my_scaler.fit(x_train)
    print("训练集的均值方差归一化为：%s" % my_scaler.transform(x_train))
    print("测试集的均值方差归一化为：%s" % my_scaler.transform(x_test))

    # kNN求分类值
    kNNobj = kNN_pro.kNN.KNNClassfier(k=3)
    kNNobj.fit(x_train, y_train)
    print("目标值的预测分类是：%s" % kNNobj.predict(xs))
    print("训练模型的准确度为：%s" % kNNobj.score(x_test, y_test))

    # 根据归一化值求目标值分类
    kNNobj.fit(my_scaler.transform(x_train),y_train)
    print(kNNobj.predict(xs))
    print(kNNobj.score(my_scaler.transform(x_test),y_test))

    # 寻找最佳的k
    best_score = 0.0
    best_k = -1
    for ks in range(1, 6):
        knn_clf = kNN_pro.kNN.KNNClassfier(k=ks)
        knn_clf.fit(x_train, y_train)
        score = knn_clf.score(x_test, y_test)
        if score > best_score:
            best_k = ks
            best_score = score

    print("best_k=", best_k)
    print("best_score=", best_score)

#线性回归 波士顿房价预测
from sklearn.datasets import load_boston#数据
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor,Ridge
from sklearn.metrics import mean_squared_error
def linear1():
    """
    正规方程的优化方法对波士顿房价进行预测
    :return:
    """
    # 1）获取数据
    boston = load_boston()
    print("特征数量：\n", boston.data.shape)
    # print('特征名：',boston.feature_names)
    # print('特征值：',boston.data)
    # print(boston)
    # 2）划分数据集
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)

    # 3）标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4）预估器   线性回归
    estimator = LinearRegression()
    estimator.fit(x_train, y_train)
    # 5）得出模型
    print("正规方程-权重系数为：\n", estimator.coef_)
    print("正规方程-偏置为：\n", estimator.intercept_)

    # 6）模型评估
    y_predict = estimator.predict(x_test)
    print("预测房价：\n", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("正规方程-均方误差为：\n", error)


def linear2():
    """
    梯度下降的优化方法对波士顿房价进行预测
    :return:
    """
    # 1）获取数据
    boston = load_boston()
    print("特征数量：\n", boston.data.shape)

    # 2）划分数据集
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)

    # 3）标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4）预估器    随机梯度下降    learning_rate学习率的算法  eta0学习率   max_iter迭代次数   正则化-惩罚
    #eta0 的值过小会增大很多步骤再达到最低限度（增加时间），过大则会导致不准
    #alpha：值越大，正则化越强。当学习率设为“最优”时，也用于计算学习率。
    estimator = SGDRegressor(learning_rate="constant",eta0=0.01, max_iter=10000, penalty="l1")
    estimator.fit(x_train, y_train)

    # 5）得出模型
    print("梯度下降-权重系数为：\n", estimator.coef_)
    print("梯度下降-偏置为：\n", estimator.intercept_)

    # 6）模型评估
    y_predict = estimator.predict(x_test)
    print("预测房价：\n", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("梯度下降-均方误差为：\n", error)

def linear3():
    """
    岭回归对波士顿房价进行预测
    :return:
    """
    # 1）获取数据
    boston = load_boston()
    print("特征数量：\n", boston.data.shape)

    # 2）划分数据集
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)

    # 3）标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    #4）预估器
    estimator = Ridge(alpha=0.5, max_iter=10000)
    estimator.fit(x_train, y_train)


    # 5）得出模型
    print("岭回归-权重系数为：\n", estimator.coef_)
    print("岭回归-偏置为：\n", estimator.intercept_)

    # 6）模型评估
    y_predict = estimator.predict(x_test)
    print("预测房价：\n", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("岭回归-均方误差为：\n", error)

if __name__ == '__main__':
    linear1()
    # linear2()
    # linear3()

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
def decision_iris():
    """
    用决策树对鸢尾花进行分类
    :return:
    """
    # 1）获取数据集
    iris = load_iris()

    # 2）划分数据集  random_state控制随机状态。为了保证程序每次运行都分割一样的训练集合测试集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)

    # 3）决策树预估器   信息增益entropy
    estimator = DecisionTreeClassifier(criterion="entropy")
    estimator.fit(x_train, y_train)

    # 4）模型评估
    # 方法1：直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print('y_test:\n',y_test)
    print("直接比对真实值和预测值:\n", y_test == y_predict)

    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为：\n", score)

    # 可视化决策树  能导出DOT格式
    export_graphviz(estimator, out_file="iris_tree.dot", feature_names=iris.feature_names)

    return None



def titanic():
    # 1、获取数据   访问不了了
    path = "http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt"
    titanic = pd.read_csv(path)
    # 筛选特征值和目标值
    x = titanic[["pclass", "age", "sex"]]
    y = titanic["survived"]
    # 2、数据处理
    # 1）缺失值处理
    x["age"].fillna(x["age"].mean(), inplace=True) #填充平均值
    # 2) 转换成字典
    x = x.to_dict(orient="records")
    # 3、数据集划分
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)
    # 4、字典特征抽取
    transfer = DictVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 3）决策树预估器
    estimator = DecisionTreeClassifier(criterion="entropy", max_depth=8)
    estimator.fit(x_train, y_train)

    # 4）模型评估
    # 方法1：直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("直接比对真实值和预测值:\n", y_test == y_predict)

    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为：\n", score)

    # 可视化决策树
    export_graphviz(estimator, out_file="titanic_tree.dot", feature_names=transfer.get_feature_names())

def suijisenlin_demo():
    # 1、获取数据  下载不了
    # path = "http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt"
    path = 'titanic.csv'
    titanic = pd.read_csv(path)
    # 筛选特征值和目标值
    x = titanic[["pclass", "age", "sex"]]
    y = titanic["survived"]
    # 2、数据处理
    # 1）缺失值处理
    x["age"].fillna(x["age"].mean(), inplace=True)
    # 2) 转换成字典
    x = x.to_dict(orient="records")#转换后list形式  split、series、list字典  dict默认字典
    # 3、数据集划分
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)
    # 4、字典特征抽取
    transfer = DictVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    #随机森林预估器
    estimator = RandomForestClassifier()
    # 加入网格搜索与交叉验证
    # 参数准备
    param_dict = {"n_estimators": [120, 200, 300, 500, 800, 1200], "max_depth": [5, 8, 15, 25, 30]}
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=3)
    estimator.fit(x_train, y_train)

    # 5）模型评估
    # 方法1：直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("y_test:\n", y_test)
    print("直接比对真实值和预测值:\n", y_test == y_predict)

    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为：\n", score)

    # 最佳参数：best_params_
    print("最佳参数：\n", estimator.best_params_)
    # 最佳结果：best_score_
    print("最佳结果：\n", estimator.best_score_)
    # 最佳估计器：best_estimator_
    print("最佳估计器:\n", estimator.best_estimator_)
    # 交叉验证结果：cv_results_
    print("交叉验证结果:\n", estimator.cv_results_)

if __name__ == '__main__':
    # decision_iris()
    # titanic()
    suijisenlin_demo()
    pass

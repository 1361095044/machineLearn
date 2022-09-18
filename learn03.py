from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
def knn_iris():
    """
    用knn算法对鸢尾花进行分类
    :return:
    """
    #1.获取数据
    iris = load_iris()
    #2.划分数据集
    x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,random_state=6)
    #3.特征工程：标准化
    tran = StandardScaler()
    x_train = tran.fit_transform(x_train)
    x_test = tran.transform(x_test)
    #4.knn算法预估器
    est = KNeighborsClassifier(n_neighbors=7)
    est.fit(x_train,y_train)
    #5.模型评估
    #方法一：直接比对真实值和预测值
    y_predic = est.predict(x_test)
    print('预测值：',y_predic)
    print('真实值：',y_test)
    #方法二：计算准确率
    score = est.score(x_test,y_test)
    print(score)

def knn_iris_gscv():
    """
    用knn算法对鸢尾花进行分类,添加网格搜索和交叉验证
    :return:
    """
    #1.获取数据
    iris = load_iris()
    #2.划分数据集
    x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,random_state=6)
    #3.特征工程：标准化
    tran = StandardScaler()
    x_train = tran.fit_transform(x_train)
    x_test = tran.transform(x_test)
    #4.knn算法预估器
    est = KNeighborsClassifier()
    #加入网格搜索和交叉验证
    param_dict = {'n_neighbors':[1,3,5,7,9,11]}
    est = GridSearchCV(est,param_grid=param_dict,cv=10)#cv:k折交叉验证 10就是9用于训练1用于测试 cv也可以为空
    est.fit(x_train,y_train)
    #5.模型评估
    #方法一：直接比对真实值和预测值
    y_predic = est.predict(x_test)
    print('预测值：',y_predic)
    print('真实值：',y_test)
    #方法二：计算准确率
    score = est.score(x_test,y_test)
    print(score)
    # 最佳参数
    print("最佳参数:\n", est.best_params_)
    # 最佳结果
    print("最佳结果:\n", est.best_score_)
    # 最佳估计器
    print("最佳估计器:\n", est.best_estimator_)
    # 交叉验证结果
    print("交叉验证结果:\n", est.cv_results_)

if __name__ == '__main__':
    #knn算法
    knn_iris()
    print('-------------')
    #改进版knn算法
    knn_iris_gscv()
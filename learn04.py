import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
#facebook的一个案例
#获取数据
data = pd.read_csv('E:\\BaiduNetdiskDownload\\day2\\02-代码\\FBlocation\\train.csv',nrows=24000)
print(data)
#数据处理
time = pd.to_datetime(data['time'],unit='s')
date = pd.DatetimeIndex(time)
# print(time)
data['day'] = date.day
data['weekday'] = date.weekday
data['hour'] = date.hour
x = data[['x','y','accuracy','day','weekday','hour']]
y = data['place_id']
print(x)
print('-------------')
print(y)
print('-------------')
x_train,x_test,y_train,y_test = train_test_split(x,y)
tran = StandardScaler()
x_train = tran.fit_transform(x_train)
x_test = tran.transform(x_test)
#4.knn算法预估器
est = KNeighborsClassifier()
#加入网格搜索和交叉验证
param_dict = {'n_neighbors':[1,3,5,7,9]}
est = GridSearchCV(est,param_grid=param_dict,cv=2)
est.fit(x_train,y_train)
# 5.模型评估
# 方法一：直接比对真实值和预测值
y_predic = est.predict(x_test)
print('预测值：', y_predic)
print('真实值：', y_test)
# 方法二：计算准确率
score = est.score(x_test, y_test)
print(score)
# 最佳参数
print("最佳参数:\n", est.best_params_)
# 最佳结果
print("最佳结果:\n", est.best_score_)
# 最佳估计器
print("最佳估计器:\n", est.best_estimator_)
# 交叉验证结果
print("交叉验证结果:\n", est.cv_results_)
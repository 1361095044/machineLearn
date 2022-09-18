#特征预处理
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
def minmax_demo():
    #归一化
    #获取数据
    data = pd.read_csv("dating.txt")
    data = data.iloc[:, :3]  # 只取数据前三列
    print("data=\n", data)
    #实例化一个转换器类
    transfer = MinMaxScaler()  # 默认0-1
    data_new = transfer.fit_transform(data)
    print("data_new=\n", data_new)

def stand_demo():
    #标准化
    #获取数据
    data = pd.read_csv("dating.txt")
    data = data.iloc[:, :3]  # 只取数据前三列
    print("data=\n", data)
    #实例化一个转换器类
    transfer = StandardScaler()
    data_new = transfer.fit_transform(data)
    print("data_new=\n", data_new)

def variance_demo():
    """
    过滤低方差特征
    :return:
    """
    # 1、获取数据
    data = pd.read_csv("factor_returns.csv")
    data = data.iloc[:, 1:-2]
    print("data:\n", data)
    # 2、实例化一个转换器类
    transfer = VarianceThreshold(threshold=10)#阈值  删除方差值小于设定的方差值的特征
    # 3、调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new, data_new.shape)
    # 计算某两个变量之间的相关系数
    r1 = pearsonr(data["pe_ratio"], data["pb_ratio"])
    print("pe_ratio与pb_ratio相关系数：\n", r1)
    #statistic反应正负相关
    #pvalue值表示显著性，一般来说，小于0.05的话即便相关系数很小，也很可能是相关的
    r2 = pearsonr(data['revenue'], data['total_expense'])
    print("revenue与total_expense之间的相关性：\n", r2)

def pca_demo():
    #主成分分析   pca降维
    data = [[2,8,4,5],
            [6,3,0,8],
            [5,4,9,1]]
    transfer = PCA(n_components=3) #降维成两个特征 整数就是降维成几个特征 小数就是保留百分之多少的信息
    data_new = transfer.fit_transform(data)
    print("data_new:\n",data_new)

if __name__ == '__main__':
    # #归一化
    # minmax_demo()
    #标准化
    # stand_demo()
    #过滤低方差特征
    # variance_demo()
    #pca降维
    pca_demo()
    pass
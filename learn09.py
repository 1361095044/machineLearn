import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def lunkuoxishu():
    # 1、获取数据  百度网盘下载过   数据量有点大
    order_products = pd.read_csv("E:\\BaiduNetdiskDownload\\02-代码\\instacart\\order_products__prior.csv",nrows=80000)
    products = pd.read_csv("E:\\BaiduNetdiskDownload\\02-代码\\instacart\\products.csv",nrows=80000)
    orders = pd.read_csv("E:\\BaiduNetdiskDownload\\02-代码\\instacart\\orders.csv",nrows=80000)
    aisles = pd.read_csv("E:\\BaiduNetdiskDownload\\02-代码\\instacart\\aisles.csv",nrows=80000)
    # 2、合并表
    # order_products__prior.csv：订单与商品信息
    # 字段：order_id, product_id, add_to_cart_order, reordered
    # products.csv：商品信息
    # 字段：product_id, product_name, aisle_id, department_id
    # orders.csv：用户的订单信息
    # 字段：order_id,user_id,eval_set,order_number,….
    # aisles.csv：商品所属具体物品类别
    # 字段： aisle_id, aisle

    # 合并aisles和products aisle和product_id
    tab1 = pd.merge(aisles, products, on=["aisle_id", "aisle_id"])
    tab2 = pd.merge(tab1, order_products, on=["product_id", "product_id"])
    tab3 = pd.merge(tab2, orders, on=["order_id", "order_id"])
    # print(tab3,tab2,tab1)
    # print(tab2.keys())
    # print(orders)
    # 3、找到user_id和aisle之间的关系
    table = pd.crosstab(tab3["user_id"], tab3["aisle"])#第一个参数是列, 第二个参数是行. 还可以添加第三个参数
    data = table[:800]
    print(data)
    # 4、PCA降维
    # 1）实例化一个转换器类
    transfer = PCA(n_components=0.95)

    # 2）调用fit_transform
    data_new = transfer.fit_transform(data)
    print(data_new.shape)
    # 预估器流程
    estimator = KMeans(n_clusters=3)
    estimator.fit(data_new)
    y_predict = estimator.predict(data_new)
    y_predict[:300]
    # 模型评估-轮廓系数
    print(silhouette_score(data_new, y_predict))

if __name__ == '__main__':
    lunkuoxishu()

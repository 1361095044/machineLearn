from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
def dataset_demo():
    """
    数据集使用   鸢yuan尾花
    :return:
    """
    iris = load_iris()
    print('数据集：\n',iris)
    print('查看数据集描述：\n',iris['DESCR'])
    print('查看特征值名字:\n',iris.feature_names)
    print('查看特征值:\n',iris.data)
    #数据集划分
    x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target)
    print('训练集的特征值：\n:',x_train,x_train.shape)

def dict_demo():
    """
    字典特征提取
    :return:
    """
    data = [
        {'city':'北京','temp': 20,'scale':'大型城市'},
        {'city': '长沙', 'temp': 30,'scale':'中型城市'},
        {'city': '上海', 'temp': 40,'scale':'大型城市'},
        # {'city': '哈尔滨', 'temp': 11,'scale':'大型城市'},
    ]
    #实例化一个转换器类
    transfer =  DictVectorizer(sparse=False)#sparse 稀疏矩阵
    #调用fit_transform
    data_new = transfer.fit_transform(data)
    print(data_new)
    print('特征名字:' , transfer.get_feature_names_out())

def txet_demo():
    """
      文本特征提取
      :return:
      """
    data = ['life is short,i like like python',
            'life is long,i dislike python']
    # 实例化一个转换器类
    stop_words_list = ['is']#停用词参数为列表list
    transfer = CountVectorizer(stop_words=stop_words_list)
    #调用fit_transform
    data_new = transfer.fit_transform(data)
    print('特征名字:' , transfer.get_feature_names_out())
    print(transfer.vocabulary_)  # 返回特征名字所在位置
    print(data_new.toarray())  # .toarray()稀疏矩阵转换回二维数组
    print(data_new)#稀疏矩阵

def cut_word(text):#分词
    return " ".join(list(jieba.cut(text)))#字符串

def txet_demo_CN():
    """
      中文文本特征提取
      引用jieba库
      """
    data = ['退一步海阔天空',
            '一寸光阴一寸金']
    data_new=[]
    for sent in data:
        data_new.append(cut_word(sent))
        print(sent)
    print(data_new)
    # 实例化一个转换器类
    transfer = CountVectorizer()#stop_words=停用词
    #调用fit_transform
    data_new = transfer.fit_transform(data_new)
    print(data_new.toarray())#.toarray()稀疏矩阵转换回二维数组
    print('特征名字:' , transfer.get_feature_names_out())

def txet_demo_tfidf():
    """
      中文文本特征提取
      引用jieba库
      """
    data = ['退一步海阔天空',
            '一寸光阴一寸金']
    data_new=[]
    for sent in data:
        data_new.append(cut_word(sent))
    # print(data_new)
    # 实例化一个转换器类
    transfer = TfidfVectorizer()#stop_words=停用词
    #调用fit_transform
    data_new = transfer.fit_transform(data_new)
    print(data_new.toarray())#.toarray()稀疏矩阵转换回二维数组
    print('特征名字:' , transfer.get_feature_names_out())

if __name__ == '__main__':
    #数据集的使用
    # dataset_demo()
    #字典特征提取
    # dict_demo()
    #文本特征提取
    # txet_demo()
    #中文文本特征提取
    # txet_demo_CN()
    #tf-idf特征提取
    txet_demo_tfidf()
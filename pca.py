# *- coding:utf-8 -*
# 调用spark内置的pca算法对机器学习实战中的第十三章数据集进行降维处理
from numpy import array
from pyspark import SparkContext
from pyspark.mllib.feature import PCA
from pyspark.mllib.linalg import Vectors

if __name__ == '__main__':
    sc = SparkContext()
    tmpdatas = sc.textFile('pcaTestSet.txt')
    datas = tmpdatas.map(lambda line: Vectors.dense(array([float(line.split('\t')[0]), float(line.split('\t')[1])])))
    print datas.collect()[0]

    # 将输入降维成1维数据，并测试降维模型的准确性
    model = PCA(1).fit(datas)
    transforms = model.transform(datas)
    print transforms.collect()[0], array(transforms.collect()).shape

    # 测试输入[10.235186,11.321997]之后的降维值
    print model.transform(array([10.235186,11.321997]))
    sc.stop()

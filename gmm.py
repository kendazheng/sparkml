# *- coding:utf-8 -*
# 调用spark内部的kmeans算法实现完成机器学习实战中的第十章示例
from numpy import array
from pyspark import SparkContext
from pyspark.mllib.clustering import GaussianMixture


if __name__ == '__main__':
    sc = SparkContext()
    datas = sc.textFile('testSet.txt')
    clusters_num = 4
    parseData = datas.map(lambda x: array([float(y) for y in x.split('\t')]))
    model = GaussianMixture.train(parseData, clusters_num, maxIterations=10)
    clusters = [[] for i in range(clusters_num)]
    labels = model.predict(parseData).collect()
    nums = len(labels)
    for i in xrange(nums):
        clusters[labels[i]].append(parseData.collect()[i])
    print clusters
    sc.stop()

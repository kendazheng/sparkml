# -*- coding:utf-8 -*-
""""
Program: KMeans
Description: 调用spark内置的Kmeans算法示例 
Author: zhenglei - zhenglei@shinezone.com
Date: 2016-01-14 13:39:46
Last modified: 2016-01-14 13:40:24
Python release: 2.7
"""
# 调用spark内部的kmeans算法实现完成机器学习实战中的第十章示例
from numpy import array
from pyspark import SparkContext
from pyspark.mllib.clustering import KMeans


if __name__ == '__main__':
    sc = SparkContext()
    datas = sc.textFile('testSet.txt')
    clusters_num = 4
    parseData = datas.map(lambda x: array([float(y) for y in x.split('\t')]))
    model = KMeans.train(parseData, clusters_num, maxIterations=10,
                         runs=10, initializationMode='random')
    clusters = [[] for i in range(clusters_num)]
    labels = model.predict(parseData).collect()
    nums = len(labels)
    for i in xrange(nums):
        clusters[labels[i]].append(parseData.collect()[i])
    print clusters
    sc.stop()

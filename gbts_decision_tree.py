# -*- coding:utf-8 -*-
""""
Program: GradientBoostedTrees
Description: 调用spark内置的梯度平滑树示例
Author: zhenglei - zhenglei@shinezone.com
Date: 2016-01-14 13:37:09
Last modified: 2016-01-14 13:38:49
Python release: 2.7
"""
from numpy import array
from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import GradientBoostedTrees

sc = SparkContext()

result = {1.0: 'yes', 0.0: 'no'}

# 机器学习实战第三章中的鱼类归属数据源
data = [
    LabeledPoint(1, [1, 1]),
    LabeledPoint(1, [1, 1]),
    LabeledPoint(0, [1, 0]),
    LabeledPoint(0, [0, 1]),
    LabeledPoint(0, [0, 1])
]

rdd = sc.parallelize(data)

print '------------------------------------'
print rdd.collect()
print '------------------------------------'
model = GradientBoostedTrees.trainClassifier(rdd, categoricalFeaturesInfo={}, numIterations=3)
# print(model)

print '********************************************************'
print(model.toDebugString())
print "test [1,0]: %s" % (result[model.predict(array([1, 0]))])
print "test [1,1]: %s" % (result[model.predict(array([1, 1]))])
print "test [0,0]: %s" % (result[model.predict(array([0, 0]))])
print '********************************************************'
sc.stop()

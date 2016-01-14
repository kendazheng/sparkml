# -*- coding:utf-8 -*-
""""
Program:als.py
Description:sparl内置als算法调用
Author: zhenglei - zhenglei@shinezone.com
Date: 2016-01-14 12:56:53
Last modified: 2016-01-14 14:45:34
Python release: 2.7
"""
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, Rating


if __name__ == '__main__':
    sc = SparkContext()
    data = sc.textFile("alsTest.data")
    ratings = data.map(lambda l: l.split(',')).map(
        lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))
    print ratings.collect()
    rank = 10
    numIterations = 10
    # 训练模型, rank是隐含影响特征，一般是初始为5-10，然后递增查看训练效果，直到效果不再改变，确定rank的值
    model = ALS.train(ratings, rank, numIterations)
    testdata = ratings.map(lambda p: (p[0], p[1]))
    # 对输入的数据进行预测
    predictions = model.predictAll(testdata).map(
        lambda r: ((r[0], r[1]), r[2]))
    print predictions.collect()
    # 获取所有预测及测试数据
    ratesAndPreds = ratings.map(lambda r: (
        (r[0], r[1]), r[2])).join(predictions)
    print ratesAndPreds.collect()
    # 计算误差
    MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()
    print("Mean Squared Error = " + str(MSE))

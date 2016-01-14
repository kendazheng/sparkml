# -* coding:utf-8 *-
""""
Program:als.py
Description:sparl内置als算法调用
Author: zhenglei - zhenglei@shinezone.com
Date: 2016-01-14 12:56:53
Last modified: 2016-01-14 13:10:33
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
    model = ALS.train(ratings, rank, numIterations)
    testdata = ratings.map(lambda p: (p[0], p[1]))
    predictions = model.predictAll(testdata).map(
        lambda r: ((r[0], r[1]), r[2]))
    ratesAndPreds = ratings.map(lambda r: (
        (r[0], r[1]), r[2])).join(predictions)
    MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()
    print("Mean Squared Error = " + str(MSE))

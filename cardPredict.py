# -*- coding:utf-8 -*-
""""
Program:carPredict.py
Description:预测上海市车牌价格
Author: zhenglei - kendazheng@163.com
Date: 2016-01-17 14:18:30
Last modified: 2016-01-17 18:32:20
Python release: 2.7
"""
from numpy import array
from pyspark import SparkContext
from pyspark.mllib.regression import LinearRegressionWithSGD, LabeledPoint


def textParser(type):
    """
    type : 0 the lowest prices and 1 is the average price
    """
    datas = []
    lines = open('cards2.txt')
    for line in lines:
        features = line.strip().split('\t')
        datas.append(LabeledPoint(float(features[type]), features[2:-1]))
    return datas

if __name__ == '__main__':
    sc = SparkContext()
    datas = sc.parallelize(textParser(1))
    model = LinearRegressionWithSGD.train(datas, step=0.00000000174434, iterations=2000, regType='l2')
    # model = LinearRegressionWithSGD.train(datas, step=0.00000000175234766555555566666, iterations=5000, regType='l2')
    print '**' * 50
    print model.weights
    print model.intercept
    print model.predict(array([9409, 187533, 84500, 84572]))
    print '**' * 50
    valuesAndPreds = datas.map(lambda p: (p.label, model.predict(p.features)))
    print valuesAndPreds.collect(), valuesAndPreds.count()
    MSE = valuesAndPreds.map(lambda (v, p): (v - p) ** 2).reduce(lambda x, y: x + y) / valuesAndPreds.count()
    print("Mean Squared Error = " + str(MSE))
    # 2016.1  9409    82200   82352   187533
    sc.stop()

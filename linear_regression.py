# *- coding:utf-8 -*
# 调用Spark内置的LinearRegression算法对机器学习实战中的第八章鲍鱼年纪的预测
from numpy import array
from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD


def textParser():
    datas = []
    lines = open('abalone.txt').readlines()
    for line in lines:
        tmp = line.strip().split('\t')
        datas.append(LabeledPoint(tmp[-1], tmp[1:-1]))
    return datas

if __name__ == '__main__':
    sc = SparkContext()
    datas = sc.parallelize(textParser())
    model = LinearRegressionWithSGD.train(datas, step=5, iterations=500, regType='l2')

    # 计算预测模型与训练值得方差
    prevals = datas.map(lambda p: (p.label, model.predict(p.features)))
    MSE = prevals.map(lambda (v, p): (v - p) **
                      2).reduce(lambda x, y: x + y) / prevals.count()
    print u'方差:', str(MSE)
    print u'测试数据值为：', datas.collect()[0]
    print u'模型预期数据：', model.predict(array(datas.collect()[0].features))
    sc.stop()

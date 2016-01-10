# -* coding:utf-8 *-
# 调用Spark中的logistic regression算法，完成机器学习实战中的第五章对马的病死情况的预测
from numpy import array
from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithLBFGS


def textParse(line):
    line_list = line.split('\t')
    return line_list[-1], line_list[0:-1]


def generateDatas(lines):
    datas = []
    for item in lines:
        label, text_list = textParse(item.strip())
        datas.append(LabeledPoint(label, text_list))
    return datas


def getDatas():
    datas = []
    training_datas = open('horseColic/horseColicTraining.txt').readlines()
    datas.extend(generateDatas(training_datas))
    test_datas = open('horseColic/horseColicTest.txt')
    datas.extend(generateDatas(test_datas))
    return datas

if __name__ == '__main__':
    sc = SparkContext()
    datas = sc.parallelize(getDatas())
    print datas.collect()[0]
    print datas.collect()[3]
    model = LogisticRegressionWithLBFGS.train(datas)
    print '--' * 50
    print model.predict(array([2.0, 1.0, 38.5, 54.0, 20.0, 0.0, 1.0, 2.0, 2.0, 3.0, 4.0, 1.0, 2.0, 2.0, 5.9, 0.0, 2.0, 42.0, 6.3, 0.0, 0.0]))
    print model.predict(array([1.0, 1.0, 37.0, 56.0, 24.0, 3.0, 1.0, 4.0, 2.0, 4.0, 4.0, 3.0, 1.0, 1.0, 0.0, 0.0, 0.0, 35.0, 61.0, 3.0, 2.0]))
    print '--' * 50
    sc.stop()

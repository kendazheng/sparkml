# -* coding:utf-8 *-
# 调用Spark内部的SVM算法，对机器学习实战中的底五章的根据马的病症预测马的死亡性
from numpy import array
from pyspark import SparkContext
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.regression import LabeledPoint


def textParse(line):
    cols = line.split('\t')
    return cols[-1], cols[0:-1]


def getDatas():
    datas = []
    lines = open('horseColic/horseColicTraining.txt').readlines()
    for item in lines:
        label, line_list = textParse(item.strip())
        datas.append(LabeledPoint(label, line_list))
    return datas

if __name__ == '__main__':
    sc = SparkContext()
    datas = sc.parallelize(getDatas())
    print '**' * 50
    print datas.collect()[0]
    print '**' * 50
    model = SVMWithSGD.train(datas, iterations=100)
    # print model.toDebugString()
    print '--' * 50
    print model.predict(array([2.0, 1.0, 38.5, 54.0, 20.0, 0.0, 1.0, 2.0, 2.0, 3.0, 4.0, 1.0, 2.0, 2.0, 5.9, 0.0, 2.0, 42.0, 6.3, 0.0, 0.0]))
    print '--' * 50
    sc.stop()

# -* coding:utf-8 *-
from numpy import array
from pyspark import SparkContext
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint


def getDatas():
    datas = []
    for i in xrange(188):
        num = [] 
        lines = open('trainingdigits/0_%d.txt' %i).readlines()
        for line in lines:
            for i in line.strip():
                num.append(int(i))
        datas.append(num) 
    return datas

if __name__ == '__main__':
    sc = SparkContext()
    datas = getDatas()
    sc.stop()


# -* coding:utf-8 *-
from numpy import array
from pyspark import SparkContext
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.regression import LabeledPoint


if __name__ == '__main__':
    sc = SparkContext()
    sc.stop()

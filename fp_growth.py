# -*- coding:utf-8 -*-
""""
Program: FPGrowth
Description:调用spark内置的fpgrowth算法示例
Author: zhenglei - kendazheng@163.com
Date: 2016-01-14 13:36:09
Last modified: 2016-01-14 13:37:01
Python release: 2.7
"""

# 调用spark内置的fp-growth算法，实现机器学习实战中的第十二章示例
from pyspark import SparkContext
from pyspark.mllib.fpm import FPGrowth


if __name__ == '__main__':
    sc = SparkContext()
    tmpdatas = sc.textFile('kosarak.dat')
    datas = tmpdatas.map(lambda line: line.strip().split(' '))
    # tmpdatas = sc.textFile('/opt/spark-1.6.0/data/mllib/sample_fpgrowth.txt')
    # datas = tmpdatas.map(lambda line: line.strip().split(' '))
    model = FPGrowth.train(datas, minSupport=0.1)
    results = model.freqItemsets().collect()
    for item in results:
        print item
    sc.stop()

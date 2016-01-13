# -* coding:utf-8*-
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

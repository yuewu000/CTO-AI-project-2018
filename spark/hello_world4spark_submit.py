from pyspark import SparkContext
from pyspark import SparkConf
import numpy as np
import os

#set spark context

sc = SparkContext(conf=SparkConf().setMaster("yarn"))
dummy_RDD = sc.parallelize(np.array(range(6*4096)), 6)
print "the dummy RDD has totally {} elements in {} partitions".format(dummy_RDD.count(), dummy_RDD.getNumPartitions())
print "spark context is created successfully!"

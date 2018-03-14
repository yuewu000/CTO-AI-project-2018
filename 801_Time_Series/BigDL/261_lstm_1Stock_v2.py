# coding: utf-8

# # Time Series Forecasting with Stock Market Prices

# This is based on a sample job and extended to the following aspects:
# 1. Input longer time series
# 2. verify many-one prediction
# 3. to test many-many prediction in bigDL

#import matplotlib
#matplotlib.use('Agg')
#get_ipython().magic(u'pylab inline')
# save the bigDL model

import pandas as pd
import datetime as dt
import time
import math
import itertools

from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *
#import matplotlib.pyplot as plt
from pyspark import SparkContext
from pyspark import SparkConf
#from pyspark import create_spark_conf

#from matplotlib.pyplot import imshow


sc=SparkContext.getOrCreate(conf=create_spark_conf().setMaster("yarn"))
# sc=SparkContext.getOrCreate(conf=SparkConf().setMaster("yarn"))
# sc=SparkContext.getOrCreate(conf=create_spark_conf().setMaster("172.30.100.3").set("spark.driver.memory","64g")).set("spark.executor.memory","16g")
# sc=SparkContext.getOrCreate(conf=create_spark_conf().setMaster("local[38]").set("spark.driver.memory","76g"))

init_engine()
def denormalise_windows(data, minV, maxV):

    denormalised_data = [float((p + 0.5) * (maxV - minV) + minV) for p in data]

    return denormalised_data

def normalise_windows(window_data):
    normalised_data = []
    maxDmd = []
    minDmd = []
    for window in window_data:
        maxd = max(window)
        mind = min(window)
        maxDmd.append(maxd)
        minDmd.append(mind)
        if (maxd -mind) < math.exp(-10):
            if mind == 0:
                normalised_window = [(float(p)) for p in window]
            else:
                normalised_window = [(float(p)/float(mind + (maxd - mind) / 2) - 1) for p in window]
        else:
            normalised_window = [(float(p - mind) / float(maxd - mind) - 0.5) for p in window]
        normalised_data.append(normalised_window)
    return [maxDmd, minDmd, normalised_data]

def load_data(stockname, No, window, h, normalise_window):
    # transform for multi-input multi-outpout forecasting models

    e = pd.read_csv("../data/" + stockname + No + ".csv")

    list1 = list(e['Close'])

    sequence_length = window + h
    result = []
    n = int(e.shape[0])
    #nobs = n - h - window + 1
    #print ("input shape 0", e.shape[0], "shape 1", e.shape[1])
    for index in range(n - sequence_length):
        result.append(list1[index: index + sequence_length])

    if normalise_window:
        maxL, minL, result = normalise_windows(result)

    result = np.array(result)

    # no testing for short series
    row = round(1 * result.shape[0])
    train = result[:int(row), :]
    #np.random.shuffle(train)
    x_train = train[:, :-h]
    y_train = train[:, -h:]
    x_test = result[int(row):, :-h]
    y_test = result[int(row):, -h:]

    #x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    #x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return [maxL, minL, x_train, y_train]

#def build_model(layers=[input_size, hidden_size, output_size]):
def build_model(layers):
    model = Sequential()
    recurrent1 = Recurrent()
    recurrent1.add(LSTM(layers[0], layers[1]))
    drop = Dropout(0.2)
    recurrent2 = Recurrent()
    recurrent2.add(LSTM(layers[1], layers[1]))

    model.add(InferReshape([-1, layers[0]], True))
    model.add(recurrent1)
    model.add(drop)
    model.add(Echo())
    model.add(recurrent2)
    model.add(drop)
    model.add(Echo())
    model.add(Select(2, -1))
    model.add(Linear(layers[1], layers[2]))
    return model

#better to admitting variant epoch based on each case, here I use 800 for all items
seq_len = 22
fwd_len = 1
#valid_ratio = 0.1
No = '01'
epochs = 300
batch_size = 32
training_split = 0.8

# Create Spark RDD for the model input
inputName = 'AAPL_EndOfDay_'
No = ''
maxList, minList, x_train, y_train = load_data(inputName, No, seq_len, fwd_len, True)

print "the shape of x_train is {}".format(x_train.shape)
print "the shape of y_train is {}".format(y_train.shape)
print ('The shape of training input data is: {}; \nAnd shape of the training output data is: {}'.format(x_train.shape, y_train.shape))

# build the RDD for the whole data first then shuffle and split
x_2d = [i.reshape(-1,x_train.shape[1]) for i in x_train]
y_2d = [[i[0]] for i in y_train]
flatten_2d = list(itertools.chain(*x_2d))

# try without reshape first
# hists = np.array(flatten_hists, dtype='float').reshape([x_train.shape[0],x_train.shape[1]])
hists_2d = np.array(flatten_2d, dtype='float')
#hists_rdd = sc.parallelize(hists_2d)
zip_2d = zip(hists_2d, y_2d)
prep_rdd = sc.parallelize(zip_2d, 36)

# build sample RDD with feature/lable pair; here the features are series of length h (22), the label is of length 1 for single-output model
sample_rdd = prep_rdd.map(lambda (s1, s2) : Sample.from_ndarray(np.array(s1), np.array(s2)))

train_rdd, val_rdd = sample_rdd.randomSplit([training_split, 1-training_split])

# define some model parameters
n_input = 22    # input set no
n_hidden = 256  #hidden layer num of features
n_classes = 1    # output forecasted set no

# build the LSTM model architecture, train and save the model

layers = [n_input, n_hidden, n_classes]
rnn_model = build_model(layers)

# Create an Optimizer

#criterion = TimeDistributedCriterion(CrossEntropyCriterion())
criterion = MSECriterion()
"""
caindidate criterion
criterion = ClassNLLCriterion()
layer = TimeDistributedCriterion(criterion, True)
"""
optimizer = Optimizer(
    model=rnn_model,
    training_rdd=train_rdd,
    criterion=criterion,
    optim_method=Adam(),
    end_trigger=MaxEpoch(epochs),
    batch_size=batch_size)

# Set the validation logic
optimizer.set_validation(
    batch_size=batch_size,
    val_rdd=val_rdd,
    trigger=EveryEpoch(),
    val_method=[Loss(criterion)]
)

app_name='rnn-'+dt.datetime.now().strftime("%Y%m%d-%H%M%S")
train_summary = TrainSummary(log_dir='/tmp/bigdl_summaries',
                                     app_name=app_name)
train_summary.set_summary_trigger("Parameters", SeveralIteration(50))
val_summary = ValidationSummary(log_dir='/tmp/bigdl_summaries',
                                        app_name=app_name)
optimizer.set_train_summary(train_summary)
optimizer.set_val_summary(val_summary)
print "saving logs to ",app_name


#traing with 64 gb, no partition number
# import msvcrt as m
def wait():
    raw_input ("Please touch the Enter key to continue." )


#wait()
start_time = time.time()
# Boot training process
trained_model = optimizer.optimize()
print "*\n**\n***\n****\n*****\n****\n***\n**\n*\nOptimization Done. Total {} epochs finished in {} S".format(epochs, time.time()-start_time)
#wait()

trained_model.saveModel("./model/261_lstm_1Stock_v2a300e.bigdl", "./model/261_lstm_1Stock_v2a300e.bin", True) # save the model to local fs
#save to hdfs
                                                            




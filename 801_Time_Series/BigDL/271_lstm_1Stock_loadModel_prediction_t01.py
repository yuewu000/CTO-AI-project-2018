import pandas as pd
import datetime as dt
import time
import math
import itertools

from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *

from util import denormalise_windows, normalise_windows, load_data, build_model

# Build training and validation RDD

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

# Load the saved BigDL model and apply to the RDD to make prediction

# load the trained model
trained_model=Model.loadModel("./model/261_lstm_1Stock_v2a300e.bigdl", "./model/261_lstm_1Stock_v2a300e.bin")
print "The model is loaded successfully to the following model".format(trained_model)

# Make prediction (just show how to predict and deploy the predicted data, which is still a RDD)
pred4 = trained_model.predict(train_rdd)
print "*\n*\n*\n Let us see what we get grom the model:\n {} \n*\n*\n*\n*\n*\n totally {} elements" \
  .format(pred4.collect(), pred4.count())


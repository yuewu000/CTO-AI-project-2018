import pandas as pd
import datetime as dt
import time
import math
import itertools

from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *

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

        normalised_data.append(normalised_window)
    return [maxDmd, minDmd, normalised_data]

def load_data(stockname, No, window, h, normalise_window):
    # transform for multi-inout nulti-outpout forecasting models
#    basename = "input_"
#    index =3
#    f = open("./input/" + basename + str(index) + ".csv", 'rb').read()

    e = pd.read_csv("../data/" + stockname + No + ".csv")

    list1 = list(e['Close'])
    #fst = util.first_Positive(list1)
    #if (fst > window - 20): # keep at least 20 windows for training, or go smaller
    #    # smaller window to forecast shorter series, keep seq_len = fwd_len here, match the month frame
    #    window = max(2, 4*int((156 - fst)/3/4)) # keep seq_len positive
    #    h = window

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



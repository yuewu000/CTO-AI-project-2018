# -*- coding: utf-8 -*-

import os
import time
import warnings
import numpy as np
from numpy import newaxis
import pandas as pd
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import math
import util

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

                       
def input_prep(filename, inputName):
                      
    f = pd.read_csv(filename) 

    f = f.reset_index()
    new = f.transpose()
    new = new.reset_index() 
    new['weekOfYear'] = (new.index-2) % 52 + 1
    inprep = new.drop([0,1])
    col1 = [[]]
    for i in range(100):
        col1.append(["weekOfYear", i])
        
    for i in range(100):
        idx = i + 1
        inprep[col1[idx]].to_csv('./input/' + inputName + str(idx) + '.csv', header = ["weekOfYear", idx])
                      
def input_prep_stock(filename, inputName):
                      
    f = pd.read_csv(filename) 

    f = f.reset_index()
    new = f.transpose()
    new = new.reset_index() 
    new['weekOfYear'] = (new.index-2) % 52 + 1
    inprep = new.drop([0,1])
    col1 = [[]]
    for i in range(100):
        col1.append(["weekOfYear", i])
        
    for i in range(100):
        idx = i + 1
        inprep[col1[idx]].to_csv('./input/' + inputName + str(idx) + '.csv', header = ["weekOfYear", idx])     
             
def load_data(stockname, No, window, h, normalise_window):
    # transform for multi-inout nulti-outpout forecasting models
#    basename = "input_"
#    index =3
#    f = open("./input/" + basename + str(index) + ".csv", 'rb').read()
    
    e = pd.read_csv("./input/" + stockname + No + ".csv")
    
    list1 = list(e['Close'])
    fst = util.first_Positive(list1)
    if (fst > window - 20): # keep at least 20 windows for training, or go smaller
        # smaller window to forecast shorter series, keep seq_len = fwd_len here, match the month frame
        window = max(2, 4*int((156 - fst)/3/4)) # keep seq_len positive
        h = window
       
    sequence_length = window + h
    result = []
    n = int(e.shape[0])
    #nobs = n - h - window + 1
    #print ("input shape 0", e.shape[0], "shape 1", e.shape[1])    
    for index in range(fst, n - sequence_length):
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

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  

    return [fst, maxL, minL, x_train, y_train]
#return [x_train, y_train, x_test, y_test]

def load_data_demandsMUL(basename, index, window, h, normalise_window):
    # transform for multi-inout nulti-outpout forecasting models
#    basename = "input_"
#    index =3
#    f = open("./input/" + basename + str(index) + ".csv", 'rb').read()
    
    e = pd.read_csv(basename + str(index) + ".csv")
    
    list1 = list(e[str(index)])
    fst = util.first_Positive(list1)
    if (fst > window - 20): # keep at least 20 windows for training, or go smaller
        # smaller window to forecast shorter series, keep seq_len = fwd_len here, match the month frame
        window = max(2, 4*int((156 - fst)/3/4)) # keep seq_len positive
        h = window
       
    sequence_length = window + h
    result = []
    n = int(e.shape[0])
    #nobs = n - h - window + 1
    #print ("input shape 0", e.shape[0], "shape 1", e.shape[1])    
    for index in range(fst, n - sequence_length):
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

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  

    return [fst, maxL, minL, x_train, y_train]

def load_data_single(filename, seq_len, normalise_window):
    f = open(filename, 'rb').read()
    data = f.decode().split('\n')
    
    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    
    if normalise_window:
        maxL, minL, result = normalise_windows(result)

    result = np.array(result)

    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  

    return [x_train, y_train, x_test, y_test]


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

def normalise_windows4pos(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

def build_model(layers, activ):
    model = Sequential()

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation(activ))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("> Compilation Time : ", time.time() - start)
    return model

def predict_point_by_point(model, data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

def predict_sequence_full(model, data, window_size):
    #Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
    return predicted

def predict_sequences_multiple(model, data, window_size, prediction_len):
    #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs


# -*- coding: utf-8 -*-

import lstm
import time
import matplotlib.pyplot as plt
import util
import pandas as pd
from numpy import newaxis
import numpy as np
import math

#Main Run Thread

#if __name__=='__main__':

filename = '../data/aapl.us.txt'
inputName = 'aapl5min'

f = pd.read_csv(filename) 

f
f['Open']
f[['Open','Close']]
f[['Close','Open']]
#f = f.reset_index()
#new = f.transpose()
#new = new.reset_index() 
f[f['Date'] == '2017-12-07']

samplePerDay = 78

f['minuteOfDay'] = f.index % samplePerDay + 1

f['minuteOfDay']

936/78

#inprep = new.drop([0,1])
#col1 = [[]]
#for i in range(936):
#    col1.append(["weekOfYear", i])
    
#for i in range(100):
#    idx = i + 1
# No the input data in case we need to do batch forecasting for multiple stocks in the future
No = '01'

f[['minuteOfDay', 'Close']].to_csv('./input/' + inputName + No + '.csv', header = ["minuteOfDay", "Close"])    

f[['minuteOfDay', 'Close']]

global_start_time = time.time()
#generate 100 input file for each product, if not done yet
#lstm.input_prep("Forecasting_Exercise.csv", "input_") 
	      
#the following numpy matrix is for storing all the forcasting results
input_name = './input/' + inputName + No + '.csv'

input_name

#forecast_allts = np.zeros((100, 52))

#for i in range(1, 101, 1):
epochs = 300   #better to admitting variant epoch based on each case, here I use 800 for all items
seq_len = 78
fwd_len = 78
valid_ratio = 0.1
act_fnc = "linear"   # tanh is used for rerunning couple of over-trained series
print('> Loading data for stock ticker/tiemFreq ' + inputName + ' No.' + No + ' ... ')
  

import lstm
fp, maxList, minList, X_train, y_train = lstm.load_data(inputName, No, seq_len, fwd_len, True) # loading data and convert for training
"""
if (fp > seq_len - 20): # keep at least 20 windows for training, or go smaller
    # smaller window to forecast shorter series, keep seq_len = fwd_len here, match the month frame
    seq_len = max(2, 4*int((156 - fp)/3/4)) # keep seq_len positive
    fwd_len = seq_len
    if (seq_len == 2):
        Print("series is too short for LSTM training, skip this item No." + str(i))
        continue
   
print('> training item No.', str(i), ' of window ', str(seq_len), ', start training......')
if seq_len in [4, 8]:
    valid_ratio = 0.25
elif seq_len == 2:
    valid_ratio = 0.5
if seq_len <= 2:
    short.append(i)
    epochs = 20
    print("extremely short sequence for item No." +  str(i))
"""
X_train
X_train.shape  

maxList

nodes = seq_len * 2

#build LSTM model activation function = "linear" or "tanh"
model = lstm.build_model([1, nodes, nodes*2, fwd_len], act_fnc)
model.fit(
           X_train,
           y_train,
           batch_size=nodes*2,
           nb_epoch= epochs,
           validation_split=valid_ratio)

print('y shape', y_train.shape, '; X shape', X_train.shape)

curr = y_train[- 1]
#decurr = lstm.denormalise_windows(list(curr), minList[-1], maxList[-1])
p3 = model.predict(curr[newaxis,:, newaxis])
dep3 = lstm.denormalise_windows(list(p3[0]), minList[-1], maxList[-1])
#util.plot_results(dep3, decurr, seq_len, i)
p3
p3.shape
dep3
len(dep3)
type(p3)
type(dep3)


# Visualizations

inname1 = './input/aapl5min01.csv'

with open(inname1) as f:
    lines = (line for line in f if not line.startswith(','))
    inRaw = np.loadtxt(lines, delimiter=',', skiprows=1)

#fc2 = np.loadtxt(inname1, delimiter = ',')
inRaw

sizex = 10
sizey = 8 
n = 1
m = 1
i = 1

fig = plt.figure(figsize=(sizex, sizey), facecolor='white')
data0 = list(inRaw[:,2])
data1 = dep3
df10 = pd.DataFrame(data0, index = range(len(data0)), columns = ['ts'])
df11 = pd.DataFrame(data1, index = range(len(data0), len(data0)+len(data1)), columns = ['ts'])
ax = plt.subplot(n,m,i)
ax.set_title(inputName + " Stock Price Forecasting " + str(epochs) + " Epochs", fontsize = 18)
ax.plot(df10.index, df10.ts, label = "hist prices")  
ax.plot(df11.index, df11.ts, label = "pred prices")  
fig.subplots_adjust(hspace=hspce)
plt.xlabel('Stock price forecasting')
plt.legend()
plt.show()    


# study the raw data
f = pd.read_csv(filename) 

f
f['Open']
f[['Open','Close']]
f[['Close','Open']]
#f = f.reset_index()
#new = f.transpose()
#new = new.reset_index() 
f[f['Date'] == '2017-12-07']

samplePerDay = 78

f['minuteOfDay'] = f.index % samplePerDay + 1

f['minuteOfDay']

# overlay the forecasted curves with the historical price
# got to predict by a daily frame
type(f)
f.shape

days = int(math.floor(f.shape[0]/78 ))
days

#type(days) # make sure it is integer
preds = []
X_train.shape
minList
maxList
len(minList)
X_train[78][0]
y_train[0]
len(y_train)

minList.extend(minList[-78:])
maxList.extend(maxList[-78:])

preds = []
for i in range(days-2):
    currHist = y_train[i*78]
    #decurr = lstm.denormalise_windows(list(curr), minList[-1], maxList[-1])
    #currPred = list(model.predict(curr[newaxis,:, newaxis])[0])
    currPred = model.predict(currHist[newaxis,:, newaxis])
    dep3 = list(lstm.denormalise_windows(list(currPred[0]), minList[i*78 + 78], maxList[i*78 + 78]))
    preds.extend(dep3)

preds

sizex = 10
sizey = 8 
n = 1
m = 1
c = 1

fig = plt.figure(figsize=(sizex, sizey), facecolor='white')
data0 = list(inRaw[:,2])
len(data0)
data1 = preds
len(data1)
df10 = pd.DataFrame(data0, index = range(len(data0)), columns = ['ts'])
df11 = pd.DataFrame(data1, index = range(78*2, 78*2+len(data1)), columns = ['ts'])
ax = plt.subplot(n,m,c)
ax.set_title(inputName + " Stock Price Forecasting " + str(epochs) + " Epochs", fontsize = 18)
ax.plot(df10.index, df10.ts, label = "hist prices")  
ax.plot(df11.index, df11.ts, label = "pred prices")
for j in range(2, days):
    ax.axvline(x=j*78, c = 'green', ls = 'dashdot', lw = 0.5)
fig.subplots_adjust(hspace=hspce)
plt.xlabel('Stock price forecasting')
plt.legend()
plt.show()    


currHist = X_train[3]
    #decurr = lstm.denormalise_windows(list(curr), minList[-1], maxList[-1])
    predRaw =model.predict(curr[newaxis,:, newaxis])
    predRaw[0][0]
    currPred = list(model.predict(curr[newaxis,:, newaxis])[0])
    preds.extend(currPred)


preds
len(preds)
len(preds[1][0])

data0

for j in range(len(dep3)):
    forecast_allts[i-1, j] = dep3[j]
    
for j in range(len(dep3), 52, 1):
    forecast_allts[i-1, j] = 0

# write the raw LSTM forecasted to csv file, for further postprocessing
np.savetxt("raw_lstm_forecast.csv", forecast_allts, fmt='%.2f', delimiter = ',')

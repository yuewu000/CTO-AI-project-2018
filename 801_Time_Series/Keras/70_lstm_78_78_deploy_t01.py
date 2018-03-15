# -*- coding: utf-8 -*-
#generate profit based on the 78-78 forecasting model

import lstm
import time
import matplotlib.pyplot as plt
import util
import pandas as pd
from numpy import newaxis
import numpy as np
import math

import sys
print sys.path

#Main Run Thread

#if __name__=='__main__':

filename = '../data/aapl.us.txt'
inputName = 'aapl5min'

No = '01'
epochs = 30   #better to admitting variant epoch based on each case, here I use 800 for all items
seq_len = 78
fwd_len = 78
valid_ratio = 0.1
act_fnc = "linear"   # tanh is used for rerunning couple of over-trained series
print('> Loading data for stock ticker/tiemFreq ' + inputName + ' No.' + No + ' ... ')

#generate 100 input file for each product, if not done yet
#lstm.input_prep("Forecasting_Exercise.csv", "input_") 
	      
#the following numpy matrix is for storing all the forcasting results
input_name = './input/' + inputName + No + '.csv'

input_name

#forecast_allts = np.zeros((100, 52))

No = '01'
epochs = 30   #better to admitting variant epoch based on each case, here I use 800 for all items
seq_len = 78
fwd_len = 78
valid_ratio = 0.1
act_fnc = "linear"   # tanh is used for rerunning couple of over-trained series
print('> Loading data for stock ticker/tiemFreq ' + inputName + ' No.' + No + ' ... ')

fp, maxList, minList, X_train, y_train = lstm.load_data(inputName, No, seq_len, fwd_len, True) # loading data and convert for training

print (X_train.shape)

nodes = seq_len * 2

#build LSTM model activation function = "linear" or "tanh"
model = lstm.build_model([1, nodes, nodes*2, fwd_len], act_fnc)
global_start_time = time.time()
model.fit(
           X_train,
           y_train,
           batch_size=nodes*2,
           nb_epoch= epochs,
           validation_split=valid_ratio)

print ("The LSTM training is done in {} seconds".format(time.time() - global_start_time))
print('y shape', y_train.shape, '; X shape', X_train.shape)

curr = y_train[- 1]
x_curr = X_train[12,:,0]
x_curr
#decurr = lstm.denormalise_windows(list(curr), minList[-1], maxList[-1])
p3 = model.predict(curr[newaxis,:, newaxis])
dep3 = lstm.denormalise_windows(list(p3[0]), minList[-1], maxList[-1])

X_train[13, :, 0].shape


print "The shape of X_train and y_train are respectly {} and {}".format(X_train.shape, y_train.shape)
print "The length of the min and max lists is {}".format(len(minList)) 

# Define the functions to generate profit of percentage gain/loss
***************************************************************************
def gen_long_short_pos(model, x, h):
    if x is None or y is None or h <= 0:
        return []
    long_short_pos = []
    for i in range(0, len(x), h):
        curr = x[i, :, 0]
        pred = model.predict(curr[newaxis,:,newaxis])
        denm_pred = lstm.denormalise_windows(list(pred[0]), minList[i], maxList[i])
        denm_curr = lstm.denormalise_windows(list(curr), minList[i], maxList[i])
        pos = 1
        # calculate predicted potential profit
        pp = denm_pred[-1] - denm_curr[-1]
        print "predicted end of day is {}; end of today is {}; the profit is {}"\
            .format(denm_pred[-1], denm_curr[-1], pp)
        if pp < 0:
            pos = -1
        long_short_pos.append(pos)
    return long_short_pos

def get_day_diff(x, y, h):
    if x is None or y is None or h <= 0:
        return []
    day_diff = []
    prices = []
    for i in range(0, len(x), h):
        y_curr = lstm.denormalise_windows(list(y[i]), minList[i], maxList[i])
        x_curr = lstm.denormalise_windows(list(x[i, :, 0]), minList[i], maxList[i])
        
        day_diff.append(y_curr[-1] - x_curr[-1])
        prices.append(x_curr[-1])
    return prices, day_diff
"""        
def cal_profit_dollar(pos, diffs, prices):
    if len(pos) != len(diffs):
        return 0
    cnt = 0
    prof = []
    for i in range(len(pos)):
        if diff1[i] > 0:
            cnt += 1
        prof.append(diffs[i]*pos[i])
    return 100*cnt/len(pos), prof
"""    
   
def cal_profit_percent(pos, diffs, prices):
    
    if len(pos) != len(diffs):
        return 0
    cnt = 0
    prof_perc = []
    for i in range(len(pos)):
        if diff1[i] > 0:
            cnt += 1
        prof_perc.append(100*pos[i]*diffs[i]/prices[i])
    return 100*cnt/len(pos), prof_perc
# ****************************************************************

pos1 = gen_long_short_pos(model, X_train, y_train, 78)
ps, diff1 = get_day_diff(X_train, y_train, 78)        
winperc, profitperc = cal_profit_percent(pos1, diff1, ps)
print "The winning percentage is {}% and the total profit \
    in percentage is {}%".format(winperc, sum(profitperc))

hist_price = list(X_train[:,0,0])
mean_price = np.mean(hist_price)
mean_price
print "The LSTM model of {} epochs generated {}% profit in {} days".format(epochs, (100*profit1/mean_price), len(pos1))

#util.plot_results(dep3, decurr, seq_len, i)

#now try 300 epochs
epochs = 300
model300e = lstm.build_model([1, nodes, nodes*2, fwd_len], act_fnc)
global_start_time = time.time()
model300e.fit(
           X_train,
           y_train,
           batch_size=nodes*2,
           nb_epoch= epochs,
           validation_split=valid_ratio)

print ("The LSTM training is done in {} seconds".format(time.time() - global_start_time))
print('y shape', y_train.shape, '; X shape', X_train.shape)


pos1 = gen_long_short_pos(model300e, X_train, y_train, 78)
ps, diff1 = get_day_diff(X_train, y_train, 78)        
winperc, profitperc = cal_profit_percent(pos1, diff1, ps)
print "The winning percentage of the {} epochs LSTM model is {}% and the total profit \
    in percentage is {}%".format(epochs, winperc, sum(profitperc))


hist_price = list(X_train[:,0,0])
mean_price = np.mean(hist_price)
mean_price
print "The {} epochs LSTM model of {} epochs generated {}% profit in {} days".format(epochs, (100*profit1/mean_price), len(pos1))




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

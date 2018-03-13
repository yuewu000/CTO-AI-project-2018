# -*- coding: utf-8 -*-
"""
1. Post processing based on the raw LSTM forecasting weekly demands for the products
Raw results are not available if the historical series is too short (<12). 
For these short series, forecasting values are set as the mean of historical values for a quick solution
2. Visualization plotting the forecasting values consecutively after the historical data 
"""

import util
import numpy as np
import pandas as pd

# load the saved 100x52 numpy matrix as csv file, for the raw LSTM forecasting values ofweekly demands
# for product items with extremely short series, use a simple constant of the demands mean
# replace negative forecasting value to 0
# some items come with forecast less than 52 weeks, 
#   extrapolate with the mean of available forecasted values

filename = "Forecasting_Exercise.csv"
inname1 = "raw_lstm_forecast.csv"
submtname = 'forecast_submission.csv'

fc2 = np.loadtxt(inname1, delimiter = ',')
#check the dimension
fc2.shape
#replace negative vlues to 0
fc2[fc2<0] = 0

#postprocessing with simple stratege of replacement by means

for i in range(fc2.shape[0]):
    locp = 51
    while locp >= 0 and fc2[i, locp] == 0:
        locp -= 1
    mean1 = 0
    # forecasted values fro item i+1    
    pos1 = fc2[i, :locp+1]
    print ("item " + str(i), "mean " , mean1,  "len pos ", len(pos1), " pos end ", locp)
    mean1 = pos1.mean() 
    if (locp < 51):
        fc2[i, locp+1:52].fill(mean1)
    #take care of the items with extremely short series
    #calculate the mean as the forecasting value
    if locp < 0:
        pdts1 = pd.read_csv("./input/input_" + str(i+1) + ".csv")
        tol1 = list(pdts1[str(i+1)])
        loc_fp = util.first_Positive(tol1)
        valid_demands = tol1[loc_fp:]
        mean2 = np.mean(valid_demands)
        median2 = np.median(valid_demands)
        fc2[i].fill(mean2)
                
# convert the forecasting 100x52 numpy matrix to pandans datafrme 
# dataframe 100 rows are identified by the product Item_Numer
# the 52 columns are representing the 52 forecasting weeks
fc2 = np.around(fc2, decimals = 0).astype(int)

import util
item_map = util.item_num_map(filename) # dictionary mapping the product index to the Item_Number from the orig table

cols_hst = []
cols_prd = []
for n1 in range(157, 157+52, 1):
    cols_hst.append("Historical Period " + str(n1))
    cols_prd.append("Forecasted Period " + str(n1-156))
    cols_prd
    
# use the columns name by predicted periods    
df_all = pd.DataFrame(fc2, columns = cols_prd)  
df_all["Item Number"] = df_all.index.map(lambda x: item_map[x+1])

cols = ["Item Number"] + cols_prd
df_all.set_index(['Item Number'], inplace=True)
df_all
df_all.to_csv('forecast_submission.csv')

n = 25 
m = 4
sizex = 6*m
sizey = 4.2*n
hspce = 0.5 
inputname = "input_"

fig = plt.figure(figsize=(sizex, sizey), facecolor='white')
numPlot = n*m
#fig.suptitle("Plots of weekly demands for each item", fontsize=16)
    
for i in range(1, numPlot+1):
    inp = pd.read_csv("./input/" + inputname + str(i) + ".csv")
    data0 = list(inp[str(i)])
    data1 = list(fc2[i-1])
    #data2= data0 + data1
    df10 = pd.DataFrame(data0, index = range(len(data0)), columns = ['ts'])
    df11 = pd.DataFrame(data1, index = range(len(data0), len(data0)+len(data1)), columns = ['ts'])
    ax = plt.subplot(n,m,i)
    ax.set_title('Item_' + str(i), fontsize = 12)
    ax.plot(df10.index, df10.ts, label = "hist demand")  
    ax.plot(df11.index, df11.ts, label = "pred demand")  
fig.subplots_adjust(hspace=hspce)
plt.xlabel('Weekly demand for products')
plt.legend()
plt.show()    
#fc2.map(lambda x:int(x))  # floor the float numbers to int

# -*- coding: utf-8 -*-
#define some help functions, such as plot ts or identify the first non-zero num

import numpy as np
from numpy import newaxis
import pandas as pd
import math
import matplotlib.pyplot as plt

def plot_results(predicted_data, true_data, l, idx):
    fig = plt.figure(facecolor='white')
    df1 = pd.DataFrame(true_data, index = range(l), columns = ['true_data'])
    df2 = pd.DataFrame(predicted_data, index = range(l, l*2, 1), columns = ['pred_data'])
    ax = fig.add_subplot(111)
    ax.plot(df1.index, df1.true_data, label='True Data: No. ' + str(idx))
    plt.plot(df2.index,  df2.pred_data, label='Predicted Data')
    plt.legend()
    plt.show()

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()


def first_Positive (data):
    i = 0
    while(data[i] == 0):
        i +=1
    return i

def ts_plot(data, idx):
    fig = plt.figure(facecolor='white')
    df1 = pd.DataFrame(data, index = range(len(data)), columns = ['ts'])
   
    ax = fig.add_subplot(111)
    ax.plot(df1.index, df1.ts, label='Time Series_' + str(idx))
    plt.legend()
    plt.show()

def item_num_map(filename):
    g = pd.read_csv(filename) 
    #specificly here, for testing
    #g = pd.read_csv("Forecasting_Exercise.csv")
    g.reset_index()
    g.index
    g.get_value(0,'Item Number')
    map = {}
    for i in range(100):
        map[i+1] = g.get_value(i, 'Item Number')
    return map

def plot_nxm(inputname, n, m, sizex, sizey, hspce):
    fig = plt.figure(figsize=(sizex, sizey), facecolor='white')
    numPlot = n*m
    #fig.suptitle("Plots of weekly demands for each item", fontsize=16)
    
    for i in range(1, numPlot+1):
        inp = pd.read_csv("./input/" + inputname + str(i) + ".csv")
        data2 = list(inp[str(i)])
        df2 = pd.DataFrame(data2, index = range(len(data2)), columns = ['ts'])
        ax = plt.subplot(n,m,i)
        ax.set_title('Item_' + str(i), fontsize = 12)
        ax.plot(df2.index, df2.ts, label = "weekly demand")  
    fig.subplots_adjust(hspace=hspce)
    plt.xlabel('Weekly demand for products')
    plt.legend()
    plt.show()    

"""
this one is implemented in postprocess, to be updated yet here

def plot_n_m_forecasting(inputname, n, m, sizex, sizey, hspce):
    fig = plt.figure(figsize=(sizex, sizey), facecolor='white')
    numPlot = n*m
    #fig.suptitle("Plots of weekly demands for each item", fontsize=16)
    
    for i in range(1, numPlot+1):
        inp = pd.read_csv("./input/" + inputname + str(i) + ".csv")
        data2 = list(inp[str(i)])
        df2 = pd.DataFrame(data2, index = range(len(data2)), columns = ['ts'])
        ax = plt.subplot(n,m,i)
        ax.set_title('Item_' + str(i), fontsize = 12)
        ax.plot(df2.index, df2.ts, label = "weekly demand")  
    fig.subplots_adjust(hspace=hspce)
    plt.xlabel('Weekly demand for products')
    plt.legend()
    plt.show()    
"""
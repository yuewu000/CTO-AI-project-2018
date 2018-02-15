#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 17:47:07 2018

@author: web
"""

import pandas as pd
import io
import requests
import time
import os

cwd = os.getcwd() 
print cwd

def google_stocks(symbol, startdate = (1, 1, 2005), enddate = None, No = '01'):
 
    startdate = str(startdate[0]) + '+' + str(startdate[1]) + '+' + str(startdate[2])
 
    if not enddate:
        enddate = time.strftime("%m+%d+%Y")
    else:
        enddate = str(enddate[0]) + '+' + str(enddate[1]) + '+' + str(enddate[2])
 
    stock_url = "http://www.google.com/finance/historical?q=" + symbol + \
                "&startdate=" + startdate + "&enddate=" + enddate + "&output=csv"
 
    raw_response = requests.get(stock_url).content
 
    stock_data = pd.read_csv(io.StringIO(raw_response.decode('utf-8')))
 
    return stock_data
     
def google_stocks_new(symbol, type = '_EndOfDay_', output_dir = './input/', startdate = (1, 1, 2005), enddate = None):
 
    startdate = str(startdate[0]) + '+' + str(startdate[1]) + '+' + str(startdate[2])
 
    if not enddate:
        enddate = time.strftime("%m+%d+%Y")
    else:
        enddate = str(enddate[0]) + '+' + str(enddate[1]) + '+' + str(enddate[2])
 
    stock_url = "http://finance.google.com/finance/historical?q=" + symbol + \
                "&startdate=" + startdate + "&enddate=" + enddate + "&output=csv"
 
    raw_response = requests.get(stock_url).content
 
    stock_data = pd.read_csv(io.StringIO(raw_response.decode('utf-8')))
    stock_rev = stock_data.sort_index(axis = 0, ascending = False)
    close_data = stock_rev[['Date', 'Close']].reset_index()
    #print (type(close_data))
    close_data.to_csv(output_dir + symbol + type + '.csv')     
    return stock_data, close_data
     
if __name__ == '__main__':
    
    aapl_data2, aapl_close2 = google_stocks_new('AAPL', startdate = (1,1,1988))
    print(aapl_close2)
    print(aapl_data2)
    print(aapl_data2[aapl_data2['Date'].str.endswith('Apr-02') ])
    print(aapl_data2[aapl_data2['Date'].str.contains('Apr-02') ].count())
    type(aapl_close2)
    type(aapl_data2)
    
    import os
    cwd = os.getcwd()    
    cwd
    print('current data files: {}'.format(os.listdir(cwd)))
    print(os.listdir(cwd+'/input'))
    
    aapl_close2.head()
    aapl_close2['Close'].mean()
    aapl_close2['Close'].max()
    aapl_close2['Close'].min()
    aapl_close2['Close'].median()
    
    aapl_data2.head()
    
    aapl_data2.sort_index(axis = 0, ascending = False)
    aapl_close2.drop('index',axis = 1).head()
    aapl_data2.tail()
    
    aapl_data2.count()
    
    pl_close2.reset_index

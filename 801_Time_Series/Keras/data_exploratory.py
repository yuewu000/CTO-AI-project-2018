# -*- coding: utf-8 -*-

# Invesigate the demand data for products
# First positive sample is extracted
# Assuming the 0 demand at the beginning of each series are due to the new product
# New products do not have demand before they are commercialized

import numpy as np
import pandas as pd
import util

filename = "Forecasting_Exercise.csv"
#generate 100 input file for each product, if not done yet
lstm.input_prep(filename, "input_") 

#calculte and plot the first positive index

first_pos = []
for i in range(1, 101, 1):
    inp = pd.read_csv("./input/input_" + str(i) + ".csv")
    first_pos.append(util.first_Positive(inp[str(i)]))


#visualizing the weekly demand series for all the 100 products
# layout of 4 for each row and 25 rows totally

rws = 25 
cls = 4
sx = 6*cls
sy = 4.2*rws
hs = 0.5 
inpname = "input_"
 

util.plot_nxm(inpname, rws, cls, sx, sy, hs)
    
    
    
   
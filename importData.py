# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np




#Load data
train=pd.read_csv("train.csv") # training data
print(train.head(3))

test=pd.read_csv("test.csv") #test data
print(test.head(3))
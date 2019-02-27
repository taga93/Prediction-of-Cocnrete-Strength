import numpy as np
import pandas as pd
import math
import xlrd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

#Reading excel file and creating 3 different datasets

#Reading data from excel

data = pd.read_excel("DataSet.xls").round(2)
data_size = data.shape[0]
#print("Number of data:",data_size,"\n",data.head())

data_less_than_3days = data[data["Days"] <= 3]
data_less_than_3days_size = data_less_than_3days.shape[0]
#print("Number of data for less than 3 days:",data_less_than_3days_size,"\n",data_less_than_3days.head())

data_less_than_28days = data[data["Days"] <= 28]
data_less_than_28days_size = data_less_than_28days.shape[0]
#print("Number of data for less than 28 days:",data_less_than_28days_size,"\n",data_less_than_28days.head())

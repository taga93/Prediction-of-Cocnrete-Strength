import numpy as np
import pandas as pd
import math
import xlrd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.preprocessing import PolynomialFeatures

#prediction accuracy of dataframe
def prediction_accuracy(input_data):

    variable_list = []
    result_list = []
    
    for item in input_data:
        variable_list.append(item)
        result_list.append(item)

    variable_list = variable_list[:-1]
    result_list = result_list[-1]
    
    variables = input_data[variable_list]
    results = input_data[result_list]

    #accuracy of prediction (splittig dataframe in train and test)
    var_train, var_test, res_train, res_test = train_test_split(variables, results, test_size = 0.3, random_state = 4)

    regression = linear_model.LinearRegression() #making linear model
    model = regression.fit(var_train, res_train) #fitting data in linear model

    #calculating accuracy score
    score = regression.score(var_test, res_test)
    score = round(score*100, 2)

    info = "Strength prediction accuracy for this dataset: " + str(score) + " %"

    return info, regression, variables, results, variables, results

#print(prediction_accuracy(data)[0])
#print(prediction_accuracy(data_less_than_3days)[0])
#print(prediction_accuracy(data_less_than_28days)[0])


#prediction of concrete strength with multivariate polynomial regression
def polynomial_prediction_of_future_strength(input_data, cement, blast_fur_slug,fly_ash,
                                              water, superpl, coarse_aggr, fine_aggr, days):

    variables = prediction_accuracy(input_data)[4]
    results = prediction_accuracy(input_data)[5]
    n = results.shape[0]
    results = results.values.reshape(n,1) #reshaping the values so that variables and results have the same shape

    #transforming the data into polynomial function
    Poly_Regression = PolynomialFeatures(degree=2)
    poly_variables = Poly_Regression.fit_transform(variables)

    #accuracy of prediction(splitting the dataset on train and test)
    poly_var_train, poly_var_test, res_train, res_test = train_test_split(poly_variables, results, test_size = 0.3, random_state = 4)

    input_values = [cement, blast_fur_slug, fly_ash, water, superpl, coarse_aggr, fine_aggr, days]
    input_values = Poly_Regression.transform([input_values]) #transforming the data for prediction in polynomial function

    regression = linear_model.LinearRegression() #making the linear model
    model = regression.fit(poly_var_train, res_train) #fitting polynomial data to the model
    
    predicted_strength = regression.predict(input_values) #strength prediction
    predicted_strength = round(predicted_strength[0,0], 2)
    
    score = model.score(poly_var_test, res_test) #accuracy prediction
    score = round(score*100, 2)

    accuracy_info = "Accuracy of concrete class prediction: " + str(score) + " %\n"
    prediction_info = "Prediction of future concrete class after "+ str(days)+" days: "+ str(predicted_strength) 

    info = "\n" + accuracy_info + prediction_info

    return info

#print(polynomial_prediction_of_future_strength(data, 214.9 , 53.8, 121.9, 155.6, 9.6, 1014.3, 780.6, 7)) #true value affter 3 days: 18.02 MPa
#print(polynomial_prediction_of_future_strength(data_less_than_3days, 260.9, 100.5, 78.3, 200.6, 8.6, 864.5, 761.5, 3)) #true value affter 14 days: 38.60 MPa
#print(polynomial_prediction_of_future_strength(data_less_than_28days, 260.9, 100.5, 78.3, 200.6, 8.6, 864.5, 761.5, 28)) #true value affter 28 days: 52.20 MPa

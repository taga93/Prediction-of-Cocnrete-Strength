import numpy as np
import pandas as pd
import math
import xlrd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.preprocessing import PolynomialFeatures


#creating new dataframe with one additional column for concrete class
def concrete_strength_class(input_data):

    MB_list = []
    for row in input_data["MPa"]:

        if row < 10:
            MB_list.append("nan/<10 MPa")
            
        elif row >= 10 and row <= 20:
            MB_list.append("MB 10-15")

        elif row > 20 and row <= 30:
            MB_list.append("MB 20-25")
            
        elif row > 30 and row <= 40:
            MB_list.append("MB 30-35")
            
        elif row > 40 and row <= 50:
            MB_list.append("MB 40-45")
            
        elif row > 50 and row <=60:
            MB_list.append("MB 50-55")

        else:
            MB_list.append(">MB 60")


    data_with_concrete_class = input_data.copy()
    data_with_concrete_class["MB"] = MB_list

    return data_with_concrete_class

#print(concrete_strength_class(data))
#print(concrete_strength_class(data_less_than_3days))
#print(concrete_strength_class(data_less_than_28days))


#prediction of future cocnrete class with decision tree classifier
def predict_concrete_class(input_data, cement, blast_fur_slug,fly_ash,
                            water, superpl, coarse_aggr, fine_aggr, days):

    data_for_tree = concrete_strength_class(input_data) #calling function to create concrete class

    variable_list = []
    result_list = []
    
    for index, row in data_for_tree.iterrows():
        variable = row.tolist()
        variable = variable[0:8]
        
        variable_list.append(variable)
        result_list.append(row[-1])

    #accuracy of prediction(splitting the dataset on train and test)
    var_train, var_test, res_train, res_test = train_test_split(variable_list, result_list, test_size = 0.3, random_state = 42)

    decision_tree = tree.DecisionTreeClassifier() #defining decision tree
    decision_tree = decision_tree.fit(var_train, res_train) #adding values to dacision tree
    
    input_values = [cement, blast_fur_slug, fly_ash, water, superpl, coarse_aggr, fine_aggr, days]

    #calculating the accuracy
    score = decision_tree.score(var_test, res_test)
    score = round(score*100, 2)

    prediction = decision_tree.predict([input_values]) #adding values for prediction
    prediction = prediction[0]

    accuracy_info = "Accuracy of concrete class prediction: " + str(score) + " %\n"
    prediction_info = "Prediction of future concrete class after "+ str(days)+" days: "+ str(prediction) 
    info = "\n" + accuracy_info + prediction_info

    return info

#print(predict_concrete_class(data, 260.9, 100.5, 78.3, 200.6, 8.6, 864.5, 761.5, 28)) #true value affter 28 days: 32.40 MPa
#print(predict_concrete_class(data_less_than_3days, 260.9, 100.5, 78.3, 200.6, 8.6, 864.5, 761.5, 3)) #true value affter 28 days: 32.40 MPa
#print(predict_concrete_class(data_less_than_28days, 214.9 , 53.8, 121.9, 155.6, 9.6, 1014.3, 780.6, 28)) #true value affter 28 days: 52.20 MPa

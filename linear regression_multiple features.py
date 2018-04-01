#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 18:20:42 2018

@author: olaolu
"""

import math
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

# Import the data
california_housing_dataframe = (pd.
                                read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=","))


#randomize the data
california_housing_dataframe = california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))

#divide data into training, validation and test sets 
def pre_features(california_housing_dataframe):
    #create another dataframe from the features to exclude median_house_value
    input_features = california_housing_dataframe[[
            "longitude","latitude","housing_median_age","total_rooms",
            "total_bedrooms", "population", "households", "median_income"
            ]]
    
    #then add a synthetic feature just for bants
    input_features["rooms_per_person"] = (input_features["total_rooms"]
                                        /input_features["population"])
    
    #returns a dataframe with the input features and a synthetic feature created
    return input_features
def pre_labels(california_housing_dataframe):
    input_labels = california_housing_dataframe[["median_house_value"]]
    #scale the data to the thousands
    input_labels /= 1000
    return input_labels

#features = pre_features(california_housing_dataframe.head(10000))
#print(features)
#targets = pre_labels(california_housing_dataframe.head(10000))["median_house_value"]
#print(targets)

def input_fn(features, labels, batch_size = 1, shuffle = True, num_epochs = None):
    
    my_features = {key : np.array(item) for key, item in dict(features).items()}
    
    #must take in a dict of names and arrays and a 1d array
    ds = Dataset.from_tensor_slices((my_features, labels))
    
    #batch and repeat for training.
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    #Shuffle, if applicable
    if shuffle:
        ds.shuffle(buffer_size = 10000)
    my_features, labels = ds.make_one_shot_iterator().get_next()
    
    return my_features, labels

def train_fn(steps, learning_rate, batch_size):
    
    periods = 10
    steps_per_period = steps/periods
    
    training_input_fn = lambda : input_fn(pre_features(california_housing_dataframe.head(10000)),
                                          pre_labels(california_housing_dataframe.head(10000))["median_house_value"],
                                          batch_size= batch_size)
    
    training_predict_input_fn = lambda : input_fn(pre_features(california_housing_dataframe.head(10000)),
                                          pre_labels(california_housing_dataframe.head(10000))["median_house_value"],
                                          shuffle = False, num_epochs = 1)
    validate_predict_input_fn = lambda : input_fn(pre_features(california_housing_dataframe.tail(7000)),
                                          pre_labels(california_housing_dataframe.tail(7000))["median_house_value"],
                                          shuffle = False, num_epochs = 1)
    
    #define multiple feature columns
    feature_columns = set([tf.feature_column.numeric_column(i) for i in 
         pre_features(california_housing_dataframe)])
        
    #configure the linear regressor
    #first configure the optimizer as gradient descent
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    #also clip the gradients so they dont get too large
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    
    linear_regressor = tf.estimator.LinearRegressor(feature_columns = feature_columns,
                                                    optimizer = my_optimizer)
    
    #now train but do so in a loop so we can track loss
    for i in range(periods):
        linear_regressor.train(input_fn = training_input_fn, steps = steps_per_period)
        
        predictions_training = linear_regressor.predict(input_fn = training_predict_input_fn)
        #convert predictions to numpy array
        predictions_training = np.array([x['predictions'][0] for x in predictions_training])
        
        #calculate mean squared loss
        MSE = metrics.mean_squared_error(predictions_training, pre_labels(california_housing_dataframe.head(10000))["median_house_value"])
        RMSE = math.sqrt(MSE)
        print("training RMSE for period {i} is {RMSE}".format(**locals()))
        
        #for validation, we only predict with the model
        predictions_validation = linear_regressor.predict(input_fn = validate_predict_input_fn)
        predictions_validation = np.array([x['predictions'][0] for x in predictions_validation])
    
        MSE = metrics.mean_squared_error(predictions_validation, pre_labels(california_housing_dataframe.tail(7000))["median_house_value"])
        RMSE2 = math.sqrt(MSE)
        print("validation RMSE for period {i} is {RMSE2}".format(**locals()))
    
    return linear_regressor
train_fn(1200, 0.00001,50)
    
        
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 12:34:51 2018

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

# reindex the data for better randomness
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))
#scale the target data to be closer to readable values
california_housing_dataframe["median_house_value"] /= 1000.0





def my_input_fn(features, targets, batch_size = 1, shuffle=True, num_epochs=None):
    
  
  
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                                           
 
    # Construct a dataset, and configure batching/repeating
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified
    if shuffle:
      ds = ds.shuffle(buffer_size=10000)
    
    # Return the next batch of data
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

  
def train(steps, batch_size, learning_rate, features):
    # Define the input feature but let it take from function arguments
    feature = california_housing_dataframe[[features]]
    
    # Configure a numeric feature column for total_rooms.
    feature_columns = [tf.feature_column.numeric_column(features)]
    # Define the label
    target = california_housing_dataframe["median_house_value"]
    
    # Use gradient descent as the optimizer for training the model.
    my_optimizer=tf.train.GradientDescentOptimizer(learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    
    # Configure the linear regression model with our feature columns and optimizer.
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=feature_columns,
        optimizer=my_optimizer
    )
    # train and predict functions expects only one argument as input function so wrap in lambda
    train_input_fn = lambda:my_input_fn(feature, target, 
                                        batch_size = batch_size)
    #for the predict input function, we want just one random pair of feature and label
    predict_input_fn = lambda:my_input_fn(feature, 
                                          target, num_epochs = 1, 
                                          shuffle=False )
    
    
    #to track loss reduction progress, we have to predict periodically and calculate loss
    periods = 10
    steps_per_period = steps / periods
    
    #convergence is assumed when loss in consecutive periods is small enough
    #to track this, we wrap in a loop of periods
    for i in range(periods):
        linear_regressor.train(
            input_fn = train_input_fn,
            steps=steps_per_period
        )
        predictions = linear_regressor.predict(input_fn = predict_input_fn)
        #predict returns dicts of numpy arrays of the form 
        #{'predictions': np.array([]), dtype}, {'predictions': np.array([]), dtype}
        #so we format to a numpy array (basically reversing the input
        #function for features)
        predictions = np.array([x['predictions'][0] for x in predictions])
        
        # we then compare predictions to labels and calculate loss from the two
        # we use means squared error from sklearn.metrics
        
        mean_squared_error = metrics.mean_squared_error(predictions, target)
        # the root mean square error is more comparable to the base data scale
        root_mean_squared_error = math.sqrt(mean_squared_error)
        print(root_mean_squared_error)

train(100, 10, 0.0001, "total_rooms")
                                              
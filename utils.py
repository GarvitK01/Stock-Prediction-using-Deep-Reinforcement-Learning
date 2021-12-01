import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def load_data():

    data = pd.read_csv("./HDB.csv")
    data.drop("Date", inplace = True, axis = 1)

    return data

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def add_columns(data):

    '''
    Adding Moving Average, and Price Rate of Change in the dataset to add more information to a state
    '''
    
    values = data["Close"]
    values = np.array(values)
    #Moving Average
    mov_avg = moving_average(values, 7)  # Final Moving Average
    mov_avg = list(mov_avg) + (len(values) - len(mov_avg)) * [mov_avg[-1]]
    
    #Price Rate of Change
    proc = [0]
    for i in range(1, len(values)):
        proc.append((values[i] - values[i-1])/values[i-1])
        
    proc = np.array(proc)   # Final Price rate of Change
    
    data["Moving_Average"] = mov_avg
    data["PROC"] = proc
    
    return data


def preprocess_data(data):

    '''
    Preprocessing the dataset, to normalize the values between 0 and 1.
    '''


    train_data = data[:int(0.8 * len(data))]
    test_data = data[int(0.8 * len(data)):]
    
    scaler = MinMaxScaler(feature_range = (0.001, 1))
    
    for column in train_data.columns:
        
        values = train_data[column]
        values = np.array(values)
        values = values.reshape(-1, 1)
        train_data[column] = scaler.fit_transform(values)
        train_data[column] = train_data[column].apply(lambda X: round(X, 5))
        
        values = test_data[column]
        values = np.array(values)
        values = values.reshape(-1, 1)
        test_data[column] = scaler.fit_transform(values)
        test_data[column] = test_data[column].apply(lambda X: round(X, 5))   
        
    train_data.to_csv("final_train.csv")
    test_data.to_csv("final_test.csv")

    return train_data, test_data, scaler


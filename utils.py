import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data():

    data = pd.read_csv("./HDB.csv")
    data.drop("Date", inplace = True, axis = 1)

    return data

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def add_columns(data):
    
    values = data["Close"]
    values = np.array(values)
    #Moving Average
    mov_avg = moving_average(values, 7)
    mov_avg = list(mov_avg) + (len(values) - len(mov_avg)) * [mov_avg[-1]]
    
    #Price Rate of Change
    proc = [0]
    for i in range(1, len(values)):
        proc.append((values[i] - values[i-1])/values[i-1])
        
    proc = np.array(proc)
    
    data["Moving_Average"] = mov_avg
    data["PROC"] = proc
    
    return data


def preprocess_data(data):


    train_data = data[:int(0.8 * len(data))]
    test_data = data[int(0.8 * len(data)):]
    
    scaler = MinMaxScaler()
    
    for column in train_data.columns:
        
        values = train_data[column]
        values = np.array(values)
        values = values.reshape(-1, 1)
        train_data[column] = scaler.fit_transform(values)
        train_data[column] = train_data[column].apply(lambda X: round(X, 4))
        
        values = test_data[column]
        values = np.array(values)
        values = values.reshape(-1, 1)
        test_data[column] = scaler.fit_transform(values)
        test_data[column] = test_data[column].apply(lambda X: round(X, 4))   
        
    return train_data, test_data, scaler


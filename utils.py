import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data():

    data = pd.read_csv("./HDB.csv")
    data.drop("Date", inplace = True, axis = 1)

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


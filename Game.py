import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import keras
import tensorflow as tf
from utils import *
from MarketAgent import *
import warnings
warnings.filterwarnings("ignore")


data = load_data()
data = add_columns(data)
train_data, test_data, scaler = preprocess_data(data)

train_data = train_data.to_numpy()
test_data = test_data.to_numpy()

#! Hyperparameters

epsilon = 0.3
gamma = 0.7
timestpes = len(train_data)
states = train_data.shape[1]
test = False
alpha = 0.1

agent = MarketAgent(epsilon, gamma, timestpes, states, test)

if __name__ == "__main__":
    
    print("\n\n ***** Starting Simulation *****\n\n")

    agent.play_episode(epsilon, gamma, alpha, train_data)


    print("\n\n **** End of Episode ****\n\n")
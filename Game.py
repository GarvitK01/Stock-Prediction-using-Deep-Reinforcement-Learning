import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import keras
import tensorflow as tf
from utils import *
from MarketAgent import *
import warnings
import sys
import getopt
from tqdm import tqdm
warnings.filterwarnings("ignore")


data = load_data()
data = add_columns(data)
training, testing, scaler = preprocess_data(data)

train_data = training.to_numpy()
test_data = testing.to_numpy()

#! Hyperparameters

epsilon = 0.3
gamma = None
timestpes = len(train_data)
states = train_data.shape[1]
test = False
alpha = None
episodes = None

argv = sys.argv[1:]

try:
    opts, args = getopt.getopt(argv, "g:a:e:")
except:
    print("Error in reading command Line Arguments")

for opt, arg in opts:

    if opt in ['-g', '--gamma']:
        gamma = float(arg)
    elif opt in ['-a', '--alpha']:
        alpha = float(arg)
    elif opt in ['-e', '--episodes']:
        episodes = int(arg)


agent = MarketAgent(epsilon, gamma, timestpes, states, test)

if __name__ == "__main__":

    for i in range(episodes):
    
        print(f"Episode {i+1}/{episodes} \n")

        agent.play_episode(epsilon, gamma, alpha, train_data)

    #! Prediction

    prediction = agent.model.predict(test_data)
    predicted_trend = prediction[:, 0]
    predicted_trend = np.tanh(predicted_trend)
    testing['Trend'] = predicted_trend
    testing.to_csv(f"Test_Prediction_{alpha}_{gamma}_new.csv")
    
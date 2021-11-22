from os import times
import numpy as np
import matplotlib.pyplot as plt	
import pandas as pd
import keras
import tensorflow as tf
import random
import datetime
import time as samay


class MarketAgent():

    def __init__(self, epsilon, gamma, timesteps, states, test = False, model_name = "Agent"):

        self.epsilon = epsilon
        self.gamma = gamma
        self.actions = ['Buy', 'Sell', 'Hold']
        self.state_value = np.random.uniform(low=-1, high=1, size=(timesteps, 1))
        self.policy = np.array((states, 1))
        self.state_size = states
        self.model = self.make_model()
        self.model_name = model_name


    def take_action(self, state):

        if random.random() >= self.epsilon:
            pass  #* Return Greedy Step
        else:
            return random.choice(self.actions)

    def make_model(self):

        '''
        Deep Learning Model to estimate the value function, which will be the price
        Returns: Keras Sequential Model
        '''

        model = keras.models.Sequential()
        model.add(keras.layers.Dense(24, input_dim = self.state_size, activation = 'relu'))
        model.add(keras.layers.Dense(32,  activation = 'relu'))
        model.add(keras.layers.Dense(16,  activation = 'relu'))
        model.add(keras.layers.Dense(1, activation = 'tanh'))
        model.compile(loss = tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0001))
        
        return model

    def play_episode(self, epsilon, gamma, alpha, data):
        
        '''
        One Episode an Agent Plays
        '''
    
        for time in range(1, len(data) - 1):

            state = data[time]   # State Vector
            reward = 100 * (state[3] - data[time-1][3])

            input_state = data[time].reshape(1, self.state_size)
            prediction = self.model.predict(input_state)   # Target
            prediction = prediction[0]

            self.state_value[time] = self.state_value[time] + alpha * (reward + gamma * prediction - self.state_value[time])  # Vst += alpha * {}
            
            #! Training Step
            self.model.fit(input_state, self.state_value[time+1], epochs = 100, verbose = 0)

        self.model.save("saved_model")

            # X. theta = y_pred
            # y_true - y_pred ^ 2 = Loss 
            # d(loss)/d(theta) = gradient
            # theta = theat + gradient * theta


        
    

 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import keras
import tensorflow as tf
import random


class MarketAgent():

    def __init__(self, epsilon, gamma, timesteps, states, test = False, model_name = "Agent"):

        self.epsilon = epsilon
        self.gamma = gamma
        self.actions = ['Buy', 'Sell', 'Hold']
        self.state_value = np.random.random((timesteps, 1))
        self.policy = np.array((states, 1))
        self.state_size = states
        self.model = self.make_model()


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
        model.add(keras.layers.Dense(16, input_dim = self.state_size, activation = 'relu'))
        model.add(keras.layers.Dense(32, activation = 'relu'))
        model.add(keras.layers.Dense(16, activation = 'relu'))
        model.add(keras.layers.Dense(1, activation = 'linear'))
        model.compile(loss = 'mse', optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001))
        
        return model

    def play_episode(self, epsilon, gamma, alpha, data):
        
        '''
        One Episode an Agent Plays
        '''
    
        for time in range(1, len(data) - 1):
            state = data[time]   # State Vector
            reward = 100 * (state[3] - data[time-1][3])/(data[time-1][3] + 0.00001)
#             print(f"Current Reward {reward}\n" )
            next_state = time + 1
            self.state_value[time] += alpha * (reward + gamma * self.state_value[next_state] - self.state_value[time])
#             print(f"Current State: {state}, Shape: {state.shape}\n")
            state = state.reshape(-1, self.state_size)
            self.model.fit(state, self.state_value[time], epochs = 3, verbose = 1)
            


        
    


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
            next_state = time + 1
            input_state = data[time].reshape(1, 6)
            prediction = self.model.predict(input_state)
            prediction = prediction[0]
            # print(f"\n\n{type(prediction)} {type(self.state_value[time])}\n\n")
            self.state_value[time] += alpha * (reward + gamma * prediction - self.state_value[time])
            state = state.reshape(-1, self.state_size)
            self.model.fit(state, self.state_value[time], epochs = 3, verbose = 1)

        curr_time = samay.time()
        curr_time = datetime.datetime.fromtimestamp(curr_time).strftime('%d_%H:%M:%S')
        self.model.save(f"model_saved_{curr_time}")
            


        
    


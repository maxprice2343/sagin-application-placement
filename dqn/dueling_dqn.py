import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import keras

NUM_INPUT = 12
NUM_OUTPUT = 5

class DuelingDQN(keras.Model):
    """The agent that learns to make choices in the environment."""
    def __init__(self, *args, **kwargs):
        super(DuelingDQN, self).__init__(*args, **kwargs)
        self.d1 = keras.layers.Dense(128, activation='relu')
        self.d2 = keras.layers.Dense(128, activation='relu')
        self.v = keras.layers.Dense(1, activation=None)
        self.a = keras.layers.Dense(NUM_OUTPUT, activation=None)
    
    def call(self, input_data):
        x = self.d1(input_data)
        x = self.d2(x)
        v = self.v(x)
        a = self.a(x)
        Q = v + (a - tf.math.reduce_mean(a, axis=1, keepdims=True))
        return Q
    
    def advantage(self, state):
        x = self.d1(state)
        x = self.d2(x)
        a = self.a(x)
        return a
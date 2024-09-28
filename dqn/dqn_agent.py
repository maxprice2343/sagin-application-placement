from dueling_dqn import DuelingDQN
from replay_buffer import ReplayBuffer
import numpy as np
import keras

OBSERVATION_SPACE = 12
ACTION_SPACE = 5

class DQNAgent:
    def __init__(self, dqn=None, gamma=0.99, replace=100, lr=0.001):
        # High gamma ensures the agent prefers long-term rewards over short
        # term rewards
        self.gamma = gamma
        # Epsilon starts at 1 and decays to 0.01 over the course of training
        self.epsilon = 1.0
        self.min_epsilon = 0.01
        self.epsilon_decay = 1e-3
        self.replace = replace
        self.trainstep = 0
        self.memory = ReplayBuffer(OBSERVATION_SPACE)
        self.batch_size = 64

        if dqn is None:
            self.q_net = DuelingDQN()
            self.target_net = DuelingDQN()
            opt = keras.optimizers.Adam(learning_rate=lr)
            self.q_net.compile(loss='mse', optimizer=opt) # type: ignore
            self.target_net.compile(loss='mse', optimizer=opt) # type: ignore
        else:
            self.q_net = dqn

    def act(self, state) -> int:
        if np.random.rand() <= self.epsilon:
            return np.random.choice([i for i in range(ACTION_SPACE)])
        else:
            actions = self.q_net.advantage(np.array([state]))
            action = np.argmax(actions)
            return int(action)
        
    def update_memory(self, state, action, reward, next_state, done):
        self.memory.store_experience(state, action, reward, next_state, done)

    def update_target(self):
        self.target_net.set_weights(self.q_net.get_weights())

    def update_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon -= self.epsilon_decay
        else:
            self.epsilon = self.min_epsilon
        return self.epsilon

    def train(self):
        if self.memory.pointer < self.batch_size:
            return
        
        if self.trainstep % self.replace == 0:
            self.update_target()
        
        states, actions, rewards, next_states, dones = self.memory.sample_batch(self.batch_size)
        target = self.q_net.predict(states)
        next_state_val = self.target_net.predict(next_states)
        max_action = np.argmax(self.q_net.predict(next_states), axis=1)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        q_target = np.copy(target)
        q_target[batch_index, actions] = rewards + self.gamma * next_state_val[batch_index, max_action]*dones
        self.q_net.train_on_batch(states, q_target)
        self.update_epsilon()
        self.trainstep += 1
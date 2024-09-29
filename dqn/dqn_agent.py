from dueling_dqn import DuelingDQN
from replay_buffer import ReplayBuffer
import numpy as np
import keras

OBSERVATION_SPACE = 12
ACTION_SPACE = 5

class DQNAgent:
    def __init__(self, dqn=None, gamma=0.99, update_frequency=100, lr=0.001):
        # High gamma ensures the agent prefers long-term rewards over short
        # term rewards
        self.gamma = gamma
        # Epsilon starts at 1 and decays to 0.01 over the course of training
        self.epsilon = 1.0
        self.min_epsilon = 0.01
        self.epsilon_decay = 1e-3

        # How often the target network should be updated
        self.update_frequency = update_frequency
        # Keeps count of the number of times the agent has been trained
        self.trainstep = 0

        self.replay_buffer = ReplayBuffer(OBSERVATION_SPACE)
        self.batch_size = 64

        if dqn is None:
            self.q_net = DuelingDQN()
            self.target_net = DuelingDQN()
            opt = keras.optimizers.Adam(learning_rate=lr)
            self.q_net.compile(loss='mse', optimizer=opt) # type: ignore
            self.target_net.compile(loss='mse', optimizer=opt) # type: ignore
        else:
            self.q_net = dqn

    def policy(self, state) -> int:
        """Epsilon-Greedy policy. Has a chance based on epsilon to return a
        random action, otherwise returns the best possible action (based on
        the DQN)"""
        # Exploration - Generates a random action if the random number is less
        # than epsilon
        if np.random.rand() <= self.epsilon:
            return np.random.choice([i for i in range(ACTION_SPACE)])
        # Exploitation - Uses the DQN to determine the best action
        else:
            actions = self.q_net.advantage(np.array([state]))
            action = np.argmax(actions)
            return int(action)
        
    def store_experience(self, state, action, reward, next_state, done):
        """Takes an experience and stores it in the replay buffer."""
        self.replay_buffer.store_experience(state, action, reward, next_state, done)

    def update_target_network(self):
        """Updates the target network with the weights from the main network"""
        self.target_net.set_weights(self.q_net.get_weights())

    def decay_epsilon(self):
        """Decays the value of epsilon"""
        # If epsilon hasn't already hit it's minimum value, it gets decayed
        if self.epsilon > self.min_epsilon:
            self.epsilon -= self.epsilon_decay
        else:
            self.epsilon = self.min_epsilon
        return self.epsilon

    def train(self):
        """Uses sampled experiences from the replay buffer to train the main
        network"""
        # If there aren't enough experiences in the buffer yet, skip the
        # training
        if self.replay_buffer.pointer < self.batch_size:
            return

        # Updates the target netowrk every self.update_frequency steps
        if self.trainstep % self.update_frequency == 0:
            self.update_target_network()

        # Samples a batch of experiences from the replay buffer and splits them
        states, actions, rewards, next_states, dones = self.replay_buffer.sample_batch(self.batch_size)

        target = self.q_net.predict(states)
        next_state_val = self.target_net.predict(next_states)
        max_action = np.argmax(self.q_net.predict(next_states), axis=1)

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        q_target = np.copy(target)
        q_target[batch_index, actions] = rewards + self.gamma * next_state_val[batch_index, max_action]*dones

        self.q_net.train_on_batch(states, q_target)
        self.decay_epsilon()
        self.trainstep += 1
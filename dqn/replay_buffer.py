from collections import deque
import numpy as np
import random

MIN_BATCH_SIZE = 128

class ReplayBuffer:
    """Stores state transitions to allow for training the dqn."""
    
    def __init__(self, observation_space, buffer_size=1_000_000):
        self.buffer_size = buffer_size
        self.state_memory = np.zeros(
            (self.buffer_size, observation_space),
            dtype=np.int32
        )
        self.action_memory = np.zeros(self.buffer_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.buffer_size, dtype=np.float32)
        self.next_state_memory = np.zeros(
            (self.buffer_size, observation_space),
            dtype=np.int32
        )
        self.done_memory = np.zeros(self.buffer_size, dtype=np.int32)
        self.pointer = 0

    def store_experience(self, state, action, reward, next_state, done):
        """Stores an experience for later training."""
        # Buffer is calculated module self.buffer_size so it doesn't get larger
        # than the buffer size
        idx = self.pointer % self.buffer_size
        # Stores the new experience in the replay buffer
        self.state_memory[idx] = state
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.next_state_memory[idx] = next_state
        self.done_memory[idx] = 1 - done
        # Increments the pointer variable
        self.pointer += 1

    def sample_batch(self, batch_size=64):
        """Samples a batch of experiences from the buffer."""
        max_memory = min(self.pointer, self.buffer_size)
        batch = np.random.choice(max_memory, batch_size, replace=False)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.next_state_memory[batch]
        dones = self.done_memory[batch]
        return states, actions, rewards, next_states, dones
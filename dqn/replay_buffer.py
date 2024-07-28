from collections import deque
import numpy as np
import random

MIN_BATCH_SIZE = 128

class ReplayBuffer:
    """Stores state transitions to allow for training the dqn."""
    
    def __init__(self):
        self.experiences = deque()

    def store_experience(self, state, next_state, reward, action, done):
        """Stores an experience for later training."""
        self.experiences.append(((state, next_state, reward, action, done)))

    def sample_batch(self):
        """Samples a batch of experiences from the buffer."""
        batch_size = random.randint(MIN_BATCH_SIZE, len(self.experiences))
        sampled_batch = random.sample(self.experiences, batch_size)
        state_batch, next_state_batch, reward_batch, action_batch, done_batch = [], [], [], [], []

        # Splits the sampled batch into 5 arrays
        for experience in sampled_batch:
            state_batch.append(experience[0])
            next_state_batch.append(experience[1])
            reward_batch.append(experience[2])
            action_batch.append(experience[3])
            done_batch.append(experience[4])
        
        return np.array(state_batch), np.array(next_state_batch), np.array(reward_batch), np.array(action_batch), np.array(done_batch)
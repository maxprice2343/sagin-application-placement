import numpy as np
import tensorflow as tf
import keras


class DQNAgent:
    """The agent that learns to make choices in the environment."""
    def __init__(self):
        self.q_net = self._build_dqn_model()
        self.target_q_net = self._build_dqn_model()

    @staticmethod
    def _build_dqn_model():
        """Constructs and returns a Sequential model with 3 layers"""
        model = keras.Sequential(
            [
                keras.layers.Input(64),
                keras.layers.Dense(32, activation="relu", name="layer1"),
                keras.layers.Dense(15, activation="linear", name="output layer")
            ]
        )
        model.compile(
            keras.optimizers.Adam(learning_rate=0.001) # type: ignore
        )
        return model

    def collect_policy(self, state):
        if np.random.random() < 0.05:
            return self.random_policy(state)
        return self.policy(state)

    def policy(self, state):
        """Returns an action that should be taken based on an environment state."""
        # Converts the state to an input suitable to the DQN
        q_net_input = tf.convert_to_tensor(state[None, :], dtype=tf.int32)
        q_value = self.q_net(q_net_input)
        # Determines the best action based on the output from the DQN
        action = np.argmax(q_value.numpy()[0], axis=0)
        return action
    
    def random_policy(self, state):
        return np.random.randint(0, 15)

    def update_target_network(self):
        """Updates the target DQN's weights with the main DQN's weights"""
        self.target_q_net.set_weights(self.q_net.get_weights())

    def train(self, batch):
        """Trains the DQN model on a batch of experiences."""
        state_batch, next_state_batch, action_batch, reward_batch, done_batch = batch
        current_q = self.q_net(state_batch).numpy()
        target_q = np.copy(current_q)
        next_q = self.target_q_net(next_state_batch).numpy()
        max_next_q = np.amax(next_q, axis=1)
        for i in range(state_batch.shape[0]):
            target_q_value = reward_batch[i]
            if not done_batch[i]:
                target_q_value += 0.95 * max_next_q[i]
            target_q[i][action_batch[i]] = target_q_value
        training_history = self.q_net.fit(x=state_batch, y=target_q, verbose="0")
        loss = training_history.history['loss']
        return loss
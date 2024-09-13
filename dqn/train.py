"""Trains the DQN agent by running it within the environment, storing the
gameplay experiences and using them to improve results."""

import gymnasium as gym
from gymnasium.wrappers.flatten_observation import FlattenObservation
import environment
from dqn_agent import DQNAgent
from replay_buffer import ReplayBuffer

def train_model(max_episodes=10000):
    agent = DQNAgent()
    buffer = ReplayBuffer()
    env = gym.make("ApplicationPlacementEnv-v0", render_mode="human")
    env = FlattenObservation(env)

    for _ in range(100):
        collect_experiences(env, agent, buffer)
    for episode_num in range(max_episodes):
        collect_experiences(env, agent, buffer)
        experience_batch = buffer.sample_batch()
        loss = agent.train(experience_batch)
        avg_reward = evaluate_training_result(env, agent)

        # Every 20 episodes update the weights from the main network to the
        # target network
        if episode_num % 20 == 0:
            agent.update_target_network()

    env.close()

def collect_experiences(env, agent, buffer):
    state, _ = env.reset()
    done = False
    while not done:
        action = agent.collect_policy(state)
        next_state, reward, done, _, _ = env.step(action)
        if done:
            reward = -1.0
        buffer.store_experience(state, next_state, reward, action, done)
        state = next_state

def evaluate_training_result(env, agent):
    reward_total = 0.0
    num_episodes = 10

    for i in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action = agent.policy()
            next_state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            state = next_state
        reward_total += episode_reward
    avg_reward = reward_total / num_episodes
    return avg_reward

train_model()
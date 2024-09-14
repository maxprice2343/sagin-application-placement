"""Trains the DQN agent by running it within the environment, storing the
gameplay experiences and using them to improve results."""

from environment.envs.application_placement_env import ApplicationPlacementEnv
import gymnasium as gym
from gymnasium.wrappers.flatten_observation import FlattenObservation
import environment
from dqn_agent import DQNAgent
from replay_buffer import ReplayBuffer
import asyncio

def train_model(max_episodes=10000):
    agent = DQNAgent()
    buffer = ReplayBuffer()
    env = ApplicationPlacementEnv(render_mode="human")

    for _ in range(100):
        asyncio.run(collect_experiences(env, agent, buffer))
    for episode_num in range(max_episodes):
        asyncio.run(collect_experiences(env, agent, buffer))
        experience_batch = buffer.sample_batch()
        loss = agent.train(experience_batch)
        avg_reward = evaluate_training_result(env, agent)

        # Every 20 episodes update the weights from the main network to the
        # target network
        if episode_num % 20 == 0:
            agent.update_target_network()

    env.close()

async def collect_experiences(env, agent, buffer):
    state, _ = env.reset()
    done = False
    while not done:
        action = agent.collect_policy(state)
        next_state, reward, done, _, _ = await env.step(action)
        if done:
            reward = -1.0
        buffer.store_experience(state, next_state, reward, action, done)
        state = next_state

async def evaluate_training_result(env, agent):
    reward_total = 0.0
    num_episodes = 10

    for i in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action = agent.policy()
            next_state, reward, done, _, _ = await env.step(action)
            episode_reward += reward
            state = next_state
        reward_total += episode_reward
    avg_reward = reward_total / num_episodes
    return avg_reward

train_model()
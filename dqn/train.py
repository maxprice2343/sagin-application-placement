"""Trains the DQN agent by running it within the environment, storing the
gameplay experiences and using them to improve results."""

from environment.envs.application_placement_env import ApplicationPlacementEnv
import gymnasium as gym
from gymnasium.wrappers.flatten_observation import FlattenObservation
import environment
from dqn_agent import DQNAgent
from replay_buffer import ReplayBuffer
import asyncio
import keras

MODEL_PATH = "model.keras"

async def train(render: bool):
    agent = DQNAgent()
    if render:
        env = ApplicationPlacementEnv(render_mode="human")
    else:
        env = ApplicationPlacementEnv()
    steps = 100
    for s in range(steps):
        print(f"Training Episode: {s + 1}")
        done = False
        state, _ = env.reset()
        total_reward = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = await env.step(action)
            if next_state is not None:
                agent.update_memory(state, action, reward, next_state, done)
                agent.train()
                state = next_state
                total_reward += reward

        print(f"Total reward after {s} episodes is {total_reward} and epsilon is {agent.epsilon}")
    agent.q_net.save("model.keras")

if __name__ == '__main__':
    asyncio.run(train(render=True))
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
import sys

MODEL_PATH = "model.keras"

async def train(render: bool, num_episodes: int, model_save_path: str):
    agent = DQNAgent()
    if render:
        env = ApplicationPlacementEnv(render_mode="human")
    else:
        env = ApplicationPlacementEnv()
    total_reward = 0
    try:
        for s in range(num_episodes):
            print(f"Training Episode: {s + 1}")
            done = False
            state, _ = env.reset()
            episode_reward = 0
            while not done:
                action = agent.policy(state)
                next_state, reward, done, _, _ = await env.step(action)
                if next_state is not None:
                    agent.store_experience(state, action, reward, next_state, done)
                    agent.train()
                    state = next_state
                    episode_reward += reward
            total_reward += episode_reward
            print(f"Reward for episode {s + 1} is {episode_reward} and epsilon is {agent.epsilon}")
    except KeyboardInterrupt:
        print("Training interrupted")
        agent.q_net.save(model_save_path)
        print(f"Saved model to {model_save_path}")
    else:
        print(f"Total reward for {num_episodes} episodes is {total_reward}.")
        print(f"Average reward is {total_reward / num_episodes}")
        agent.q_net.save(model_save_path)
        print(f"Saved model to {model_save_path}")

if __name__ == '__main__':
    if len(sys.argv) == 4:
        render = sys.argv[1]
        assert render == "h" or render == "n", "Render mode must be 'h' (human) or 'n' (none)"
        render = True if render == "h" else False
        num_episodes = int(sys.argv[2])
        assert num_episodes > 0, "Number of episodes must be at least 1"
        asyncio.run(train(render, num_episodes, sys.argv[3]))
    else:
        print("Please provide arguments: <Render mode> <# Episodes> <Model save path>")
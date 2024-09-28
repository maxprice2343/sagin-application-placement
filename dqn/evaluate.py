import asyncio
from environment.envs.application_placement_env import ApplicationPlacementEnv
import environment
from dqn_agent import DQNAgent
import sys
import keras

async def evaluate_training_result(agent: DQNAgent, rendering:bool = True) -> float:
    reward_total = 0.0
    num_episodes = 10
    if rendering:
        env = ApplicationPlacementEnv(render_mode="human")
    else:
        env = ApplicationPlacementEnv()

    for i in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = await env.step(action)
            episode_reward += reward
            state = next_state
        reward_total += episode_reward
    avg_reward = reward_total / num_episodes
    return avg_reward

def train():
    if len(sys.argv) > 1:
        assert isinstance(sys.argv[1], str)
        dqn = keras.models.load_model(sys.argv[1])
        agent = DQNAgent(dqn)
        asyncio.run(evaluate_training_result(agent, rendering=True))
    else:
        print("Need path to model")        

if __name__ == "__main__":
    train()
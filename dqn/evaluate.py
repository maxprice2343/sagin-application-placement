import asyncio
from environment.envs.application_placement_env import ApplicationPlacementEnv
import environment
from dqn_agent import DQNAgent
import sys
import keras

async def evaluate_training_result(agent: DQNAgent, rendering:bool, num_episodes: int) -> float:
    reward_total = 0.0
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

if __name__ == "__main__":
    if len(sys.argv) == 4:
        render = sys.argv[2]
        assert render == "h" or render == "n", "Render mode must be 'h' (human) or 'n' (none)"
        render = True if render == "h" else False
        num_episodes = int(sys.argv[3])
        assert num_episodes > 0, "Need at least 1 episode"

        dqn = keras.models.load_model(sys.argv[1])
        agent = DQNAgent(dqn)
        asyncio.run(evaluate_training_result(agent, render, num_episodes))
    else:
        print("Please provide arguments: <Model path> <Render mode> <# Episodes>")
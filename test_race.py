import gymnasium as gym
import highway_env

env = gym.make("racetrack-v0", render_mode="human")

obs, info = env.reset()

for _ in range(1000):
    action = [0]   # düz gitmeye zorla
    obs, reward, done, truncated, info = env.step(action)
    env.render()

    if done or truncated:
        obs, info = env.reset()

env.close()

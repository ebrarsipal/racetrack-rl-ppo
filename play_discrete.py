import gymnasium as gym
import highway_env
from stable_baselines3 import PPO

# 1️⃣ Environment oluştur (render açık)
env = gym.make("racetrack-v0", render_mode="human")

env.unwrapped.configure({
    "action": {
        "type": "DiscreteAction",
        "longitudinal": False,
        "lateral": True
    },
    "duration": 200,
    "collision_reward": -100,
    "offroad_terminal": True
})

env.reset()

# 2️⃣ Eğitilmiş modeli yükle
model = PPO.load("ppo_racetrack_discrete")

print("Model yüklendi. Oynatılıyor...")

obs, info = env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()

    if done or truncated:
        print("Episode bitti, yeniden başlıyor...")
        obs, info = env.reset()

import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# ---- Callback: Eğitim ilerlemesini göstermek için ----
class ProgressCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        if self.n_calls % 5000 == 0:
            print(f"Şu anki timestep: {self.num_timesteps}")
        return True


def make_env():
    env = gym.make("racetrack-v0")

    env.unwrapped.configure({
        "action": {
            "type": "DiscreteAction",
            "longitudinal": False,
            "lateral": True
        },
        "duration": 40,
        "collision_reward": -100,
        "offroad_terminal": True
    })

    env.reset()
    return Monitor(env)


env = DummyVecEnv([make_env])

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    tensorboard_log="./ppo_racetrack_tensorboard_v2/"
)

print("Eğitim başlıyor...")

model.learn(
    total_timesteps=100000,
    callback=ProgressCallback()
)

model.save("ppo_racetrack_discrete_v2")

print("Eğitim tamamlandı.")

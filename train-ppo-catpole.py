import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("CartPole-v1")

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    device="cpu",
)
model.learn(total_timesteps=50_000)

# This will create ppo_cartpole.zip in the *current directory*
model.save("ppo_cartpole")

env.close()

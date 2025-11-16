import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


ENV_ID = "LunarLander-v3"

# --- q-network maps state -> q-values for each action ---


class QNetwork(nn.Module):

    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --- simple replay buffer ---
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


def train_dqn():
    # Use CPU to avoid your sad GPU errors
    device = torch.device("cpu")

    env = gym.make(ENV_ID)
    obs, info = env.reset(seed=0)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy_net = QNetwork(obs_dim, act_dim).to(device)
    target_net = QNetwork(obs_dim, act_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)

    buffer = ReplayBuffer(capacity=100_000)

    gamma = 0.99
    batch_size = 64
    target_update_freq = 1000  # steps
    max_episodes = 1000
    max_steps_per_episode = 1000

    # Epsilon-greedy schedule
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay_steps = 100_000

    total_steps = 0

    for episode in range(max_episodes):
        obs, info = env.reset()
        episode_reward = 0.0

        for t in range(max_steps_per_episode):
            total_steps += 1

            # Compute epsilon (decays exponentially)
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(
                -1.0 * total_steps / epsilon_decay_steps
            )

            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_v = torch.tensor(
                        obs, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    q_values = policy_net(state_v)
                    action = int(torch.argmax(q_values, dim=1).item())

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            buffer.push(obs, action, reward, next_obs, done)

            obs = next_obs
            episode_reward += reward

            # Start learning once we have enough samples
            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)

                states = states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                next_states = next_states.to(device)
                dones = dones.to(device)

                # Q(s,a) for the chosen actions
                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                # Target: r + gamma * max_a' Q_target(s', a') * (1 - done)
                with torch.no_grad():
                    next_q_values = target_net(next_states).max(dim=1)[0]
                    targets = rewards + gamma * next_q_values * (1.0 - dones)

                loss = nn.functional.mse_loss(q_values, targets)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

            # Periodically update target network
            if total_steps % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        print(
            f"Episode {episode:4d} | reward: {episode_reward:8.1f} | epsilon: {epsilon:5.3f}"
        )

    env.close()

    # Save final model
    torch.save(policy_net.state_dict(), "dqn_lunar_lander.pt")
    print("Saved model to dqn_lunar_lander.pt")


def watch_trained():
    device = torch.device("cpu")
    env = gym.make(ENV_ID, render_mode="human")

    obs, info = env.reset()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy_net = QNetwork(obs_dim, act_dim).to(device)
    policy_net.load_state_dict(torch.load("dqn_lunar_lander.pt", map_location=device))
    policy_net.eval()

    for episode in range(5):
        obs, info = env.reset()
        episode_reward = 0.0
        for t in range(1000):
            with torch.no_grad():
                state_v = torch.tensor(
                    obs, dtype=torch.float32, device=device
                ).unsqueeze(0)
                q_values = policy_net(state_v)
                action = int(torch.argmax(q_values, dim=1).item())
            next_obs, reward, terminated, truncated, info = env.step(action)
            obs = next_obs
            episode_reward += reward
            if terminated or truncated:
                break
        print(f"[WATCH] Episode {episode}: reward {episode_reward:.1f}")

    env.close()


if __name__ == "__main__":
    # train_dqn()
    # After training, comment out train_dqn() and call:
    watch_trained()

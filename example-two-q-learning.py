import numpy as np

random_number_generator = np.random.default_rng()

# ---- envirement: 4x4 grid world ----

n_rows = 4
n_cols = 4
n_states = n_rows * n_cols
n_actions = 4
start_state = 0
goal_state = n_states - 1


def step(state: int, action: int) -> tuple[int, float, bool]:
    """take a step in the gridworld"""
    row, col = divmod(state, n_cols)

    if action == 0:  # up
        row = max(row - 1, 0)
    elif action == 1:  # down
        row = min(row + 1, n_rows - 1)
    elif action == 2:  # left
        col = max(col - 1, 0)
    elif action == 3:  # right
        col = min(col + 1, n_cols - 1)

    new_state = row * n_cols + col
    # Change rewards (e.g., add a small −0.01 per step to encourage shorter paths).
    reward = -0.01 if new_state == goal_state else 0.0
    done = new_state == goal_state

    return new_state, reward, done


# --- Q_learning agent ---
gamma = 0.99  # discount factor
alpha = 0.1  # learning rate

# Increase/decrease epsilon: more exploration vs more exploitation.
epsilon = 0.1  # exploration rate

n_episodes = 1000
max_steps_per_episode = 50

Q = np.zeros((n_states, n_actions))


def epsilon_greedy_policy(state: int) -> int:
    if random_number_generator.random() < epsilon or np.all(Q[state] == 0):
        return random_number_generator.integers(n_actions)
    return int(np.argmax(Q[state]))


for episode in range(n_episodes):
    state = start_state
    for t in range(max_steps_per_episode):
        action = epsilon_greedy_policy(state)
        new_state, reward, done = step(state, action)

        # Q-learning update
        best_next_action = np.max(Q[new_state])
        td_target = reward + (0 if done else gamma * best_next_action)
        td_error = td_target - Q[state][action]
        Q[state][action] += alpha * td_error

        state = new_state

        if done:
            break

# --- test the learned policy ---
state = start_state
path = [state]
for _ in range(20):
    action = int(np.argmax(Q[state]))
    state, reward, done = step(state, action)
    path.append(state)
    if done:
        break

print("Learned Q-table (rounded):")
print(np.round(Q, 2))
print("\nPath from start to goal (state indices):", path)

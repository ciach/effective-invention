"""Gridworld Q-learning walkthrough."""

import numpy as np
from tabulate import tabulate

ACTIONS = ("Up", "Down", "Left", "Right")


def step(
    state: int,
    action: int,
    *,
    n_rows: int,
    n_cols: int,
    goal_state: int,
    goal_reward: float,
    step_penalty: float,
) -> tuple[int, float, bool]:
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
    done = new_state == goal_state
    reward = goal_reward if done else step_penalty
    return new_state, reward, done


def epsilon_greedy_action(
    state: int, q_table: np.ndarray, epsilon: float, rng: np.random.Generator
) -> int:
    if rng.random() < epsilon or np.all(q_table[state] == 0):
        return int(rng.integers(q_table.shape[1]))
    return int(np.argmax(q_table[state]))


def run_q_learning(
    *,
    n_rows: int = 4,
    n_cols: int = 4,
    start_state: int = 0,
    goal_state: int | None = None,
    goal_reward: float = 1.0,
    step_penalty: float = -0.01,
    gamma: float = 0.99,
    alpha: float = 0.1,
    # Increase/decrease epsilon: more exploration vs more exploitation.
    epsilon: float = 0.1,
    n_episodes: int = 1_000,
    max_steps_per_episode: int = 50,
) -> dict:
    rng = np.random.default_rng()
    total_states = n_rows * n_cols
    goal_state = goal_state if goal_state is not None else total_states - 1
    q_table = np.zeros((total_states, len(ACTIONS)))

    for _ in range(n_episodes):
        state = start_state
        for _ in range(max_steps_per_episode):
            action = epsilon_greedy_action(state, q_table, epsilon, rng)
            new_state, reward, done = step(
                state,
                action,
                n_rows=n_rows,
                n_cols=n_cols,
                goal_state=goal_state,
                goal_reward=goal_reward,
                step_penalty=step_penalty,
            )

            best_next_action = np.max(q_table[new_state])
            td_target = reward + (0 if done else gamma * best_next_action)
            td_error = td_target - q_table[state][action]
            q_table[state][action] += alpha * td_error

            state = new_state
            if done:
                break

    path = derive_greedy_path(
        q_table,
        start_state=start_state,
        n_rows=n_rows,
        n_cols=n_cols,
        goal_state=goal_state,
        goal_reward=goal_reward,
        step_penalty=step_penalty,
    )

    return {
        "q_table": q_table,
        "path": path,
        "n_rows": n_rows,
        "n_cols": n_cols,
    }


def derive_greedy_path(
    q_table: np.ndarray,
    *,
    start_state: int,
    n_rows: int,
    n_cols: int,
    goal_state: int,
    goal_reward: float,
    step_penalty: float,
    max_horizon: int = 20,
) -> list[int]:
    state = start_state
    path = [state]
    for _ in range(max_horizon):
        action = int(np.argmax(q_table[state]))
        state, _, done = step(
            state,
            action,
            n_rows=n_rows,
            n_cols=n_cols,
            goal_state=goal_state,
            goal_reward=goal_reward,
            step_penalty=step_penalty,
        )
        path.append(state)
        if done:
            break
    return path


def format_q_table(q_table: np.ndarray, *, n_cols: int) -> str:
    rows = []
    for state, values in enumerate(q_table):
        row, col = divmod(state, n_cols)
        rows.append([state, f"({row},{col})", *values])
    headers = ["State", "Coord", *ACTIONS]
    return tabulate(rows, headers=headers, floatfmt=".2f", tablefmt="github")


def format_path(path: list[int], *, n_cols: int) -> str:
    rows = []
    for step_idx, state in enumerate(path):
        row, col = divmod(state, n_cols)
        rows.append((step_idx, state, row, col))
    return tabulate(rows, headers=["Step", "State", "Row", "Col"], tablefmt="github")


def main() -> None:
    results = run_q_learning()
    q_table = results["q_table"]
    path = results["path"]
    n_cols = results["n_cols"]

    print("Learned Q-table:")
    print(format_q_table(q_table, n_cols=n_cols))
    print("\nGreedy path from start to goal:")
    print(format_path(path, n_cols=n_cols))


if __name__ == "__main__":
    main()

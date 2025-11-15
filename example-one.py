import numpy as np
from tabulate import tabulate

random_number_generator = np.random.default_rng()

# --- Envirement: 5 slot machines with fixed (hidden) probabilities of winning ---

n_arms = 5
true_probs = random_number_generator.random(n_arms)

print("True reward probabilities (secret):", true_probs)


def pull_arm(arm: int) -> int:
    """return 1 with probability true_probs[arm] and 0 otherwise"""
    return int(random_number_generator.random() < true_probs[arm])


# --- Agent: Epsilon-greedy agent ---
# Change n_steps to a very small number and see how unstable estimates are.
n_steps = 10000

# Change epsilon to 0.0 (purely greedy). Watch it often lock on a bad arm.
epsilon = 0.1

q_estimates = np.zeros(n_arms)
counts = np.zeros(n_arms, dtype=int)
rewards_history = []

for t in range(n_steps):
    # choose action (arm) based on epsilon-greedy policy
    if random_number_generator.random() < epsilon:
        # explore: choose random arm
        arm = random_number_generator.integers(n_arms)
    else:
        # exploit: choose arm with highest estimated value
        arm = int(np.argmax(q_estimates))

    # pull chosen arm and observe reward
    reward = pull_arm(arm)
    rewards_history.append(reward)

    # update estimate of chosen arm
    counts[arm] += 1
    alpha = 1.0 / counts[arm]
    q_estimates[arm] += alpha * (reward - q_estimates[arm])

avg_reward = np.mean(rewards_history)

print("")
summary_rows = [
    (arm, true_probs[arm], q_estimates[arm], counts[arm]) for arm in range(n_arms)
]

print(
    tabulate(
        summary_rows,
        headers=["Arm", "True P(reward)", "Estimated Q", "Pulls"],
        floatfmt=".3f",
        tablefmt="github",
    )
)
print("Best arm according to agent:", int(np.argmax(q_estimates)))
print("Best arm in reality:", int(np.argmax(true_probs)))
print("Average reward:", avg_reward)

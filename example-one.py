import numpy as np
from tabulate import tabulate


def pull_arm(arm: int, true_probs: np.ndarray, rng: np.random.Generator) -> int:
    """Return 1 with probability true_probs[arm] and 0 otherwise."""
    return int(rng.random() < true_probs[arm])


def run_epsilon_greedy(
    *,
    n_arms: int = 5,
    # Change n_steps to a very small number and see how unstable estimates are.
    n_steps: int = 10000,
    # Change epsilon to 0.0 (purely greedy). Watch it often lock on a bad arm.
    epsilon: float = 0.1,
) -> dict:
    rng = np.random.default_rng()
    true_probs = rng.random(n_arms)

    q_estimates = np.zeros(n_arms)
    counts = np.zeros(n_arms, dtype=int)
    rewards_history = []

    for _ in range(n_steps):
        explore = rng.random() < epsilon
        arm = rng.integers(n_arms) if explore else int(np.argmax(q_estimates))

        reward = pull_arm(arm, true_probs, rng)
        rewards_history.append(reward)

        counts[arm] += 1
        alpha = 1.0 / counts[arm]
        q_estimates[arm] += alpha * (reward - q_estimates[arm])

    return {
        "true_probs": true_probs,
        "q_estimates": q_estimates,
        "counts": counts,
        "avg_reward": float(np.mean(rewards_history)),
    }


def print_summary(
    true_probs: np.ndarray, q_estimates: np.ndarray, counts: np.ndarray
) -> None:
    summary_rows = [
        (arm, true_probs[arm], q_estimates[arm], counts[arm])
        for arm in range(len(true_probs))
    ]

    print(
        tabulate(
            summary_rows,
            headers=["Arm", "True P(reward)", "Estimated Q", "Pulls"],
            floatfmt=".3f",
            tablefmt="github",
        )
    )


def main() -> None:
    print("True reward probabilities (secret):")
    results = run_epsilon_greedy()
    true_probs = results["true_probs"]
    q_estimates = results["q_estimates"]
    counts = results["counts"]

    print(true_probs)
    print("")
    print_summary(true_probs, q_estimates, counts)
    print("Best arm according to agent:", int(np.argmax(q_estimates)))
    print("Best arm in reality:", int(np.argmax(true_probs)))
    print("Average reward:", results["avg_reward"])


if __name__ == "__main__":
    main()

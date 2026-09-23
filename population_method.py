"""Population search with a fixed deterministic evaluation benchmark."""

import argparse
import csv
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn


DEFAULT_SEED = 1234
# Fixed benchmark scenarios, independent of the training seed.
EVAL_SEEDS = (1234, 1235, 1236, 1237, 1238)
GAME = "LunarLander-v3"
MAX_EPISODE_STEPS = 500
POP_SIZE = 40
NUM_GENERATIONS = 101
PARENT_FRAC = 0.25
SIGMA = 0.1
CHILD_EPISODES = 5


def environment_settings(env):
    """Print settings from the actual experiment environment."""
    print(f"Game {env.spec.id} settings:\n")
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    print(f"Episode Steps: {env.spec.max_episode_steps}")
    print(f"Nondeterministic: {env.spec.nondeterministic}")
    print(f"Reward Threshold: {env.spec.reward_threshold}")


class Policy(nn.Module):
    """Two-layer policy with bounded continuous actions."""

    def __init__(self, env):
        super().__init__()
        observation_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
        self.net = nn.Sequential(
            nn.Linear(observation_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
            nn.Tanh(),
        )

    def forward(self, observation):
        return self.net(observation)


@torch.no_grad()
def evaluate(policy, env, episode_seeds, rng):
    """
        Return child fitness using exponential smoothing with alpha 0.05.

        Equal weighting would normally estimate expected return more naturally
        for the same policy across rollouts. This smoothing rule is inherited 
        from the original course-project skeleton and is intentionally preserved.
    """
    device = next(policy.parameters()).device
    action_size = env.action_space.shape[0]
    action_bound = env.action_space.high[0]
    smoothed_reward = 0.0
    for episode_seed in episode_seeds:
        state, _ = env.reset(seed=episode_seed)
        episode_over = False
        cumulative_reward = 0.0
        while not episode_over:
            state_tensor = torch.as_tensor(
                state, dtype=torch.float32, device=device
            ).unsqueeze(0)
            action = policy(state_tensor).cpu().numpy().squeeze()
            action += rng.normal(0, SIGMA, size=action_size)
            action = np.clip(action, -action_bound, action_bound)
            state, reward, terminated, truncated, _ = env.step(action)
            cumulative_reward += reward
            episode_over = terminated or truncated

        smoothed_reward *= 0.95
        smoothed_reward += 0.05 * cumulative_reward
    return smoothed_reward


@torch.no_grad()
def evaluate_parent(policy, env, evaluation_seeds):
    """
        Return the arithmetic mean of deterministic, noise-free rollouts.

        The parent policy is unchanged across evaluation episodes, so the arithmetic
        mean gives each rollout equal weight when estimating its expected return.
    """
    device = next(policy.parameters()).device
    action_bound = env.action_space.high[0]
    episode_rewards = []
    for episode_seed in evaluation_seeds:
        state, _ = env.reset(seed=episode_seed)
        episode_over = False
        cumulative_reward = 0.0
        while not episode_over:
            state_tensor = torch.as_tensor(
                state, dtype=torch.float32, device=device
            ).unsqueeze(0)
            action = policy(state_tensor).cpu().numpy().squeeze()
            action = np.clip(action, -action_bound, action_bound)
            state, reward, terminated, truncated, _ = env.step(action)
            cumulative_reward += reward
            episode_over = terminated or truncated
        episode_rewards.append(cumulative_reward)
    return float(np.mean(episode_rewards))


def main(argv=None):
    """Run population search using the requested training seed and device."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="training seed (default: 1234); evaluation seeds remain fixed",
    )
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cpu",
        help="execution device (default: cpu)",
    )
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**64:
        parser.error("--seed must be an integer")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable; use --device cpu")

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    env = gym.make(GAME, continuous=True, max_episode_steps=MAX_EPISODE_STEPS)
    try:
        environment_settings(env)
        print(f"\n[Models running on {device}, training seed {args.seed}]\n")
        policy = Policy(env).to(device)
        population = [Policy(env).to(device) for _ in range(POP_SIZE)]

        output_dir = Path("results")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"population_method_seed-{args.seed}.csv"
        with output_path.open("w", newline="") as result_file:
            writer = csv.writer(result_file)
            writer.writerow(("generation", "reward"))
            for generation in range(1, NUM_GENERATIONS + 1):
                episode_seeds = rng.integers(
                    0, 2**31 - 1, size=CHILD_EPISODES
                ).tolist()
                # All children share the same seed set.
                rewards = [
                    evaluate(child, env, episode_seeds, rng)
                    for child in population
                ]
                parent_count = int(PARENT_FRAC * POP_SIZE)
                parent_indices = np.argsort(rewards)[-parent_count:]
                parent_policies = [population[i] for i in parent_indices]

                new_population = []
                for _ in range(POP_SIZE):
                    parent_index = rng.choice(parent_count)
                    child = Policy(env).to(device)
                    with torch.no_grad():
                        child.load_state_dict(
                            parent_policies[parent_index].state_dict()
                        )
                        for parameter in child.parameters():
                            parameter.add_(torch.randn_like(parameter) * SIGMA)
                    new_population.append(child)
                population = new_population

                with torch.no_grad():
                    for name, parameter in policy.named_parameters():
                        parent_parameters = [
                            parent.state_dict()[name]
                            for parent in parent_policies
                        ]
                        parameter.copy_(
                            torch.mean(torch.stack(parent_parameters), dim=0)
                        )

                evaluation_reward = evaluate_parent(policy, env, EVAL_SEEDS)
                if (generation - 1) % 4 == 0:
                    print(f"Generation: {generation}, "
                          f"reward: {evaluation_reward:.4f}")
                writer.writerow((generation, evaluation_reward))
    finally:
        env.close()


if __name__ == "__main__":
    main()

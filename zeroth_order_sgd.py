"""Fixed-sigma zeroth-order SGD with a deterministic evaluation benchmark."""

import argparse
import csv
from pathlib import Path
import time
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as onp
from jax import flatten_util

from utils import Config, format_elapsed_time, centered_ranks

# Fixed benchmark scenarios, independent of the training seed.
EVAL_SEEDS = (1234, 1235, 1236, 1237, 1238)


def main(argv=None):
    """Train fixed-sigma SGD and record the fixed-benchmark mean each iteration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=int, default=1234,
        help="training seed (default: 1234); evaluation seeds remain fixed",
    )
    args = parser.parse_args(argv)

    cfg = Config()
    cfg.seed = args.seed
    cfg.lr = 0.5  # SGD learning rate on mean parameters
    cfg.pop_size = 128
    cfg.max_iters = 101
    print(f"\nSeed: {cfg.seed}\n")
    start_training = time.time()

    env = gym.make(cfg.env_name
                   , continuous=cfg.continuous, enable_wind=False
                   , max_episode_steps=cfg.max_ep_steps
                   )
    try:
        observation_dim = env.observation_space.shape[0]
        action_dimension = env.action_space.shape[0]
        action_bound = env.action_space.high[0]

        def init_params(key):
            k1, k2 = jax.random.split(key, 2)
            w1 = jax.random.normal(k1, (observation_dim, cfg.hidden)) * 0.1
            b1 = jnp.zeros((cfg.hidden,))
            w2 = jax.random.normal(k2, (cfg.hidden, action_dimension)) * 0.1
            b2 = jnp.zeros((action_dimension,))
            return dict(w1=w1, b1=b1, w2=w2, b2=b2)

        key = jax.random.key(cfg.seed)
        # Consume a separate initialization key before the training splits.
        key, init_key = jax.random.split(key)
        params_dict = init_params(init_key)
        theta, unravel_fn = flatten_util.ravel_pytree(params_dict)

        def _forward_step(flat_params, observation):
            """Forward pass for a single observation using flattened parameters."""
            # Reconstruct params tree
            params_tree = unravel_fn(flat_params)
            # Manually evaluate MLP
            w1 = params_tree['w1']
            b1 = params_tree['b1']
            w2 = params_tree['w2']
            b2 = params_tree['b2']
            x = jnp.tanh(observation @ w1 + b1)
            x = jnp.tanh(x @ w2 + b2)

            return x * action_bound


        forward_step = jax.jit(_forward_step)  # JIT compile for speed


        # policy update is performed before any parent/child launch
        def run_episode(flat_params, random_state):
            episode_over = False
            cumulative_reward = 0.0
            state, _ = env.reset(seed=random_state)
            while not episode_over:
                action = forward_step(flat_params, state)
                state, reward, terminated, truncated, info = env.step(action)
                # Update statistics
                cumulative_reward += reward
                # check if reached goal or timeLimit
                episode_over = terminated or truncated
            return cumulative_reward

        PARAM_DIM = theta.size
        sigma = cfg.sigma0

        print(f"[TRAINING ZEROTH-ORDER OPTIMIZATION METHOD ON {cfg.env_name.upper()}]")
        output_dir = Path("results")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"zeroth_order_sgd_seed-{cfg.seed}.csv"
        with output_path.open("w", newline="") as result_file:
            writer = csv.writer(result_file)
            writer.writerow(("generation", "reward"))
            for it in range(1, cfg.max_iters + 1):
                key, eps_key, seed_key = jax.random.split(key, 3)
                eps = jax.random.normal(eps_key, shape=(cfg.pop_size, PARAM_DIM))
                random_state = jax.random.randint(seed_key, shape=(), minval=0, maxval=2**31 - 1).item()

                rewards_pos = []
                rewards_neg = []
                for j in range(cfg.pop_size):
                    rewards_pos.append(run_episode(theta + sigma * eps[j], random_state + j))
                    rewards_neg.append(run_episode(theta - sigma * eps[j], random_state + j))

                rewards_pos = jnp.asarray(rewards_pos)
                rewards_neg = jnp.asarray(rewards_neg)

                # Wierstra et al. (2014) fitness shaping
                paired_rewards = jnp.stack([rewards_pos, rewards_neg], axis=1)
                ranked_rewards = centered_ranks(paired_rewards) # * 2
                rank_difference = ranked_rewards[:, 0] - ranked_rewards[:, 1]
                # A_pos = 2 * centered_ranks(rewards_pos)
                # A_neg = 2 * centered_ranks(rewards_neg)
                # diff = A_pos - A_neg

                # gradient = (diff.reshape(-1, 1) * eps).mean(axis=0) / sigma

                gradient = (rank_difference[:, None] * eps).sum(axis=0) / paired_rewards.size
                theta += cfg.lr * gradient  # shift θ in the direction of the gradient (SGD)

                # parent policy performance
                rewards = [run_episode(theta, seed) for seed in EVAL_SEEDS]
                mean_r = onp.mean(rewards)
                writer.writerow((it, mean_r))

                if it % cfg.eval_every == 0 or it == 1:
                    print(f"Iter {it:4d} | σ={sigma:.3f} |  Mean reward {mean_r:.1f} ± {onp.std(rewards):.1f}")

                # # 1/5th success rule for σ adaptation
                # successes = (rewards_pos > rewards_neg).mean()
                # sigma *= jnp.exp(cfg.beta * (successes - cfg.success_ratio))
                # sigma *= cfg.sigma_decay  # slow geometric decay (backup)

        print("\n[TRAINING FINISHED]")
        time_taken = format_elapsed_time(time.time() - start_training)
        print(f'[SESSION TRAINING TOOK {time_taken} ] \n')
    finally:
        env.close()


if __name__ == "__main__":
    main()

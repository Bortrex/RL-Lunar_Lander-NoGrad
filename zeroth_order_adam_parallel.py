import argparse
import csv
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from jax import flatten_util
from jax.example_libraries import optimizers as jax_opt

from utils import Config, centered_ranks, format_elapsed_time

# Set CPU execution before initializing parent or spawned-worker arrays.
jax.config.update("jax_platforms", "cpu")

cfg = Config()
env = gym.make(
    cfg.env_name,
    continuous=cfg.continuous,
    enable_wind=cfg.enable_wind,
    max_episode_steps=cfg.max_episode_steps,
)

observation_dim = env.observation_space.shape[0]
action_dimension = env.action_space.shape[0]
action_bound = env.action_space.high[0]


def init_params(key):
    k1, k2 = jax.random.split(key, 2)
    w1 = jax.random.normal(k1, (observation_dim, cfg.hidden_size)) * 0.1
    b1 = jnp.zeros((cfg.hidden_size,))
    w2 = jax.random.normal(k2, (cfg.hidden_size, action_dimension)) * 0.1
    b2 = jnp.zeros((action_dimension,))
    return dict(w1=w1, b1=b1, w2=w2, b2=b2)


def _forward_step(flat_params, observation):
    """Forward pass for a single observation using flattened parameters."""
    params_tree = unravel_fn(flat_params)
    w1 = params_tree['w1']
    b1 = params_tree['b1']
    w2 = params_tree['w2']
    b2 = params_tree['b2']
    x = jnp.tanh(observation @ w1 + b1)
    x = jnp.tanh(x @ w2 + b2)

    return x * action_bound


forward_step = jax.jit(_forward_step)  # JIT compile for speed


def run_episode(flat_params, random_state):
    episode_over = False
    cumulative_reward = 0.0
    state, _ = env.reset(seed=random_state)
    while not episode_over:
        action = forward_step(flat_params, state)
        state, reward, terminated, truncated, _ = env.step(action)
        cumulative_reward += reward
        episode_over = terminated or truncated
    return cumulative_reward


def _run(params_and_seed):
    flat_p, seed = params_and_seed
    return run_episode(flat_p, seed)


# Spawned workers need the parameter structure before evaluating tasks.
_, unravel_fn = flatten_util.ravel_pytree(init_params(jax.random.key(0)))


def main(argv=None):
    """Train the fixed-sigma Adam baseline and log benchmark mean rewards."""
    global unravel_fn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=int, default=Config.seed,
        help=("training seed (default: %(default)s); "
              "evaluation seeds remain fixed"),
    )
    parser.add_argument(
        "--workers", type=int, default=mp.cpu_count(),
        help="number of parallel workers (default: all available cores)",
    )
    args = parser.parse_args(argv)
    cfg.seed = args.seed
    workers = args.workers
    print(f"\nSeed: {cfg.seed} and workers: {workers}\n")
    learning_rate = 0.15
    population_size = 256

    try:
        start_training = time.time()

        key = jax.random.key(cfg.seed)
        key, init_key = jax.random.split(key)
        params_dict = init_params(init_key)
        theta, unravel_fn = flatten_util.ravel_pytree(params_dict)

        parameter_dim = theta.size
        sigma = cfg.sigma

        opt_init, opt_update, opt_get = jax_opt.adam(learning_rate)
        opt_state = opt_init(theta)

        print(
            f"[TRAINING ZEROTH-ORDER OPTIMIZATION METHOD "
            f"ON {cfg.env_name.upper()}]"
        )
        ctx = mp.get_context("spawn")

        output_dir = Path("results")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"zeroth_order_adam_seed-{cfg.seed}.csv"
        with (
            output_path.open("w", newline="") as result_file,
            ProcessPoolExecutor(
                mp_context=ctx, max_workers=workers
            ) as executor,
        ):
            writer = csv.writer(result_file)
            writer.writerow(("generation", "reward"))
            for it in range(1, cfg.max_iters + 1):
                key, eps_key, seed_key = jax.random.split(key, 3)
                eps = jax.random.normal(
                    eps_key, shape=(population_size, parameter_dim)
                )
                random_state = jax.random.randint(
                    seed_key, shape=(), minval=0, maxval=2**31 - 1
                ).item()

                task_pos = [
                    (theta + sigma * eps[j], random_state + j)
                    for j in range(population_size)
                ]
                rewards_pos = list(executor.map(_run, task_pos))
                task_neg = [
                    (theta - sigma * eps[j], random_state + j)
                    for j in range(population_size)
                ]
                rewards_neg = list(executor.map(_run, task_neg))

                rewards_pos = jnp.asarray(rewards_pos)
                rewards_neg = jnp.asarray(rewards_neg)
                # Rank both members of every mirrored pair together.
                paired_rewards = jnp.stack([rewards_pos, rewards_neg], axis=1)
                ranked_rewards = centered_ranks(paired_rewards)
                rank_difference = ranked_rewards[:, 0] - ranked_rewards[:, 1]
                gradient = (
                    rank_difference[:, None] * eps
                ).sum(axis=0) / paired_rewards.size

                opt_state = opt_update(it - 1, -gradient, opt_state)
                theta = opt_get(opt_state)

                rewards = [run_episode(theta, seed) for seed in cfg.eval_seeds]
                mean_r = np.mean(rewards)
                writer.writerow((it, mean_r))

                if it % cfg.eval_every == 0 or it == 1:
                    print(
                        f"Iter {it:4d} | σ={sigma:.3f} |  "
                        f"Mean reward {mean_r:.1f} ± {np.std(rewards):.1f}"
                    )

        print("\n[TRAINING FINISHED]")
        time_taken = format_elapsed_time(time.time() - start_training)
        print(f'[SESSION TRAINING TOOK {time_taken} ] \n')
    finally:
        env.close()


if __name__ == "__main__":
    main()

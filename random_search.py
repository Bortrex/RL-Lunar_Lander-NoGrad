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

from utils import Config, format_elapsed_time

# Configure CPU execution before initializing parent or worker arrays.
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
action_low = env.action_space.low
action_high = env.action_space.high


@jax.jit
def linear_policy(policy, observation):
    """Apply the bias-free linear policy and clip to environment bounds."""
    return jnp.clip(policy @ observation, action_low, action_high)


def run_episode(policy, seed):
    """Return the cumulative reward of one deterministic-policy rollout."""
    state, _ = env.reset(seed=seed)
    cumulative_reward = 0.0
    episode_over = False
    while not episode_over:
        action = np.asarray(linear_policy(policy, state))
        state, reward, terminated, truncated, _ = env.step(action)
        cumulative_reward += reward
        episode_over = terminated or truncated
    return cumulative_reward


def _run(policy_and_seed):
    policy, seed = policy_and_seed
    return run_episode(policy, seed)


def main(argv=None):
    """Train ARS V1 and log fixed-benchmark evaluation after every update."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=int, default=Config.seed,
        help="training seed (default: %(default)s)",
    )
    parser.add_argument(
        "--workers", type=int, default=min(8, mp.cpu_count()),
        help="parallel worker processes (default: %(default)s)",
    )
    # Experimental starting values.
    parser.add_argument(
        "--directions", type=int, default=32,
        help="directions per iteration (initial default: %(default)s)",
    )
    parser.add_argument(
        "--noise-std", type=float, default=0.13,
        help="perturbation standard deviation (initial default: %(default)s)",
    )
    parser.add_argument(
        "--step-size", type=float, default=0.2,
        help="finite-difference step size (initial default: %(default)s)",
    )
    parser.add_argument(
        "--top-b", type=int, default=24,
        help="number of top-performing directions to use (initial default: %(default)s)",
    )
    args = parser.parse_args(argv)
    if args.workers < 1 or args.directions < 1:
        parser.error("--workers and --directions must be positive integers")
    if not np.isfinite(args.noise_std) or args.noise_std <= 0:
        parser.error("--noise-std must be finite and positive")
    if not np.isfinite(args.step_size) or args.step_size <= 0:
        parser.error("--step-size must be finite and positive")
    if not (1 <= args.top_b <= args.directions):
        parser.error("--top-b must be a positive integer no larger than --directions")

    cfg.seed = args.seed
    workers = args.workers
    directions = args.directions
    noise_std = args.noise_std
    step_size = args.step_size
    top_b = args.top_b

    type_ARS = "V1" if top_b == directions else "V1-t"  

    print(
        f"\nARS {type_ARS}: seed={cfg.seed}, workers={workers}, directions={directions}, "
        f"noise_std={noise_std}, step_size={step_size}, "
        f"iterations={cfg.max_iters}, top_b={top_b}\n"
    )
    
    try:
        start_training = time.time()
        key = jax.random.key(cfg.seed)
        policy = jnp.zeros((action_dimension, observation_dim))
        ctx = mp.get_context("spawn")
        output_dir = Path("results")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"ars_seed-{cfg.seed}.csv"

        with (
            output_path.open("w", newline="") as result_file,
            ProcessPoolExecutor(mp_context=ctx, max_workers=workers
                ) as executor,
        ):
            writer = csv.writer(result_file)
            writer.writerow(("generation", "reward"))
            for iteration in range(1, cfg.max_iters + 1):
                key, delta_key, pos_seed_key, neg_seed_key = jax.random.split(key, 4)
                delta = jax.random.normal(
                    delta_key, shape=(directions, action_dimension, observation_dim))
                # Independent draws for the two sides, not matched pair seeds.
                positive_seeds = jax.random.randint(pos_seed_key, shape=(directions,),
                    minval=0, maxval=2**31 - 1).tolist()
                negative_seeds = jax.random.randint(neg_seed_key, shape=(directions,),
                    minval=0, maxval=2**31 - 1).tolist()
                task_pos = [(policy + noise_std * delta[j], positive_seeds[j])
                    for j in range(directions)]
                rewards_pos = jnp.asarray(list(executor.map(_run, task_pos)))
                task_neg = [(policy - noise_std * delta[j], negative_seeds[j])
                    for j in range(directions)]
                rewards_neg = jnp.asarray(list(executor.map(_run, task_neg)))

                scores = np.maximum(rewards_pos, rewards_neg)
                top_idx = np.argsort(scores)[-top_b:]

                top_delta = delta[top_idx]
                top_pos = rewards_pos[top_idx]
                top_neg = rewards_neg[top_idx]

                reward_std = np.std(np.concatenate([top_pos, top_neg]))
                reward_difference = top_pos - top_neg
                
                gradient = (reward_difference[:, None, None] * top_delta
                            ).sum(axis=0) / (top_b * reward_std)
                
                # reward_difference = rewards_pos - rewards_neg
                # reward_std = np.std(
                #         np.concatenate([rewards_pos, rewards_neg])
                #     )
                # # print(reward_std, directions * reward_std, directions * noise_std)
                # # break
                # # Use finite-difference factor 1 / noise_std.
                # gradient = (
                #     reward_difference[:, None, None] * delta
                # # ).sum(axis=0) / (directions * noise_std)
                # ).sum(axis=0) / (directions * reward_std)
                # # ).sum(axis=0) / reward_std

                policy += step_size * gradient

                rewards = [
                    run_episode(policy, seed) for seed in cfg.eval_seeds
                    ]
                mean_reward = np.mean(rewards)
                writer.writerow((iteration, mean_reward))
                if iteration % cfg.eval_every == 0 or iteration == 1:
                    print(
                        f"Iter {iteration:4d} | Mean reward "
                        f"{mean_reward:.1f} ± {np.std(rewards):.1f}"
                    )

        elapsed = format_elapsed_time(time.time() - start_training)
        print(f"\n[ARS {type_ARS} TRAINING FINISHED IN {elapsed}]\n")
    finally:
        env.close()


if __name__ == "__main__":
    main()

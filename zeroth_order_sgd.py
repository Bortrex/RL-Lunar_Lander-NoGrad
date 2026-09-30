import argparse
import csv
import time
from pathlib import Path

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from jax import flatten_util

from utils import Config, centered_ranks, format_elapsed_time

# Set CPU execution before initializing policy arrays.
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


forward_step = jax.jit(_forward_step)


def run_episode(flat_params, random_state):
    """Return one episode reward using the shared environment and policy."""
    episode_over = False
    cumulative_reward = 0.0
    state, _ = env.reset(seed=random_state)
    while not episode_over:
        action = forward_step(flat_params, state)
        state, reward, terminated, truncated, _ = env.step(action)
        cumulative_reward += reward
        episode_over = terminated or truncated
    return cumulative_reward


def sample_perturbations(key, pop_size, parameter_dim, sampling="gaussian"):
    """Sample iid or orthogonal vectors with standard Gaussian marginals.

    Orthogonal exploration is an experimental variance-reduction strategy
    inspired by Choromanski et al. (2018), not a lower-MSE guarantee for
    this centered-rank estimator.
    """
    if sampling == "gaussian":
        return jax.random.normal(key, shape=(pop_size, parameter_dim))
    if sampling != "orthogonal":
        raise ValueError(f"Unknown perturbation sampling method: {sampling}")
    if pop_size > parameter_dim:
        raise ValueError(
            "Orthogonal sampling requires pop_size <= parameter_dim "
            f"(got {pop_size} > {parameter_dim})"
        )

    direction_key, radius_key = jax.random.split(key)
    matrix = jax.random.normal(direction_key, shape=(parameter_dim, pop_size))
    directions, triangular = jnp.linalg.qr(matrix, mode="reduced")
    # Positive R diagonal makes Q Haar-distributed on orthonormal frames.
    signs = jnp.where(jnp.diag(triangular) < 0, -1.0, 1.0)
    directions = directions * signs
    # Independent chi_d radii restore N(0, I) marginals, not unit vectors.
    radial_samples = jax.random.normal(
        radius_key, shape=(pop_size, parameter_dim)
    )
    radii = jnp.linalg.norm(radial_samples, axis=1)
    return directions.T * radii[:, None]


# Initialize the parameter structure used by the module-level forward pass.
_, unravel_fn = flatten_util.ravel_pytree(init_params(jax.random.key(0)))


def main(argv=None):
    """Train fixed-sigma SGD and log benchmark mean rewards."""
    global unravel_fn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=int, default=Config.seed,
        help=("training seed (default: %(default)s); "
              "evaluation seeds remain fixed"),
    )
    parser.add_argument(
        "--sampling", choices=("gaussian", "orthogonal"), default="gaussian",
        help="perturbation sampling strategy (default: gaussian)",
    )
    args = parser.parse_args(argv)

    cfg.seed = args.seed
    learning_rate = 4.0
    population_size = 128
    print(f"\nSeed: {cfg.seed}\n")
    start_training = time.time()

    try:
        key = jax.random.key(cfg.seed)
        # Consume a separate initialization key before the training splits.
        key, init_key = jax.random.split(key)
        params_dict = init_params(init_key)
        theta, unravel_fn = flatten_util.ravel_pytree(params_dict)

        parameter_dim = theta.size
        sigma = cfg.sigma

        print(
            f"[TRAINING ZEROTH-ORDER OPTIMIZATION METHOD "
            f"ON {cfg.env_name.upper()}]"
        )
        output_dir = Path("results")
        output_dir.mkdir(parents=True, exist_ok=True)
        method = (
            "zeroth_order_sgd_orthogonal"
            if args.sampling == "orthogonal" else "zeroth_order_sgd"
        )
        output_path = output_dir / f"{method}_seed-{cfg.seed}.csv"
        with output_path.open("w", newline="") as result_file:
            writer = csv.writer(result_file)
            writer.writerow(("generation", "reward"))
            for it in range(1, cfg.max_iters + 1):
                key, eps_key, seed_key = jax.random.split(key, 3)
                eps = sample_perturbations(
                    eps_key, population_size, parameter_dim, args.sampling
                )
                random_state = jax.random.randint(
                    seed_key, shape=(), minval=0, maxval=2**31 - 1
                ).item()

                rewards_pos = []
                rewards_neg = []
                for j in range(population_size):
                    rewards_pos.append(
                        run_episode(
                            theta + sigma * eps[j], random_state + j,
                        )
                    )
                    rewards_neg.append(
                        run_episode(
                            theta - sigma * eps[j], random_state + j,
                        )
                    )

                rewards_pos = jnp.asarray(rewards_pos)
                rewards_neg = jnp.asarray(rewards_neg)

                # Wierstra et al. (2014) fitness shaping
                paired_rewards = jnp.stack([rewards_pos, rewards_neg], axis=1)
                ranked_rewards = centered_ranks(paired_rewards)
                rank_difference = ranked_rewards[:, 0] - ranked_rewards[:, 1]

                gradient = (
                    rank_difference[:, None] * eps
                ).sum(axis=0) / paired_rewards.size
                theta += learning_rate * gradient

                rewards = [
                    run_episode(theta, seed)
                    for seed in cfg.eval_seeds
                ]
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

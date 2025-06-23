import time
import jax
import jax.numpy as jnp
import numpy as onp
import gymnasium as gym
import multiprocessing as mp

from jax.example_libraries import optimizers as jax_opt
from jax import flatten_util
from utils import Config, asHHMMSS, centered_ranks
from concurrent.futures import ProcessPoolExecutor

cfg = Config()
env = gym.make(cfg.env_name
               , continuous=cfg.continuous, enable_wind=False
               , max_episode_steps=cfg.max_ep_steps
               )

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


def _run(p_sigma_seed):
    flat_p, seed = p_sigma_seed
    return run_episode(flat_p, seed)


# required for global access in _run
_, unravel_fn = flatten_util.ravel_pytree(init_params(jax.random.key(0)))

if __name__ == '__main__':

    SEED = int(time.time())
    print(f"\nSeed: {SEED}\n")
    cfg.seed = SEED
    cfg.lr = 0.15  # Adam learning rate on mean parameters
    cfg.pop_size = 256

    start_training = time.time()

    key = jax.random.key(cfg.seed)
    params_dict = init_params(key)
    theta, unravel_fn = flatten_util.ravel_pytree(params_dict)

    PARAM_DIM = theta.size
    sigma = cfg.sigma0

    # setting up Adam optimizer
    opt_init, opt_update, opt_get = jax_opt.adam(cfg.lr)
    opt_state = opt_init(theta)

    print(f"[TRAINING ZEROTH-ORDER OPTIMIZATION METHOD ON {cfg.env_name.upper()}]")

    for it in range(1, cfg.max_iters + 1):
        key, eps_key, seed_key = jax.random.split(key, 3)
        eps = jax.random.normal(eps_key, shape=(cfg.pop_size, PARAM_DIM))
        random_state = jax.random.randint(seed_key, shape=(), minval=0, maxval=1e7).item()

        # run parallel processes for perturbations
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(mp_context=ctx) as executor:
            task_pos = [(theta + sigma * eps[j], random_state + j) for j in range(cfg.pop_size)]
            rewards_pos = list(executor.map(_run, task_pos))
            task_neg = [(theta - sigma * eps[j], random_state + j) for j in range(cfg.pop_size)]
            rewards_neg = list(executor.map(_run, task_neg))

        rewards_pos = jnp.asarray(rewards_pos)
        rewards_neg = jnp.asarray(rewards_neg)
        # compute gradient eq.12 Salimas et al., 2017
        # rank‑normalised advantages
        A_pos = 2 * centered_ranks(rewards_pos)
        A_neg = 2 * centered_ranks(rewards_neg)
        diff = A_pos - A_neg
        gradient = (diff.reshape(-1, 1) * eps).mean(axis=0) / sigma

        opt_state = opt_update(it - 1, -gradient, opt_state)  # Adam update
        theta = opt_get(opt_state)

        # parent policy performance
        rewards = [run_episode(theta, random_state - j) for j in range(cfg.episode_average)]
        mean_r = onp.mean(rewards)

        if it % cfg.eval_every == 0 or it == 1:
            print(f"Iter {it:4d} | σ={sigma:.3f} |  Mean reward {mean_r:.1f} ± {onp.std(rewards):.1f}")

        # 1/5th success rule for σ adaptation
        successes = (rewards_pos > rewards_neg).mean()
        sigma *= jnp.exp(cfg.beta * (successes - cfg.success_ratio))
        sigma *= cfg.sigma_decay  # slow geometric decay (backup)

    print("\n[TRAINING FINISHED]")
    time_taken = asHHMMSS(time.time() - start_training)
    print(f'[SESSION TRAINING TOOK {time_taken} ] \n')
    env.close()

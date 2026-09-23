import os
# import gym
import gymnasium as gym
import sys
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from pathlib import Path


# USE_CUDA = torch.cuda.is_available()
USE_CUDA = False

device = torch.device("cuda" if USE_CUDA else "cpu")

SEED = 1234
EVAL_EPISODES = 5
EVAL_SEEDS = [SEED + i for i in range(EVAL_EPISODES)]

GAME = "LunarLander-v3"


def environment_settings(name):
    env = gym.make(name, continuous=True,)
    spec = gym.spec(name)
    print(f"Game {name} settings:\n")
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")

    print(f"Episode Steps: {spec.max_episode_steps}")
    print(f"Nondeterministic: {spec.nondeterministic}")
    # print(f"Reward Range: {env.reward_range}")
    print(f"Reward Threshold: {spec.reward_threshold}")

# Create the Gym environment.
environment_settings(GAME)
env = gym.make(GAME
        , continuous = True
        , max_episode_steps=500)


print(f"\n[Models running on {device}.]\n")

# Define the neural network architecture
input_size = env.observation_space.shape[0]
action_space = env.action_space.shape[0]
action_bound = env.action_space.high[0]

class Policy(nn.Module):
    def __init__(self, env):
        super().__init__()

        state_spc = env.observation_space.shape[0]
        action_spc = env.action_space.shape[0]

        self.net = nn.Sequential(
                nn.Linear(state_spc, 128),
                nn.ReLU(),
                nn.Linear(128, action_spc),
                nn.Tanh(),
            )

    def forward(self, x):
        return self.net(x)

@torch.no_grad()
def evaluate(policy, episode_seeds=None):
    ''' Evaluate the policy over a number of episodes and return the smoothed reward. 
    
        Returns: `smoothed_reward` 
        
        This funciton exponentially smooth the rollout returns (alpha=0.05).
        Since the policy is fixed across these episodes, equal weighting would be a more
        natural estimate of expected return. This smoothing rule is inherited from the
        original course-project skeleton and is intentionally preserved.
    '''
    smoothed_reward = 0.0
    for epi_seed in episode_seeds:
        state, _  = env.reset(seed=epi_seed)
        episode_over = False
        cumulative_reward = 0.0

        while not episode_over:
            sts_tensor = torch.FloatTensor(state).unsqueeze(0)            
            action = policy(sts_tensor).cpu().numpy().squeeze() 
            
            action += rng.normal(0, SIGMA, size=action_space)  # Add Gaussian noise for exploration
            action = np.clip(action, -action_bound, action_bound )
            state, reward, terminated, truncated, info = env.step(action)
            
            # Update statistics
            cumulative_reward += reward
            episode_over = (terminated or truncated)

        # Per-episode statistics
        # Exponentially smooth episode returns (alpha = 0.05).
        smoothed_reward *= 0.95
        smoothed_reward += 0.05 * cumulative_reward

    return smoothed_reward

@torch.no_grad()
def evaluate_parent(policy, evaluation_seeds):
    """
    The parent policy is unchanged across evaluation episodes, so the arithmetic
    mean gives each rollout equal weight when estimating its expected return.
    """
    episode_rewards = []

    for episode_seed in evaluation_seeds:
        state, _ = env.reset(seed=episode_seed)
        episode_over = False
        cumulative_reward = 0.0

        while not episode_over:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = policy(state_tensor).cpu().numpy().squeeze()
            action = np.clip(action, -action_bound, action_bound)

            state, reward, terminated, truncated, _ = env.step(action)
            cumulative_reward += reward
            episode_over = terminated or truncated

        episode_rewards.append(cumulative_reward)
    return np.mean(episode_rewards)


# Hyperparameters
POP_SIZE = 40           # Population child size
NUM_GENERATIONS = 101   # Number of generations
PARENT_FRAC = 0.25      # Fraction of top performers to keep
SIGMA = 0.1             # Standard deviation for perturbing weights
CHILD_EPIS = 5          # number of evals per child



# Setting seed
rng = np.random.default_rng(SEED)  # generator
torch.cuda.empty_cache()
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

policy = Policy(env).to(device) # setting main agent
# each child inits with random weights
population = [Policy(env).to(device) for _ in range(POP_SIZE)]


output_dir = Path("dataPlots/")
output_dir.mkdir(parents=True, exist_ok=True)
f = open(f"dataPlots/dataPopu-{SEED}.dat", "w")
print("RETURN", f"population-{SEED}", file=f)

for gen in range(NUM_GENERATIONS):

    episode_seeds = rng.integers(0, 2**31 - 1, size=CHILD_EPIS).tolist()  # episode seeds for child evaluation
    # evaluating childs
    rewards = [evaluate(p, episode_seeds=episode_seeds) 
               for p in population]
    topK = int(PARENT_FRAC * POP_SIZE)
    parent_indices = np.argsort(rewards)[-topK:]
    parent_policies = [population[i] for i in parent_indices]
    

    new_population = []
    for _ in range(POP_SIZE):
        parent_idx = rng.choice(topK)
        child = Policy(env).to(device)
        # updating child
        with torch.no_grad():
            child.load_state_dict(parent_policies[parent_idx].state_dict())
            for param in child.parameters():                    
                param.add_(torch.randn_like(param.data) * SIGMA)
        new_population.append(child)
    
    population = new_population


    # combining all parameters from parents
    for name, param in policy.named_parameters():
        parent_params = [p.state_dict()[name] for p in parent_policies]
        params_mean = torch.mean(torch.stack(parent_params), dim=0)
        param.data.copy_(params_mean)
    
    evaluation_reward = evaluate_parent(policy, EVAL_SEEDS)
            
    if gen%4==0:
        print(f"RETURN \tepisode: {gen+1}, \t reward: {evaluation_reward:.4f}")
    print("RETURN", gen+1, evaluation_reward, file=f)
    
print('e', file=f)

# Close the environment
env.close()
f.close()
    


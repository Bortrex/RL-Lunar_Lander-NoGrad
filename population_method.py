import os
import gym
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
# Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args,
#                                                                                                                 **kwargs)

device = torch.device("cuda" if USE_CUDA else "cpu")

SEED = 1234

GAME = "LunarLander-v2"


def environment_settings(name):
    env = gym.make(name, continuous = True,)
    spec = gym.spec(name)
    print(f"Game {name} settings:\n")
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")

    print(f"Episode Steps: {spec.max_episode_steps}")
    print(f"Nondeterministic: {spec.nondeterministic}")
    print(f"Reward Range: {env.reward_range}")
    print(f"Reward Threshold: {spec.reward_threshold}")

# Create the Gym environment.
environment_settings(GAME)
env = gym.make(GAME, 
        continuous = True,)
env.seed(SEED * 2)

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
def evaluate(policy, episodes=10):
    
    average_cumulative_reward = 0.0
    for _ in range(episodes):
        state = env.reset()
        terminate = False
        cumulative_reward = 0.0

        while not terminate:
            sts_tensor = torch.FloatTensor(state).unsqueeze(0)            
            action = policy(sts_tensor).cpu().numpy().squeeze() 
            
            action += rng.normal(0, SIGMA, size=action_space)  # Add Gaussian noise for exploration
            action = np.clip(action, -action_bound, action_bound )
            state, reward, terminate, _ = env.step(action)
            
            # Update statistics
            cumulative_reward += reward

        # Per-episode statistics
        average_cumulative_reward *= 0.95
        average_cumulative_reward += 0.05 * cumulative_reward
    
    return average_cumulative_reward

@torch.no_grad()
def single_eval(policy):
    state = env.reset()
    terminate = False
    cumulative_reward = 0.0

    while not terminate:
        sts_tensor = torch.FloatTensor(state).unsqueeze(0)        
        action = policy(sts_tensor).cpu().numpy().squeeze() 
        action = np.clip(action, -action_bound, action_bound )
        state, reward, terminate, _ = env.step(action)
        
        # Update statistics
        cumulative_reward += reward
    return cumulative_reward


# Hyperparameters
POP_SIZE = 40           # Population child size
NUM_generations = 101   # Number of generations
PARENT_frac = 0.25      # Fraction of top performers to keep
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


file = Path("dataPlots/").mkdir(parents=True, exist_ok=True)
f = open(f"dataPlots/dataPopu-{SEED}.dat", "w")
print("RETURN", f"population-{SEED}", file=f)
for gen in range(NUM_generations):
    
    # evaluating childs
    rewards = [evaluate(p, episodes=CHILD_EPIS) for p in population]
    topK = int(PARENT_frac * POP_SIZE)
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
    
    single = single_eval(policy)
            
    if gen%4==0:
        print(f"RETURN \tepisode: {gen+1}, \t reward: {single:.4f}")
    print("RETURN", gen+1, single, file=f)
    
print('e', file=f)

# Close the environment
env.close()
    


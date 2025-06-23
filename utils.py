import math
from dataclasses import dataclass


@dataclass
class Config:
    env_name: str = "LunarLander-v3"
    exploration_type: str = "gaussian noise"
    continuous: bool = True
    seed: int = 1234
    hidden: int = 128
    max_ep_steps: int = 500
    pop_size: int = 128  # will be doubled after mirroring ⇒ 80 rollouts / iter

    sigma0: float = 0.5  # initial exploration std
    max_iters: int = 150
    eval_every: int = 5  # perform an evaluation on the parent policy
    episode_average: int = 5  # deterministic eval episodes

    lr: float = 0.5
    sigma_decay: float = 0.999  # geometric decay
    success_ratio: float = 0.5
    beta: float = 0.4


def asHHMMSS(s):
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dh:%dm:%ds' % (h, m, s)


def centered_ranks(x):
    y = x.ravel().argsort().argsort()
    y = y / (x.size - 1) - 0.5  # set range [-0.5, 0.5]
    return y.reshape(x.shape)


import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    env_name: str = "LunarLander-v3"
    exploration_type: str = "gaussian noise"
    continuous: bool = True
    seed: int = 1234
    hidden: int = 128
    max_ep_steps: int = 500
    pop_size: int = 128  

    sigma0: float = 0.5  # initial exploration std
    max_iters: int = 101
    eval_every: int = 5  # perform an evaluation on the parent policy
    episode_average: int = 5  # deterministic eval episodes

    lr: float = 0.5
    sigma_decay: float = 0.999  # geometric decay
    success_ratio: float = 0.5
    beta: float = 0.4


def format_elapsed_time(s):
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dh:%dm:%ds' % (h, m, s)


def centered_ranks(x):
    y = x.ravel().argsort().argsort()
    y = y / (x.size - 1) - 0.5  # set range [-0.5, 0.5]
    return y.reshape(x.shape)


def plot_learning_curves(results_dir="results", save_path=None, show=False):
    """Plot mean reward ± one population standard deviation across seeds.

    Read <method>_seed-<seed>.csv files with generation,reward columns.
    Rows are aligned by generation; runs of a method must have identical
    generation sets, otherwise ValueError is raised rather than averaging
    changing subsets of runs. A single run has zero standard deviation.
    Return (figure, axes); the caller owns the figure and may close it.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    methods = {}
    for path in sorted(Path(results_dir).glob("*_seed-*.csv")):
        match = re.fullmatch(r"(.+)_seed-([0-9]+)\.csv", path.name)
        if match is None:
            continue
        method, seed = match.group(1), int(match.group(2))
        runs = methods.setdefault(method, {})
        if seed in runs:
            raise ValueError(f"Duplicate training seed {seed} for {method}")
        rewards = {}
        with path.open(newline="") as result_file:
            reader = csv.DictReader(result_file)
            if reader.fieldnames != ["generation", "reward"]:
                raise ValueError(f"{path}: expected generation,reward header")
            for row in reader:
                try:
                    generation = int(row["generation"])
                    reward = float(row["reward"])
                    if (None in row or generation < 1
                            or generation in rewards or not np.isfinite(reward)):
                        raise ValueError
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"{path}:{reader.line_num}: invalid or duplicate data"
                    ) from error
                rewards[generation] = reward
        if not rewards:
            raise ValueError(f"{path}: no result rows")
        runs[seed] = rewards

    if not methods:
        raise ValueError(f"No seed CSV result files found in {results_dir}")

    summaries = []
    for method, runs in methods.items():
        generations = sorted(next(iter(runs.values())))
        if any(sorted(run) != generations for run in runs.values()):
            raise ValueError(f"{method}: mismatched generation sets across seeds")
        values = np.array([
            [run[generation] for generation in generations]
            for run in runs.values()
        ])
        summaries.append((method, generations, values.mean(axis=0),
                          values.std(axis=0, ddof=0), len(runs)))

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150, layout="constrained")
    # Restrained, color-blind-friendly Okabe-Ito palette.
    colors = ("#0072B2", "#E69F00", "#009E73", "#CC79A7",
              "#D55E00", "#56B4E9", "#000000")
    for index, (method, generations, mean, std, count) in enumerate(summaries):
        label = method.replace("_", " ").capitalize().replace("sgd", "SGD")
        color = colors[index % len(colors)]
        ax.plot(generations, mean, color=color, linewidth=1.8,
                label=f"{label} (n={count})")
        ax.fill_between(generations, mean - std, mean + std,
                        color=color, alpha=0.18, linewidth=0)

    ax.axhline(y=200, color="red", alpha=0.3, linestyle="--", linewidth=1.25,
               label="Reward threshold (200)")
    ax.set_title("Evaluation learning curves", fontsize=14)
    ax.set_xlabel("Generation / iteration", fontsize=12)
    ax.set_ylabel("Evaluation reward", fontsize=12)
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    if save_path is not None:
        fig.savefig(save_path, dpi=300)
    if show:
        plt.show()
    return fig, ax


def plot_parallel_scaling():
    import matplotlib.pyplot as plt
    import pandas as pd

    workers = [1, 2, 4, 8, 16]
    times_seconds = [1423, 765, 466, 343, 282]

    baseline = times_seconds[0]
    speedup = [baseline / t for t in times_seconds]
    efficiency = [s / w for s, w in zip(speedup, workers)]
    times_minutes = [t / 60 for t in times_seconds]
    ideal_minutes = [times_minutes[0] / w for w in workers]

    df = pd.DataFrame({
        "workers": workers,
        "runtime_seconds": times_seconds,
        "runtime_minutes": times_minutes,
        "speedup": speedup,
        "parallel_efficiency": efficiency,
    })

    png_path = Path("./docs/images/adam_parallel_scaling.png")
    csv_path = Path("./results/adam_parallel_scaling.csv")

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)
    ax.plot(workers, times_minutes, marker="o", linewidth=2, label="Measured runtime")
    ax.plot(workers, ideal_minutes, linestyle="--", linewidth=1.5, label="Ideal linear scaling")

    for w, mins, s, secs in zip(workers, times_minutes, speedup, times_seconds):
        m = secs // 60
        sec = secs % 60
        ax.annotate(
            f"{m}m{sec:02d}s\n{s:.2f}×",
            (w, mins),
            textcoords="offset points",
            xytext=(-0.5, 6.5),
            ha="left",
            fontsize=9,
        )

    ax.set_title(
        "Parallel Scaling of Zeroth-Order Adam\n"
        "LunarLander-v3 · seed 3210 · population 256 · 101 iterations"
    )
    ax.set_xlabel("Worker processes")
    ax.set_ylabel("Training time (minutes)")
    ax.set_xticks(workers)
    ax.set_ylim(bottom=0)
    ax.spines[['right', 'top']].set_visible(False)

    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    df.to_csv(csv_path, index=False)

    print(df.to_string(index=False, formatters={
        "runtime_minutes": "{:.2f}".format,
        "speedup": "{:.2f}".format,
        "parallel_efficiency": "{:.1%}".format,
    }))


if __name__ == "__main__":
    # plot_learning_curves(results_dir="results", save_path="docs/images/learning_curves.png", show=True)
    plot_parallel_scaling()
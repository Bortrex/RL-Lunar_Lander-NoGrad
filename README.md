# Gradient-Free Reinforcement Learning for LunarLander-v3

Gradient-Free optimization methods are able to optimize a function without computing its gradient.
In RL, this means that they allow to improve a policy without having to compute the gradient of its parameters.

This project starts with a simple population-based method and then explores zeroth-order optimization with SGD and Adam, followed by Augmented Random Search (ARS) with a much simpler linear policy.

## Environment

Experiments use the continuous version of Lunar Lander v3 [Gym Documentation](https://gymnasium.farama.org/environments/box2d/lunar_lander/), with episodes limited to 500 steps. Policies output the two continuous engine-control actions directly.

<p align="center">
<img src="https://github.com/user-attachments/assets/3621e183-082e-4489-8473-d98b6cab81fe" width="600" height="400">
<figcaption><small>The resulting .gif comes from a trained agent following population method.</small></figcaption>
</p>

The default training seed is `1234` and can be changed from the command line. At every generation or iteration, the current policy is evaluated without training noise or parameter perturbations on the same five fixed environment seeds (`1234 – 1238`), independently of the training seed. Each run stores the evaluation reward in a method-specific CSV file under `results/`.

## Methods

### Population method

The first approach uses a population of neural-network policies. At each generation, the best-performing policies are selected and used to produce a new population through parameter perturbations. This provides a simple gradient-free baseline before moving to zeroth-order gradient estimation.

#### Usage

```bash
# Default run
python population_method.py
# Select the training seed and device
python population_method.py --seed 5678 --device cuda
```

The default training seed is 1234 and the default device is CPU; CUDA is
optional and requires an available CUDA device. 



### Zeroth-order Optimization 

Zeroth-order optimization estimates an update direction from policy evaluations rather than backpropagating through the environment. For each Gaussian perturbation, the policy is evaluated at both $\theta+ \sigma \varepsilon$  and $\theta - \sigma \varepsilon$. Ranked rewards from these mirrored evaluations are then used to estimate how the policy parameters should move.

Both zeroth-order implementations use the same fixed perturbation scale and evaluation protocol. The main difference is how the estimated update is applied.

### Zeroth-order SGD

The SGD version applies the estimated zeroth-order update directly to the policy parameters. A fixed learning rate of $\alpha=4.0$ is used with a population of 128 perturbation directions.

Gaussian perturbations are used by default. An optional orthogonal Gaussian sampler is also included as an experimental variance-reduction strategy.

#### Usage

``` bash
# Default run
python zeroth_order_sgd.py
# Select the training seed and optional orthogonal Gaussian sampling
python zeroth_order_sgd.py --seed 5678 --sampling orthogonal
```


### Zeroth-order Adam

The Adam version uses the same zeroth-order gradient estimate but applies it with Adam instead of plain SGD. Adam adapts the update scale for each parameter and produces a smoother learning curve in these experiments.

The Adam experiment uses a larger population of 256 perturbation directions and evaluates them in parallel using worker processes.

#### Usage

``` bash
# Default run
python zeroth_order_adam_parallel.py
# Select the training seed and number of workers
python zeroth_order_adam_parallel.py --seed 5678 --workers 8
```

#### Parallel evaluation

Zeroth-order methods require many independent policy evaluations, which makes them well suited to parallel execution. Reusing a persistent pool of worker processes reduced the 101-iteration Adam experiment from 23m43s with one worker to 4m42s with 16 workers, corresponding to a 5.05× wall-clock speedup.

<p align="center">
<img src="docs/images/adam_parallel_scaling.png" width="800">
</p>


### Augmented Random Search (ARS)

The ARS implementation follows the random-search approach of Mania et al.
(2018). Instead of using a neural network (NN), the policy is a bias-free linear
mapping from the 8-dimensional observation directly to the 2 continuous
actions, giving only 16 trainable parameters.

The final experiment uses $N=32$ sampled directions and retains the top-$b=20$ directions for the update, with exploration noise $\nu=0.13$ and step size $\alpha=0.25$. 

#### Usage

```bash
# Default ARS V1-t run
python random_search.py
# Change the ARS search settings
python random_search.py --directions 32 --top-b 24 --noise-std 0.13 --step-size 0.30 --workers 4
```

## Experimental Settings

<!-- | Method | Framework | Population | Update | Learning rate | Noise / perturbation | 
|---|---|---:|---|---:|---:|
| Population method | PyTorch | 40 policies | Selection + mutation | — | 0.1 | 
| Zeroth-order SGD | JAX | 128 directions | SGD | 4.0 | σ = 0.5 | 
| Zeroth-order Adam | JAX | 256 directions | Adam | 0.15 | σ = 0.5 |  -->

| Method | Policy | Population | Update | Learning rate | Noise / Perturbation |
|---|---|---:|---|---:|---:|
| Population method | 8 → 128 → 2 NN | 40 policies | Selection + mutation | — | σ = 0.1 |
| Zeroth-order SGD | 8 → 128 → 2 NN | 128 directions | SGD | 4.0 | σ = 0.5 |
| Zeroth-order Adam | 8 → 128 → 2 NN | 256 directions | Adam | 0.15 | σ = 0.5 |
| ARS V1-t | 8 → 2 linear  | 32 directions, top 20 | Reward-scaled + RS | 0.25 | ν = 0.13 |



## Results

The environment is considered solved if the agent scores at least 200 points.

The figure below shows the mean evaluation reward across independent training runs. Each trained policy is evaluated on the same five fixed environment seeds.

<p align="center">
<img src="docs/images/learning_curves.png" width="900" height="500">
</p>

All four methods eventually reach the 200-point reward. Zeroth-order Adam improves the fastest and reaches the highest average reward, stabilizing close to 280. Zeroth-order SGD also reaches the benchmark reliably, but with more variation between iterations. ARS improves more gradually but eventually stabilizes around ~255, with lower variation between runs toward the end of training. The population method learns more slowly and remains more variable, but still reaches successful policies.

An important result is that ARS reaches competitive performance with a simple linear policy containing only 16 trainable parameters. It doesn't use any neural network hidden layer nor an optimizer. Its performance is sensitive to its hyperparameter values but the small policy and parallel runs make these values inexpensive to explore.

## References

- Salimans, T., Ho, J., Chen, X., Sidor, S., & Sutskever, I. (2017).
  *Evolution Strategies as a Scalable Alternative to Reinforcement Learning.*
  Main reference for the zeroth-order evolution-strategy approach, mirrored
  perturbations, fitness shaping, and scalable parallel evaluation.

- Wierstra, D., Schaul, T., Glasmachers, T., Sun, Y., Peters, J.,
  & Schmidhuber, J. (2014).
  *Natural Evolution Strategies.*
  Background on evolution strategies, fitness shaping, and optimization of
  search distributions.

- Choromanski, K., Rowland, M., Sindhwani, V., Turner, R., & Weller, A. (2018).
  *Structured Evolution with Compact Architectures for Scalable Policy Optimization.*
  Motivation for the optional orthogonal Gaussian perturbation sampler.

- Hansen, N. (2015).
  *The CMA Evolution Strategy: A Tutorial / Evolution Strategies overview.*
  Used as background when exploring step-size adaptation and the 1/5 success rule.

- Mania, H., Guy, A., & Recht, B. (2018).
  *Simple random search of static linear policies is competitive for reinforcement learning.*
  Reference for Basic Random Search and the ARS variants, including reward
  normalization, top-direction selection, and state normalization.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Author

– [@Bortrex](https://github.com/Bortrex)

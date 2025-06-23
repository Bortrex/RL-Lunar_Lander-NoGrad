# DeepRL - Gradient-free optimization method

Gradient-Free optimization methods are able to optimize a function without computing its gradient.
In RL, this means that they allow to improve a policy without having to compute the gradient of its parameters.

## Population methods

Population methods are a nice choice for complex optimization problems where traditional methods fail to perform adequately due to issues like non-differentiability, high dimensionality, or rugged search landscapes. 

## Environment
Continuous Lunar Lander [Gym Documentation](https://gymnasium.farama.org/environments/box2d/lunar_lander/)

## Results

<img src="https://github.com/user-attachments/assets/3621e183-082e-4489-8473-d98b6cab81fe" width="600" height="400">


The environment is considered solved if the agent scores at least 200 points.

The model stabilises after 60 iterations of the algorithm. 

<img src="https://github.com/user-attachments/assets/dca7ff98-4c87-4ac8-a253-e55c4f8730ac" width="800" height="400">

## *UPDATE

## Zeroth-order Optimization 

These methods estimate the gradient using function evaluations of the objective, often by perturbing the policy and observing the resulting reward. The algorithm perturbs θ in two opposite directions, evaluates them,
and uses which one achieves the best results in order to compute how to move θ so that the quality of the policy improves.

## Results

We approach two methods of optimization: SGD and Adam both very popular methods.

|| SGD | Adam |
| --- | --- | --- |
learning rate | 0.5 | 0.15 |
population | 128 | 256 |


Part of the rapid solution of the environment is due to the size of the population we use.

<img src="https://github.com/user-attachments/assets/ae111b9d-d9bf-40de-877f-2c8704c19eb1" width="800" height="400">


We included the standard deviation of the parent policy during evaluation. This allows us to observe the role played by the learning rate. Having a smoother progression with the Adam method.

<img src="https://github.com/user-attachments/assets/6891d4b0-29ae-4c16-80bb-0c00ba6138b5" width="800" height="400">


## License

Distributed under the MIT License. See `LICENSE` for more information.

## Authors

– [@Bortrex](https://github.com/Bortrex)

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

<img src="https://github.com/user-attachments/assets/3669ba23-f1b3-42e2-a984-0d3350d36e62" width="800" height="400">

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Authors

– [@Bortrex](https://github.com/Bortrex)

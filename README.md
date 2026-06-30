# Hidden-Neural-States
## Project Overview
Neural activity is typically high-dimensional and noisy making it very difficult to interpret directly. Neural population recordings often record from thousands of neurons at once, resulting in a large matrix of neural responses over time. However, in many systems, the true structure maybe of a lower-dimension, i.e., a hidden neural state evolves over time while the recorded activity is simply a noisy projection of that state. 

The goal of this project is to simulate a hidden dynamical system, generate noisy data from it, and then attempt to recover the underlying latent structure using PCA, SVD, and state-space modeling. Ultimately, it aims to recover the neural dynamics from high-dimensional and noisy neural data.
## Core Idea
The project begins with a low dimensional state: $x_t = \text{ hidden neural state}$. This state evolves according to linear dynamics described by the equation:

$$
x_{t+1} = Ax_{t} + w_{t}
$$

A simulated neural population then observes this state through a noisy linear readout:

$$
y_{t} = Cx_{t} + \varepsilon_{t}
$$

where<br>
- $x_t$ is the low dimensional hidden state
- $A$ controls the dynamics
- $C$ maps hidden states onto observed neurons
- $y_{t}$ is the observed neural population activity
- $w_t$ is process noise
- $\varepsilon_t$ is observation noise<br>

The project then uses dimensionality reduction methods to ask whether the hidden state can be recovered from the observed neural data.
## Project Structure

```text
hidden-neural-states/
│
├── README.md
├── requirements.txt
├── environment.yml
│
├── notebooks/
│   ├── 01_latent_dynamics.ipynb
│   ├── 02_neural_population_simulation.ipynb
│   ├── 03_pca_and_svd_recovery.ipynb
│   ├── 04_noise_and_dimensionality.ipynb
│   └── 05_state_space_extension.ipynb
│
├── src/
│   ├── dynamics.py
│   ├── simulation.py
│   ├── dimensionality_reduction.py
│   ├── visualization.py
│   └── metrics.py
│
├── figures/
│   ├── latent_trajectory.png
│   ├── neural_activity_heatmap.png
│   ├── pca_reconstruction.png
│   └── noise_comparison.png
│
├── reports/
│   └── project_summary.pdf
│
└── data/
    └── simulated/
```

## Folder Descriptions

### notebooks/
This folder contains the central educational reports. Each notebook adds an additional layer to the project. 

#### 01_latent_dynamics.ipynb
Introduces the hidden dynamical system. This notebook simulates simple 2D linear trajectories such as:
- stable and unstable fixed point dynamics
- rotational dynamics
- damped oscillations

#### 02_neural_population_simulation.ipynb
Generates synthetic neural activity from the hidden state. The hidden trajectory is projected onto a higher-dimensional neural population using a random observation matrix. 

### src/
This folder contains the reusable Python code

#### dynamics.py
Functions for generating 2D linear system and various types of rotation matrices. Functions include:
```text
- simulate_linear_system
- rotation_matrix
- damped_rotation_matrix
```

#### simulations.py
Functions for generating neural population activity from latent states. Functions include:
```text
- generate_observation_matrix
- simulate_neural_activity
- add_observation_noise
```

## Project Status 
This project is currently under development as a summer computational neuroscience artifact. Planned extensions include: 
* nonlinear latent dynamics 
* Poisson spike generation 
* Kalman filtering 
* comparison of PCA with factor analysis 
* application to publicly available neural datasets

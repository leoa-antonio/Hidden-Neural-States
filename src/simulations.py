import numpy as np

def generate_observation_matrix(n_neurons, latent_dim, scale=1.0, rng = None):
    if rng == None:
        rng = np.random.default_rng()

    C = rng.standard_normal((n_neurons, latent_dim))
    return C * scale

def simulate_neural_activity(X, C):
    return X @ C.T

def add_observation_noise(Y, noise_std, rng = None):
    if rng == None:
        rng = np.random.default_rng()
    return Y + rng.standard_normal(Y.shape) * noise_std
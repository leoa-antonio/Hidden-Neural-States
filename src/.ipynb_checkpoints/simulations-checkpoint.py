import numpy as np

def generate_observation_matrix(n_neurons, latent_dim, scale=1.0):
    C = np.random.randn(n_neurons, latent_dim)
    return C * scale

def simulate_neural_activity(X, C):
    return X @ C.T

def add_observation_noise(Y, noise_std):
    return Y + np.random.randn(*Y.shape) * noise_std
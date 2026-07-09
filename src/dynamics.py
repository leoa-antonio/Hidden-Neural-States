import numpy as np

def simulate_linear_system(A, x0, T, noise_std=0.0, rng = None):
    if rng == None:
        rng = np.random.default_rng()

    x_prev = x0
    X = np.zeros((T+1, x0.shape[0]))
    X[0] = x0
    for i in range(T):
        w_t = noise_std * rng.standard_normal(x0.shape)
        x_next = A @ x_prev + w_t
        X[i+1] = x_next
        x_prev = x_next
    return X

def rotation_matrix(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ]);

def damped_rotation_matrix(theta, r):
    if not (0 <= r <= 1):
        raise ValueError("r must satisfy 0 <= r <= 1")
    return r * rotation_matrix(theta)


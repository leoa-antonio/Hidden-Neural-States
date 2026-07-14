import numpy as np
import scipy.linalg as la

def reconstruction_error(Y_true, Y_hat):
    if Y_true.shape != Y_hat.shape:
        raise ValueError(f"Y_true and Y_hat must have the same shape, but was given {Y_true.shape} and {Y_hat.shape}")
    return np.mean((Y_true - Y_hat) ** 2)

def subspace_similarity(X, Z_pca):
    if X.shape[0] != Z_pca.shape[0]:
        raise ValueError(f"X and Z_pca shape must match, but received {X.shape} and {Z_pca.shape}")
    
    # Center each column across time
    n = X.shape[0]
    H = np.eye(n) - np.ones((n,n))/n
    X_c, Z_c = H @ X, H @ Z_pca
    
    # Compute QR
    Q_X, R_X = la.qr(X_c, mode="economic")
    Q_Z, R_Z = la.qr(Z_c, mode = "economic")

    # Compute SVD
    M = Q_X.T @ Q_Z
    singular_values = np.clip(np.linalg.svd(M, compute_uv=False), 0.0, 1.0)
    
    # Compute Principal Angles and Similarity Score
    principal_angles_rad = np.arccos(singular_values)
    principal_angles_deg = np.rad2deg(principal_angles_rad)
    
    similarity = np.mean(singular_values ** 2)

    return principal_angles_deg, similarity

def correlation_with_latent_state(X, Z_pca):
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array")
    if Z_pca.ndim != 2:
        raise ValueError("Z_pca must be a two-dimensional array")
    if X.shape[0] != Z_pca.shape[0]:
        raise ValueError("X and Z_pca must have the same number of time points")
    
    n_latent_variables = X.shape[1]
    n_components = Z_pca.shape[1]

    # Create Correlation Matrix
    R = np.zeros((n_latent_variables, n_components))

    # Compute Correlation Coefficients
    for i in range(n_latent_variables):
        for j in range(n_components):
            R[i, j] = np.corrcoef(X[:, i], Z_pca[:, j])[0, 1]
    
    return R

def mean_max_latent_correlation(X, Z_pca):
    R = correlation_with_latent_state(X, Z_pca)
    max_correlations = np.max(np.abs(R), axis=1)
    return np.mean(max_correlations)

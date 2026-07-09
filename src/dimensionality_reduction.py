import numpy as np
from numpy import linalg
from sklearn.decomposition import PCA

def run_pca(Y, n_components):
    pca_model = PCA(n_components)
    Z_pca = pca_model.fit_transform(Y)
    return Z_pca, pca_model

def run_svd(Y):
    mean = np.mean(Y, axis=0)
    Y_centered = Y - mean
    
    U, S, Vt = linalg.svd(Y_centered, full_matrices=False)
    return U, S, Vt, mean

def compute_variance_explained(singular_values):
    variance_explained = singular_values ** 2 / np.sum(singular_values ** 2)
    cumulative_variance = np.cumsum(variance_explained)
    return variance_explained, cumulative_variance

def reconstruct_from_pca(Z_pca, pca_model):
    return pca_model.inverse_transform(Z_pca)

def reconstruction_error(Y_true, Y_reconstructed):
    return np.mean((Y_true - Y_reconstructed) ** 2)

import numpy as np
from matplotlib.pyplot import subplots

def plot_latent_trajectory(X, ax=None, title="Latent Trajectory", label=None, show_arrows=True, arrow_step=10):
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array.")
    if X.shape[1] != 2:
        raise ValueError(f"X must have shape (T, 2), but received {X.shape}")

    if ax is None:
        fig, ax = subplots()
    else:
        fig = ax.figure
    
    ax.plot(X[:, 0], X[:, 1], label=label)
    if show_arrows:
        scale = np.ptp(X, axis=0).max()
        for i in range(0, X.shape[0] - 1, arrow_step):
            ax.arrow(
                X[i, 0],
                X[i, 1],
                X[i+1, 0] - X[i, 0],
                X[i+1, 1] - X[i, 1],
                head_width=0.03 * scale,
                length_includes_head=True
    )
    ax.set_xlabel("State 1")
    ax.set_ylabel("State 2")
    ax.set_title(title)
    if label is not None:
        ax.legend()
    ax.set_aspect("equal", adjustable = "datalim")

    return fig, ax

def plot_state_time_series(X, time=None, ax=None, title="Latent State Time Series", labels=None):
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array")
    
    n_time_steps, n_states = X.shape

    if time is None:
        time=np.arange(n_time_steps)
    else: 
        time = np.asarray(time)
        if time.ndim != 1:
            raise ValueError("time must be a one-dimensional array")
        if len(time) != n_time_steps:
            raise ValueError("time must have the same number of elements as X has rows")
    
    if labels is None:
        labels =[f"State {i+1}" for i in range(n_states)]
    elif len(labels) != n_states:
        raise ValueError("labels must contain one label for each state dimension")
    
    if ax is None:
        fig, ax = subplots()
    else:
        fig = ax.figure
    
    for i in range(n_states):
        ax.plot(time, X[:, i], label=labels[i])
    
    ax.set_xlabel("Time")
    ax.set_ylabel("State Value")
    ax.set_title(title)
    ax.legend()

    return fig, ax

def plot_neural_activity_heatmap(Y, ax=None, title= "Population Activity Heatmap", cmap="Viridis", colorbar=True):
    Y = np.asarray(Y)
    
    if Y.ndim != 2:
        raise ValueError("Y must be a two-dimensional array")
    
    if ax is None:
        fig, ax = subplots()
    else:
        fig = ax.figure
    
    im = ax.imshow(Y.T, aspect="auto", orgin="lower", cmap=cmap)

    ax.set_xlabel("Time")
    ax.set_ylabel("Neuron")
    ax.set_title(title)

    if colorbar:
        cbar = fig.colorbar(im, ax=ax, pad=0.0-2)
        cbar.set_label("Activity")
    
    return fig, ax

def plot_neuron_traces(Y, neurons=None, time=None, ax=None, title="Example Neuron Activity"):
    Y = np.asarray(Y)
    if Y.ndim != 2:
        raise ValueError("Y must be a two-dimensional array")

    n_time, n_neurons = Y.shape

    if time is None:
        time = np.arange(n_time)

    if neurons is None:
        neurons = range(min(5, n_neurons))
    
    if ax is None:
        fig, ax = subplots()
    else:
        fig = ax.figure
    
    for neuron in neurons:
        if neuron < 0 or neuron >= n_neurons:
            raise ValueError(f"Neuron index {neuron} is out of bounds.")
        
        ax.plot(time, Y[:, neuron], label=f"Neuron {neuron + 1}")
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Activity")
    ax.set_title(title)
    ax.legend()

    return fig, ax
    
def plot_pca_trajectory(X, Z_pca, arrow_step=10, figsize=(10, 5)):
    X = np.asarray(X)
    Z_pca = np.asarray(Z_pca)
    
    if X.shape[0] != Z_pca.shape[0]:
        raise ValueError("X and Z_pca must contain the same number of time steps")
    
    fig, ax = subplots(nrows=1, ncols=2, figsize=figsize)

    plot_latent_trajectory(X, ax=ax[0], title = "True Latent Trajectory", show_arrows=True, arrow_step=arrow_step)
    plot_latent_trajectory(Z_pca, ax=ax[0], title = "PCA-Recovered Latent Trajectory", show_arrows=True, arrow_step=arrow_step)

    fig.tight_layout()
    
    return fig, ax

def plot_variance_explained(variance_explained, cumulative_variance=None, k=None, ax=None, title="Variance Explained"):
    variance_explained = np.asanyarray(variance_explained)

    if variance_explained.ndim != 1:
        raise ValueError("variance_explained must be a one-dimensional array")
    
    n_components = len(variance_explained)

    if k is None:
        k = n_components
    elif not isinstance(k, int) or k <=0:
        raise ValueError("k must be a positive integer.")
    
    k = min(k, n_components)

    components = np.arange(1, k+1)

    if ax is None:
        fig, ax = subplots()
    else:
        fig = ax.figure

    ax.plot(components, variance_explained[:k], marker = 'o', label="Individual")

    if cumulative_variance is not None:
        cumulative_variance = np.asarray(cumulative_variance)

        if cumulative_variance.ndim != 1:
            raise ValueError("cumulative_variance must be a one-dimensional array.")
        
        if len(cumulative_variance) != n_components:
            raise ValueError("cumulative_variance and variance_explained must have the same length")
        
        ax.plot(components, cumulative_variance[:k], marker="o", label="Cumulative")
        ax.legend()

    ax.set_title(title)
    ax.set_xlabel("Principle Component")
    ax.set_ylabel("Proportion of Variance Explained")
    ax.set_xticks(components)
    ax.set_ylim(0, 1)

    return fig, ax

def plot_reconstruction(Y, Y_hat, neurons=None, time=None, figsize=None, title="Original vs. Reconstructed Neural Activity"):

    Y = np.asarray(Y)
    Y_hat = np.asarray(Y_hat)

    if Y.ndim != 2:
        raise ValueError("Y must be a two-dimensional array.")
    if Y_hat.ndim != 2:
        raise ValueError("Y_hat must be a two-dimensional array.")
    if Y.shape != Y_hat.shape:
        f"Y and Y_hat must have the same shape, but received {Y.shape} and {Y_hat.shape}"
    
    n_time, n_neurons = Y.shape
    if time is None:
        time = np.arange(n_time)
    else:
        time = np.asarray(time)

        if time.ndim != 1 or len(time) != n_time:
            raise ValueError("time must be one-dimensional array with one value for each time step")
    
    if neurons is None:
        neurons = list(range(min(3, n_neurons)))
    else:
        neurons = list(neurons)
    
    if len(neurons) == 0:
        raise ValueError("neurons must contain at least one neuron index")

    for neuron in neurons:
        if neuron < 0 or neuron >= n_neurons:
            raise ValueError(
                f"Neuron index {neuron} is out of bounds for "
                f"{n_neurons} neurons."
            )

    if figsize is None:
        figsize = (8, 3 * len(neurons))

    fig, ax = plt.subplots(
        nrows=len(neurons),
        ncols=1,
        figsize=figsize,
        sharex=True,
        squeeze=False,
    )

    ax = ax.ravel()

    for current_ax, neuron in zip(ax, neurons):
        current_ax.plot(
            time,
            Y[:, neuron],
            label="Original",
        )

        current_ax.plot(
            time,
            Y_hat[:, neuron],
            alpha=0.75,
            label="Reconstructed",
        )

        current_ax.set_title(f"Neuron {neuron + 1}")
        current_ax.set_ylabel("Activity")
        current_ax.legend()

    ax[-1].set_xlabel("Time")

    fig.suptitle(title)
    fig.tight_layout()

    return fig, ax

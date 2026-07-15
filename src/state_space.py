import numpy as np

def kalman_predict(x_estimate, P_estimate, A, Q):
    x_predicted = A @ x_estimate
    P_predicted = A @ P_estimate @ A.T + Q 
    return x_predicted, P_predicted

def kalman_update(x_predicted, P_predicted, y_observed, C, R):
    y_predicted = C @ x_predicted
    residual = y_observed - y_predicted
    
    S = C @ P_predicted @ C.T + R

    K = np.linalg.solve(S, C @ P_predicted).T

    x_updated = x_predicted + K @ residual

    I = np.eye(P_predicted.shape[0])
    P_updated = (I - K @ C) @ P_predicted

    return x_updated, P_updated

def kalman_filter(Y, A, C, Q, R, x0, P0):
    Y = np.asarray(Y)

    T = Y.shape[0]
    n_states = x0.shape[0]

    X_filtered = np.zeros((T, n_states))
    P_filtered = np.zeros((T, n_states, n_states))

    x_estimate = x0.copy()
    P_estimate = P0.copy()

    for t in range(T):
        x_predicted, P_predicted = kalman_predict(x_estimate, P_estimate, A, Q)
        x_estimate, P_estimate = kalman_update(x_predicted, P_predicted, Y[t], C, R)

        X_filtered[t] = x_estimate
        P_filtered[t] =  P_estimate
    
    return X_filtered, P_filtered

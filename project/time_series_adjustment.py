import numpy as np
from pykalman import KalmanFilter


def MovingAverage(initial_value, quantified_prevs, window):

    ma_prevs = []
    _ = np.hstack([initial_value, quantified_prevs])
    for i in range(len(initial_value), len(quantified_prevs)+len(initial_value)):
        if i >= window:
            total = _[i]
            for j in range(window):
                total = total + _[i - j - 1]
            ma_prevs.append(total / (window+1))
        else:
            ma_prevs.append(_[i])
    return np.array(ma_prevs)


def KalmanMA(initial_value, observations, qtfy_error, state_dim, q):

    _ = np.hstack([initial_value, observations])
    states_observed = []
    for i in range(state_dim, len(_)+1):
        state = [_[i+j-state_dim] for j in range(state_dim)]
        states_observed.append(state)
    states_observed = np.array(states_observed)

    initial_window = np.array(states_observed)[0]

    A = np.vstack([
                   np.hstack([np.zeros((state_dim-1, 1)), np.eye(state_dim-1)]),
                   np.array([1/state_dim for i in range(state_dim)])
                  ])
    Q = q * np.eye(state_dim)
    H = np.eye(state_dim)
    R = qtfy_error * np.eye(state_dim)

    kf = KalmanFilter(
        initial_state_mean=initial_window,
        initial_state_covariance=R,
        transition_matrices=A,
        transition_covariance=Q,
        observation_matrices=H,
        observation_covariance=R,
    )

    pred_state, state_cov = kf.filter(states_observed)
    pred_temp = pred_state[1:, -1]
    final_pred_state = np.hstack([initial_window, pred_temp])[len(initial_value):]
    return final_pred_state


def TimeSeriesAdjustment(observed_series, init_value, window_set, method, inf_val, classes):

    modified_dsts = []
    if method == 'MA':
        for index in range(len(classes)):
            modified_prevs = MovingAverage(initial_value=init_value[:, index],
                                           quantified_prevs=observed_series[:, index],
                                           window=max(window_set))
            modified_dsts.append(modified_prevs)

    elif method == 'KFMA':
        val_mse = inf_val[2]
        q = 10 ** (-2.5)
        for index in range(len(classes)):
            modified_prevs = KalmanMA(initial_value=init_value[:, index],
                                      observations=observed_series[:, index],
                                      qtfy_error=val_mse,
                                      state_dim=max(window_set),
                                      q=q
                                      )
            modified_dsts.append(modified_prevs)

    modified_dsts = np.array(modified_dsts).T
    modified_dsts = modified_dsts / (np.sum(modified_dsts, axis=1).reshape(-1, 1))
    return modified_dsts

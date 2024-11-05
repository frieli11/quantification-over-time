import numpy as np
import pandas as pd
from time_series_adjustment import KalmanMA


def mae(arr1, arr2):
    AEs = np.mean(abs(arr1 - arr2), axis=1)
    MAE = AEs.sum() / len(arr1)

    return MAE


def mse(arr1, arr2):
    SEs = np.mean((arr1 - arr2)**2, axis=1)
    MSE = SEs.sum() / len(arr1)

    return MSE


def val_test_split(data, prevalences, val_length):
    if val_length == 0:
        return pd.DataFrame(), data, prevalences

    else:
        validation_set = data[0]
        del data[0]
        for i in range(1, val_length):
            validation_set = pd.concat([validation_set, data[i]], ignore_index=True)
            del data[i]

        test_sets = {}
        for i in range(len(data)):
            test_sets[i] = data[i+val_length]
        del data

        test_prevalences = prevalences.iloc[val_length:, :].copy().reset_index(drop=True)

        return validation_set, test_sets, test_prevalences


def ternary_search_1d(f, left, right, eps=1e-4):

    while True:
        if abs(left - right) < eps:
            argmin = (left + right) / 2
            return f(argmin), argmin

        leftThird = left + (right - left) / 3
        rightThird = right - (right - left) / 3

        if f(leftThird) > f(rightThird):
            left = leftThird
        else:
            right = rightThird


def ternary_search_2d(func, left_x, right_x, left_y, right_y, epsilon=1e-6):

    while (right_x - left_x > epsilon) or (right_y - left_y > epsilon):
        third_x = left_x + (right_x - left_x) / 3
        third_y = left_y + (right_y - left_y) / 3
        f_third = func(third_x, third_y)

        two_third_x = right_x - (right_x - left_x) / 3
        two_third_y = right_y - (right_y - left_y) / 3
        f_two_third = func(two_third_x, two_third_y)

        if f_third < f_two_third:
            right_x = two_third_x
            right_y = two_third_y
        else:
            left_x = third_x
            left_y = third_y

    x_min = (left_x + right_x) / 2
    y_min = (left_y + right_y) / 2
    min_value = func(x_min, y_min)

    return min_value, x_min, y_min


def params_KFMA(val, val_mse):
    init_value, val_observed, val_true, classes, window = val[0], val[1], val[2], val[3], val[4]
    low = -4
    upper = -1

    def maeKFMA(_q):
        q = 10**_q
        modified_dsts = []
        for index in range(len(classes)):
            modified_prevs = KalmanMA(initial_value=init_value[:, index],
                                      observations=val_observed[:, index],
                                      qtfy_error=val_mse,
                                      state_dim=window,
                                      q=q
                                      )
            modified_dsts.append(modified_prevs)
        modified_dsts = np.array(modified_dsts).T
        modified_dsts = modified_dsts / (np.sum(modified_dsts, axis=1).reshape(-1, 1))
        MAE = mae(val_true, modified_dsts)
        return MAE

    return ternary_search_1d(maeKFMA, low, upper)

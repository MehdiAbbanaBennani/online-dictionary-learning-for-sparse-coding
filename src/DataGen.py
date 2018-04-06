import numpy as np
from numpy.random import uniform


def generate_wave(w, interval, splits):
    a = interval[0]
    b = interval[1]
    step = (b - a) / splits

    return [np.cos(2 * np.pi * w * (a + step * t)) for t in range(splits)]


def generate_waves(w, interval, split, n_samples):
    return [generate_wave(w, interval, split) for _ in range(n_samples)]


def add_noise(data, st_dev):
    return [[x + np.random.normal(scale=st_dev) for x in wave] for wave in data]


def generate(data_type, w_list, n_obs, interval, split, st_dev, coefficients_range, sparsity):
    n_base_signals = len(w_list)

    if data_type == "wave":
        base_signals = [generate_wave(w_list[i], interval, split)
                        for i in range(len(w_list))]
        coefficients_list = [generate_coefficients(dict_size=n_base_signals,
                                                   sparsity=sparsity,
                                                   coefficients_range=coefficients_range)
                             for _ in range(n_obs)]
        data = merge_signals(base_signals=base_signals, coefficients_list=coefficients_list)
        noisy_data = add_noise(data, st_dev)
    return noisy_data


def merge_signals(base_signals, coefficients_list):
    return [np.dot(coefficient, base_signals) for coefficient in coefficients_list]


def generate_coefficients(dict_size, sparsity, coefficients_range):
    bernoulli_mask = np.random.binomial(1, 1 - sparsity, dict_size)
    coefficients_list = [uniform(coefficients_range[0], coefficients_range[1])
                         for _ in range(dict_size)]
    return [bernoulli_mask[i] * coefficients_list[i] for i in range(dict_size)]

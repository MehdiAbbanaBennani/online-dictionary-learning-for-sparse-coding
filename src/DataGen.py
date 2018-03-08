import numpy as np
from src.tools import merge


def generate_wave(w, interval, splits):
    a = interval[0]
    b = interval[1]
    step = (b - a) / splits

    return [np.cos(2 * np.pi * w *(a + step * t)) for t in range(splits)]


def generate_waves(w, interval, split, n_samples):
    return [generate_wave(w, interval, split) for _ in range(n_samples)]


def add_noise(data, st_dev):
    return [[x + np.random.normal(scale=st_dev) for x in wave] for wave in data]


def generate(data_type, w_list, w_counts, interval, split, st_dev):
    if data_type == "wave" :
        data = merge([generate_waves(w_list[i], interval, split, w_counts[i])
                      for i in range(len(w_list))])
        noisy_data = add_noise(data, st_dev)
    return noisy_data

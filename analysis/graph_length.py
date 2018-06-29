import csv
import json
import os

import numpy as np
from matplotlib import pyplot as plt

import experiments.util

experiment_name = 'g_smaller_room.py'
with open(os.path.join(experiments.util.BASE_DATA_DIR, experiment_name, 'data.csv'), 'r') as f:
    generation = []
    snake_length = []
    for row in csv.DictReader(f):
        generation.append(int(row['generation']))
        snake_length.append([int(t) for t in json.loads(row['snake_length'])])
    generation = np.array(generation)
    snake_length = np.array(snake_length)

    # Do some calculations.
    # mean = np.mean(snake_length, axis=1)
    min = np.min(snake_length, axis=1)
    max = np.max(snake_length, axis=1)
    q_1 = np.percentile(snake_length, q=50, axis=1)
    q_2 = np.percentile(snake_length, q=75, axis=1)
    q_3 = np.percentile(snake_length, q=95, axis=1)

    # Plot
    plt.plot(generation, min, 'k--')
    plt.plot(generation, max, 'k--')
    plt.plot(generation, q_2, 'k-')
    plt.fill_between(generation, q_1, q_3)
    plt.show()

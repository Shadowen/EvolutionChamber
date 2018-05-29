import csv
import json
import os

import numpy as np
from matplotlib import pyplot as plt

import experiments.util

experiment_name = 'a_basic_experiment.py'
with open(os.path.join(experiments.util.BASE_DATA_DIR, experiment_name, 'data.csv'), 'r') as f:
    generation = []
    timestep = []
    for row in csv.DictReader(f):
        generation.append(int(row['generation']))
        timestep.append([int(t) for t in json.loads(row['timesteps'])])
    generation = np.array(generation)
    timestep = np.array(timestep)

    # Do some calculations.
    # mean = np.mean(timestep, axis=1)
    q_1 = np.percentile(timestep, q=25, axis=1)
    q_2 = np.percentile(timestep, q=50, axis=1)
    q_3 = np.percentile(timestep, q=75, axis=1)
    min = np.min(timestep, axis=1)
    max = np.max(timestep, axis=1)

    # Plot
    plt.plot(generation, min, 'k--')
    plt.plot(generation, max, 'k--')
    # plt.plot(generation, mean, 'k')
    plt.plot(generation, q_2, 'k-')
    plt.fill_between(generation, q_1, q_3)
    plt.show()

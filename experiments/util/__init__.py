import os

import __main__ as main

DATA_DIR = '/home/wesley/data/evolution_chamber'


def get_or_make_data_dir():
    """Gets or creates a data path for the given experiment name."""
    data_dir = os.path.join(DATA_DIR, os.path.basename(main.__file__))
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def get_empty_data_file(name):
    data_dir = get_or_make_data_dir()
    data_path = os.path.join(data_dir, name)
    return data_path

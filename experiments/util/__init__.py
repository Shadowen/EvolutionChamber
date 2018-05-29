import os

import __main__ as main

BASE_DATA_DIR: str = '/home/wesley/data/evolution_chamber'

def get_or_make_data_dir(subdir: str = None) -> str:
    """
    Gets or creates a data path for the current experiment.
    :param subdir: if specified, creates a subdirectory within the data folder.
    """
    data_dir = os.path.join(BASE_DATA_DIR, os.path.basename(main.__file__))
    if subdir is not None:
        data_dir = os.path.join(data_dir, subdir)
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def get_empty_data_file(name: str) -> str:
    data_dir = get_or_make_data_dir()
    data_path = os.path.join(data_dir, name)
    return data_path

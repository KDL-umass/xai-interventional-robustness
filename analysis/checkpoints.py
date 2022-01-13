import numpy as np

all_checkpoints = list(range(0, 100000, 10000))
all_checkpoints.extend(list(range(100000, 1000000, 100000)))
all_checkpoints.extend(list(range(1000000, 11000000, 1000000)))


def find_nearest(array, value):
    # https://stackoverflow.com/a/2566508/13989862
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


vals = np.linspace(5e4, 1e7, num=5, dtype=int)
checkpoints = [find_nearest(all_checkpoints, val) for val in vals]

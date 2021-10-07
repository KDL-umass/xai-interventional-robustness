import numpy as np


def shannon(dist):
    dist = dist + 1e-10  # eps to prevent div 0
    return -np.sum(dist * np.log2(dist))


def js_divergence(dists):
    weight = 1 / len(dists)  # equally weight distributions
    left = shannon(np.sum(weight * dists, axis=0))  # sum along columns
    right = sum([weight * shannon(dist) for dist in dists])
    return left - right

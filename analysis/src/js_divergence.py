import numpy as np


def shannon(dist):
    dist = dist + 1e-10  # eps to prevent div 0
    return -np.sum(dist * np.log2(dist))


def js_divergence(dists):
    """Standard JS Divergence, bounded b/t [0,log_2(10)]"""
    weight = 1 / len(dists)  # equally weight distributions
    left = shannon(np.sum(weight * dists, axis=0))  # sum along columns
    right = sum([weight * shannon(dist) for dist in dists])
    return left - right


def get_js_divergence_matrix(data, vanilla):
    """
    Returns mat, nmat, van_mat, intv_mat
    according to (intervention) data and vanilla data provided,
    normalize bounds to [-1,1] with vanilla set to 0.

    `mat` is normalized between [0,1].
    `nmat` is normalized between [-1,1], where 0 is the unintervened state's jsdiv.
    """
    state = data[:, 1]  # 0 indexed
    intv = data[:, 2]  # 0 indexed
    samples = data[:, 3]
    nAgents = 10

    nstates = np.max(state).astype(int) + 1
    assert nstates == len(vanilla)

    nintv = np.max(intv).astype(int) + 1

    intv_mat = np.zeros((nstates, nintv))
    for s in range(nstates):
        for i in range(nintv):
            intv_mat[s, i] = samples[s * nintv + i] / np.log2(nAgents)

    van_mat = vanilla[:, 3] / np.log2(nAgents)

    mat = np.zeros((nstates, nintv + 1))
    mat[:, 0] = van_mat
    mat[:, 1:] = intv_mat

    # normalized
    n_intv_mat = intv_mat - van_mat.reshape(-1, 1)
    n_van_mat = 0
    nmat = np.zeros((nstates, nintv + 1))
    nmat[:, 0] = n_van_mat
    nmat[:, 1:] = n_intv_mat

    return mat, nmat, van_mat, intv_mat, n_intv_mat

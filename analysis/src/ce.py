import numpy as np


def norm_cross_entropy(pi, pj):
    ce_sum = 0
    for i in range(len(pi)):
        ce_sum += (pi[i]+1e-12)*np.log2((pj[i]+1e-12)) 
    return -1*ce_sum


def norm_sym_cross_entropy(dists):
    n = len(dists)
    ces = 0
    norm = 0
    for i in range(n):
        for j in range(n):
            if j != i:
                ces += norm_cross_entropy(dists[i], dists[j])
    norm = norm_cross_entropy([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return (1/(n*(n-1))) * ces / norm


def get_ce_matrix(data, vanilla):
    """
    Returns mat, nmat, van_mat, intv_mat
    according to (intervention) data and vanilla data provided,
    normalize bounds to [-1,1] with vanilla set to 0.

    `mat` is normalized between [0,1].
    `nmat` is normalized between [-1,1], where 0 is the unintervened state's normalized symmetric cross entropy.
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
            intv_mat[s, i] = samples[s * nintv + i] #/ np.log2(nAgents)

    van_mat = vanilla[:, 3] #/ np.log2(nAgents)

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

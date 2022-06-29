import numpy as np


def compute_correlation(x, xhat):
    """
    Compute the correlation between two lists of vectors.
    """
    if len(x.shape) < 3:
        x = x[None, :, :]
        xhat = xhat[None, :, :]
    corr = np.zeros([x.shape[0], x.shape[1]])
    for j in range(x.shape[0]):
        for i in range(x.shape[1]):
            corr[j, i] = np.corrcoef(x[j, i, :], xhat[j, i, :])[0, 1]
    return np.squeeze(corr)

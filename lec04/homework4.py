import numpy as np

def fourier_matrix(N):
    k = np.arange(N).reshape((N, 1))
    n = np.arange(N).reshape((1, N))
    W = np.exp(-2j * np.pi * k * n / N)
    return W

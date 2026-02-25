import numpy as np

def minimum_Fs(f):
    return 2 * f

def omega(f, Fs):
    return 2 * np.pi * f / Fs

def pure_tone(omega, N):
    return np.cos(omega * np.arange(N))

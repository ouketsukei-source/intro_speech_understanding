import numpy as np

def major_chord(f, Fs):
    N = int(0.5 * Fs)
    t = np.arange(N) / Fs

    f_root = f
    f_third = f * 2**(4/12)
    f_fifth = f * 2**(7/12)

    x = (np.cos(2*np.pi*f_root*t) +
         np.cos(2*np.pi*f_third*t) +
         np.cos(2*np.pi*f_fifth*t)) / 3

    return x

def dft_matrix(N):
    k = np.arange(N).reshape((N, 1))
    n = np.arange(N).reshape((1, N))
    W = np.exp(-2j * np.pi * k * n / N)
    return W

def spectral_analysis(x, Fs):
    N = len(x)
    X = np.fft.fft(x)
    freqs = np.fft.fftfreq(N, 1/Fs)

    magnitude = np.abs(X)
    idx = np.argsort(magnitude)[-3:]

    f = np.sort(np.abs(freqs[idx]))

    return f[0], f[1], f[2]

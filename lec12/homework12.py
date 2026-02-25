import numpy as np


def voiced_excitation(duration, F0, Fs):
    excitation = np.zeros(duration)

    period = int(np.round(Fs / F0))

    for n in range(duration):
        if n % period == 0:
            excitation[n] = -1
        else:
            excitation[n] = 0

    return excitation


def resonator(x, F, BW, Fs):
    y = np.zeros_like(x)

    r = np.exp(-np.pi * BW / Fs)
    theta = 2 * np.pi * F / Fs

    a1 = -2 * r * np.cos(theta)
    a2 = r ** 2

    for n in range(len(x)):
        if n == 0:
            y[n] = x[n]
        elif n == 1:
            y[n] = x[n] - a1 * y[n-1]
        else:
            y[n] = x[n] - a1 * y[n-1] - a2 * y[n-2]

    return y


def synthesize_vowel(duration, F0, F1, F2, F3, F4,
                     BW1, BW2, BW3, BW4, Fs):

    x = voiced_excitation(duration, F0, Fs)

    y1 = resonator(x, F1, BW1, Fs)
    y2 = resonator(y1, F2, BW2, Fs)
    y3 = resonator(y2, F3, BW3, Fs)
    y4 = resonator(y3, F4, BW4, Fs)

    return y4

import numpy as np

def waveform_to_frames(waveform, frame_length, step):
    N = len(waveform)
    num_frames = 1 + (N - frame_length) // step

    frames = np.zeros((num_frames, frame_length))

    for m in range(num_frames):
        start = m * step
        end = start + frame_length
        frames[m, :] = waveform[start:end]

    return frames

def frames_to_mstft(frames):
    return np.abs(np.fft.fft(frames, axis=1))

def mstft_to_spectrogram(mstft):
    eps = 0.001 * np.max(mstft)
    mstft = np.maximum(eps, mstft)

    spectrogram = 20 * np.log10(mstft)

    floor = np.max(spectrogram) - 60
    spectrogram = np.maximum(spectrogram, floor)

    return spectrogram

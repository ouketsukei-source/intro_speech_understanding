import numpy as np
import librosa


def lpc(speech, frame_length, frame_skip, order):
    N = len(speech)
    num_frames = 1 + (N - frame_length) // frame_skip

    A = []
    excitation = []

    for i in range(num_frames):
        start = i * frame_skip
        end = start + frame_length
        frame = speech[start:end]

        # LPC coefficients
        a = librosa.lpc(frame, order)
        A.append(a)

        # prediction
        pred = np.zeros_like(frame)
        for n in range(order, len(frame)):
            pred[n] = -np.dot(a[1:], frame[n-order:n][::-1])

        e = frame - pred
        excitation.append(e)

    A = np.array(A)
    excitation = np.array(excitation)

    return A, excitation


def synthesize(e, A, frame_skip):
    num_frames = A.shape[0]
    frame_length = e.shape[1]

    output = np.zeros(num_frames * frame_skip)

    for i in range(num_frames):
        a = A[i]
        frame = np.zeros(frame_length)

        for n in range(frame_length):
            if n < len(a):
                frame[n] = e[i, n]
            else:
                frame[n] = e[i, n] - np.dot(a[1:], frame[n-len(a)+1:n][::-1])

        start = i * frame_skip
        output[start:start+frame_skip] = frame[-frame_skip:]

    return output


def robot_voice(excitation, T0, frame_skip):
    num_frames = excitation.shape[0]
    frame_length = excitation.shape[1]

    gains = np.sqrt(np.mean(excitation**2, axis=1))

    e_robot = np.zeros(num_frames * frame_skip)

    for i in range(num_frames):
        pulse = np.zeros(frame_skip)
        pulse[::T0] = gains[i]

        start = i * frame_skip
        e_robot[start:start+frame_skip] = pulse

    return gains, e_robot

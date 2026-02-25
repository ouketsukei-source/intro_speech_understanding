import numpy as np
import torch
import torch.nn as nn


def get_features(waveform, Fs):
    # pre-emphasis
    waveform = np.append(waveform[0], waveform[1:] - 0.97 * waveform[:-1])

    # STFT params
    frame_len = int(0.004 * Fs)
    step = int(0.002 * Fs)

    N = len(waveform)
    num_frames = 1 + (N - frame_len) // step

    features = []
    labels = []

    current_label = 0

    for i in range(num_frames):
        start = i * step
        end = start + frame_len
        frame = waveform[start:end]

        spec = np.abs(np.fft.fft(frame))
        half = spec[:len(spec)//2]

        features.append(half)

        # fake labeling every 5 frames as one segment
        labels.append(current_label)

        if (i + 1) % 5 == 0:
            current_label += 1

    features = np.array(features)
    labels = np.array(labels)

    return features, labels


def train_neuralnet(features, labels, iterations):
    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    n_features = X.shape[1]
    n_classes = int(torch.max(y).item()) + 1

    model = nn.Sequential(
        nn.LayerNorm(n_features),
        nn.Linear(n_features, n_classes)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    lossvalues = []

    for _ in range(iterations):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        lossvalues.append(loss.item())

    lossvalues = np.array(lossvalues)

    return model, lossvalues


def test_neuralnet(model, features):
    X = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(X)
        probs = torch.softmax(outputs, dim=1)

    return probs.detach().numpy()

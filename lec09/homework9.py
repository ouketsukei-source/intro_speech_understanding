import numpy as np

def VAD(waveform, Fs):
    frame_len = int(0.025 * Fs)
    step = int(0.01 * Fs)

    N = len(waveform)
    num_frames = 1 + (N - frame_len) // step

    energies = []
    frames = []

    for i in range(num_frames):
        start = i * step
        end = start + frame_len
        frame = waveform[start:end]
        energy = np.sum(frame ** 2)

        energies.append(energy)
        frames.append(frame)

    energies = np.array(energies)
    threshold = 0.1 * np.max(energies)

    segments = []
    for i in range(len(frames)):
        if energies[i] > threshold:
            segments.append(frames[i])

    return segments

def segments_to_models(segments, Fs):
    models = []

    for seg in segments:
        seg = np.append(seg[0], seg[1:] - 0.97 * seg[:-1])

        frame_len = int(0.004 * Fs)
        step = int(0.002 * Fs)

        N = len(seg)
        num_frames = 1 + (N - frame_len) // step

        specs = []

        for i in range(num_frames):
            start = i * step
            end = start + frame_len
            frame = seg[start:end]

            spectrum = np.abs(np.fft.fft(frame))
            half = spectrum[:len(spectrum)//2]
            specs.append(half)

        specs = np.array(specs)
        log_spec = 20 * np.log10(np.maximum(specs, 1e-6))
        model = np.mean(log_spec, axis=0)

        models.append(model)

    return models

def recognize_speech(testspeech, Fs, models, labels):
    segments = VAD(testspeech, Fs)
    test_models = segments_to_models(segments, Fs)

    sims = []
    outputs = []

    for test_model in test_models:
        similarities = []

        for model in models:
            num = np.dot(test_model, model)
            den = np.linalg.norm(test_model) * np.linalg.norm(model)
            similarities.append(num / den)

        similarities = np.array(similarities)
        sims.append(similarities)

        best = np.argmax(similarities)
        outputs.append(labels[best])

    sims = np.array(sims)

    return sims, outputs

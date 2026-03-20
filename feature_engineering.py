import numpy as np
import os
import glob
from scipy.fftpack import fft
from scipy.stats import skew, kurtosis, mode



def calculate_27_features(signal):

    if np.std(signal) < 1e-6:
        signal[0] += 1e-6

    features = []

    mu = np.mean(signal)
    std = np.std(signal)
    minimum = np.min(signal)
    maximum = np.max(signal)
    try:
        m_val = mode(np.round(signal, 1), keepdims=False)[0]
    except:
        m_val = mu
    rng = maximum - minimum
    centered = signal - mu

    mcr = np.sum(np.diff(np.sign(centered)) != 0) / len(signal)
    features.extend([mu, std, minimum, maximum, m_val, rng, mcr])


    L = len(signal)
    fft_vals = np.abs(fft(signal))[:L // 2]

    fft_freqs = np.linspace(0, 12.5, L // 2)

    dc = fft_vals[0]
    features.append(dc)


    if len(fft_vals) > 1:
        sorted_indices = np.argsort(fft_vals[1:])[::-1]
        top_indices = sorted_indices[:5] + 1
        top_peaks = fft_vals[top_indices]
        top_freqs = fft_freqs[top_indices]
    else:
        top_peaks = np.zeros(5)
        top_freqs = np.zeros(5)

    if len(top_peaks) < 5:
        top_peaks = np.pad(top_peaks, (0, 5 - len(top_peaks)), 'constant')
        top_freqs = np.pad(top_freqs, (0, 5 - len(top_freqs)), 'constant')
    features.extend(top_peaks)
    features.extend(top_freqs)


    energy = np.sum(signal ** 2) / L
    features.append(np.log(energy + 1e-6))  # Log Energy


    sk = skew(signal, nan_policy='omit')
    kt = kurtosis(signal, nan_policy='omit')
    rms = np.sqrt(np.mean(signal ** 2))
    abs_mean = np.mean(np.abs(signal))

    safe_abs = abs_mean if abs_mean > 1e-6 else 1.0
    safe_rms = rms if rms > 1e-6 else 1.0
    mean_sqrt_abs = np.mean(np.sqrt(np.abs(signal)))
    safe_sqrt = mean_sqrt_abs if mean_sqrt_abs > 1e-6 else 1.0


    shape_factors = [
        rms / safe_abs,
        np.max(np.abs(signal)) / safe_abs,
        np.max(np.abs(signal)) / safe_rms,
        np.max(np.abs(signal)) / safe_sqrt ** 2
    ]

    features.extend([sk, kt, rms, abs_mean])
    features.extend(shape_factors)


    return np.nan_to_num(features).tolist()


def extract_features(data, window_size=128, step_size=64):
    n_samples, n_channels = data.shape
    acc_mag = np.sqrt(np.sum(data[:, 0:3] ** 2, axis=1))
    gyro_mag = np.sqrt(np.sum(data[:, 3:6] ** 2, axis=1))

    all_features = []
    for start in range(0, n_samples - window_size, step_size):
        win_acc = acc_mag[start:start + window_size]
        win_gyro = gyro_mag[start:start + window_size]

        row = []
        row.extend(calculate_27_features(win_acc))
        row.extend(calculate_27_features(win_gyro))
        all_features.append(row)

    return np.array(all_features)


def process_and_save():
    data_dir = 'processed_data'
    files = glob.glob(os.path.join(data_dir, '*_raw.npy'))
    if not files: print(" No raw data found."); return
    print("Feature Engineering: ")

    for f in files:
        raw = np.load(f, allow_pickle=True).item()
        X_feat = extract_features(raw['X'])

        y_feat = []
        n_windows = X_feat.shape[0]
        for i in range(n_windows):
            start = i * 64
            labels = raw['y'][start: start + 128]
            if len(labels) > 0:
                y_feat.append(mode(labels, keepdims=False)[0])
            else:
                y_feat.append(raw['y'][-1])

        new_name = f.replace('_raw.npy', '_final.npy')
        np.save(new_name, {'X': X_feat, 'y': np.array(y_feat)})
        print(f"Saved {os.path.basename(new_name)} (Shape: {X_feat.shape})")


if __name__ == '__main__':
    process_and_save()
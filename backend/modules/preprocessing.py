from __future__ import annotations

import numpy as np
from scipy import signal


def apply_filter(data: np.ndarray, fs: float, filter_type: str, low_cut: float | None = None, high_cut: float | None = None, notch: float | None = None, order: int = 4) -> np.ndarray:
    if data.ndim == 1:
        data = data.reshape(1, -1)
    filtered = data.copy()
    if filter_type == "highpass":
        b, a = signal.butter(order, low_cut / (fs / 2.0), btype='high', output='ba')
        filtered = signal.filtfilt(b, a, data, axis=-1)
    elif filter_type == "lowpass":
        b, a = signal.butter(order, high_cut / (fs / 2.0), btype='low', output='ba')
        filtered = signal.filtfilt(b, a, data, axis=-1)
    elif filter_type == "bandpass":
        b, a = signal.butter(order, [low_cut / (fs / 2.0), high_cut / (fs / 2.0)], btype='bandpass', output='ba')
        filtered = signal.filtfilt(b, a, data, axis=-1)
    elif filter_type == "notch":
        b, a = signal.iirnotch(notch, Q=30, fs=fs)
        filtered = signal.filtfilt(b, a, data, axis=-1)
    else:
        raise ValueError(f"Unsupported filter type: {filter_type}")
    if notch is not None and filter_type != 'notch':
        b, a = signal.iirnotch(notch, Q=30, fs=fs)
        filtered = signal.filtfilt(b, a, filtered, axis=-1)
    return filtered

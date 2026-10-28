from __future__ import annotations

import numpy as np

from backend.config import EEG_BANDS


def hjorth_parameters(signal: np.ndarray) -> tuple[float, float, float]:
    signal = np.asarray(signal, dtype=float)
    if signal.size < 2:
        return 0.0, 0.0, 0.0
    first_diff = np.diff(signal)
    activity = np.var(signal)
    mobility = np.sqrt(np.var(first_diff) / activity) if activity > 0 else 0.0
    second_diff = np.diff(first_diff)
    complexity = np.sqrt(np.var(second_diff) / np.var(first_diff)) / mobility if mobility > 0 and np.var(first_diff) > 0 else 0.0
    return float(activity), float(mobility), float(complexity)


def spectral_entropy(psd: np.ndarray) -> float:
    psd = np.asarray(psd, dtype=float)
    psd = psd[psd > 0]
    if psd.size == 0:
        return 0.0
    psd = psd / psd.sum()
    return float(-(psd * np.log(psd + 1e-12) / np.log(2.0)).sum())


def extract_features(data: np.ndarray, fs: float, selected_channels: list[str] | None = None, feature_names: list[str] | None = None) -> dict[str, float | list[float]]:
    if data.ndim == 1:
        data = data.reshape(1, -1)
    selected_channels = selected_channels or list(range(data.shape[0]))
    feature_names = feature_names or ["mean", "variance", "rms", "hjorth_activity", "hjorth_mobility", "hjorth_complexity"]
    results: dict[str, float | list[float]] = {}
    channel_results: dict[str, dict[str, float]] = {}
    for idx, channel in enumerate(selected_channels):
        channel_values = data[idx]
        if channel_values.size == 0:
            channel_results[str(channel)] = {name: 0.0 for name in feature_names}
            continue
        band_power = {}
        freqs = np.fft.rfftfreq(channel_values.size, d=1.0 / fs)
        spectrum = np.abs(np.fft.rfft(channel_values)) ** 2
        for band_name, (low, high) in EEG_BANDS.items():
            mask = (freqs >= low) & (freqs <= high)
            band_power[band_name] = float(np.trapz(spectrum[mask], freqs[mask])) if np.any(mask) else 0.0
        values = {
            "mean": float(np.mean(channel_values)),
            "variance": float(np.var(channel_values)),
            "rms": float(np.sqrt(np.mean(np.square(channel_values)))),
            "hjorth_activity": hjorth_parameters(channel_values)[0],
            "hjorth_mobility": hjorth_parameters(channel_values)[1],
            "hjorth_complexity": hjorth_parameters(channel_values)[2],
        }
        for band_name, power in band_power.items():
            values[f"band_power_{band_name}"] = power
        channel_results[str(channel)] = values
    results["channels"] = channel_results
    return results

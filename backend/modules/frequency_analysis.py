from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal

from backend.config import EEG_BANDS


def welch_psd(data: np.ndarray, fs: float, nperseg: int | None = None, freq_range: tuple[float, float] | None = None) -> dict[str, Any]:
    if data.ndim == 1:
        data = data.reshape(1, -1)
    nperseg = nperseg or min(256, data.shape[-1])
    freqs, psd = signal.welch(data, fs=fs, nperseg=nperseg, axis=-1)
    if freq_range is not None:
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        freqs = freqs[mask]
        psd = psd[..., mask]
    band_results = {}
    for name, (low, high) in EEG_BANDS.items():
        mask = (freqs >= low) & (freqs <= high)
        if not np.any(mask):
            band_results[name] = {"available": False, "abs_power": np.nan, "relative_power": np.nan, "range": (low, high)}
            continue
        band_power = np.trapz(psd[..., mask], freqs[mask], axis=-1)
        band_results[name] = {"available": True, "abs_power": float(np.asarray(band_power).mean()) if np.asarray(band_power).size else np.nan, "range": (low, high)}
    total_power = float(np.trapz(psd, freqs, axis=-1).mean()) if psd.size else np.nan
    for key, value in band_results.items():
        if value["available"] and np.isfinite(total_power) and total_power > 0:
            value["relative_power"] = float(value["abs_power"] / total_power)
        else:
            value["relative_power"] = np.nan
    return {"freqs": freqs, "psd": np.asarray(psd), "bands": band_results, "total_power": total_power}

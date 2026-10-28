from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


VALID_UNITS = {"v": 1.0, "mv": 1e-3, "uv": 1e-6, "µv": 1e-6, "uV": 1e-6}


def validate_dataframe(df: pd.DataFrame, time_col: str | None, channel_cols: list[str], fs: float | None, amplitude_unit: str, event_col: str | None = None) -> dict[str, Any]:
    if df.empty:
        raise ValueError("The uploaded file is empty.")
    if time_col and time_col in df.columns:
        timestamps = pd.to_numeric(df[time_col], errors='coerce')
        if timestamps.isna().any():
            raise ValueError(f"The time column '{time_col}' contains some invalid values.")
        if timestamps.duplicated().any():
            raise ValueError("Duplicate timestamps were found in the time column.")
        diffs = np.diff(timestamps)
        if len(diffs) and not np.all(np.isfinite(diffs)):
            raise ValueError("The time column contains non-finite intervals.")
        if len(diffs) and np.any(diffs <= 0):
            raise ValueError("Timestamps must be strictly increasing and non-monotonic values are invalid.")
    missing = [c for c in channel_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing EEG channel columns: {missing}")
    numeric = df[channel_cols].apply(pd.to_numeric, errors='coerce')
    if numeric.isnull().values.any():
        raise ValueError("Channel data contains non-numeric values.")
    if not np.all(np.isfinite(numeric.to_numpy(dtype=float))):
        raise ValueError("Channel data contains non-finite values.")
    if amplitude_unit.lower() not in VALID_UNITS:
        raise ValueError("Amplitude units must be V, mV, or µV.")
    if event_col and event_col in df.columns:
        if df[event_col].isna().all():
            raise ValueError("The selected event column contains no valid labels.")
    sfreq = fs
    if time_col and sfreq:
        if len(df) > 1:
            intervals = np.diff(pd.to_numeric(df[time_col], errors='coerce').to_numpy(dtype=float))
            valid = intervals[intervals > 0]
            if len(valid) > 0:
                estimated = 1.0 / np.median(valid)
                if not np.isclose(estimated, sfreq, rtol=0.15, atol=0.0):
                    raise ValueError(
                        f"Estimated sampling frequency ({estimated:.3f} Hz) is inconsistent with the supplied frequency ({sfreq} Hz)."
                    )
    return {"channels": channel_cols, "sampling_frequency": float(fs or 0), "n_samples": int(len(df)), "n_channels": len(channel_cols), "amplitude_unit": amplitude_unit}


def validate_filter_configuration(fs: float, low_cut: float | None, high_cut: float | None, notch: float | None, filter_type: str) -> dict[str, Any]:
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive.")
    nyquist = fs / 2.0
    if filter_type in {"highpass", "lowpass", "bandpass"}:
        if low_cut is not None and low_cut <= 0:
            raise ValueError("Cutoff frequencies must be positive.")
        if high_cut is not None and high_cut <= 0:
            raise ValueError("Upper cutoff must be positive.")
        if filter_type == "bandpass" and low_cut is not None and high_cut is not None and low_cut >= high_cut:
            raise ValueError("Band-pass lower cutoff must be smaller than the upper cutoff.")
        if low_cut is not None and low_cut >= nyquist:
            raise ValueError("Low cutoff exceeds the Nyquist frequency for the current sampling rate.")
        if high_cut is not None and high_cut >= nyquist:
            raise ValueError("High cutoff must remain below the Nyquist frequency.")
    if notch is not None:
        if notch <= 0 or notch >= nyquist:
            raise ValueError("Notch frequency must be greater than zero and below the Nyquist frequency.")
    return {"nyquist": nyquist}

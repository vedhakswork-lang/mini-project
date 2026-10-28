from __future__ import annotations

import os
from typing import Any

import mne
import numpy as np
import pandas as pd


def load_csv_dataframe(file_obj) -> pd.DataFrame:
    file_obj.seek(0)
    return pd.read_csv(file_obj)


def load_recording_from_file(file_obj, filename: str) -> dict[str, Any]:
    ext = os.path.splitext(filename)[1].lower().lstrip('.')
    if ext == "csv":
        df = load_csv_dataframe(file_obj)
        return {"type": "csv", "dataframe": df, "filename": filename}
    if ext in {"edf", "fif"}:
        raw = mne.io.read_raw(filename if isinstance(filename, str) else file_obj.name, preload=True, verbose=False)
        data = raw.get_data()
        ch_names = raw.ch_names
        sfreq = raw.info['sfreq']
        return {
            "type": ext,
            "data": data,
            "channels": ch_names,
            "sfreq": sfreq,
            "times": raw.times,
            "filename": filename,
            "raw": raw,
        }
    raise ValueError(f"Unsupported file extension: {ext}")

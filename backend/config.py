from __future__ import annotations

import os

APP_NAME = "NeuroBridge"
DATA_DIR = os.getenv("NEUROBRIDGE_DATA_DIR", os.path.join(os.getcwd(), "data"))
MAX_FILE_SIZE_MB = 50
ALLOWED_FORMATS = {"csv", "edf", "fif"}
EEG_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

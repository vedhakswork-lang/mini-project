from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "synthetic_demo_eeg.csv"


def generate_demo_data(
    channels: int = 8,
    sampling_frequency: float = 256.0,
    duration_seconds: float = 10.0,
    random_seed: int = 42,
) -> list[list[float | str]]:
    rng = np.random.default_rng(random_seed)
    n_samples = int(sampling_frequency * duration_seconds)
    time = np.linspace(0.0, duration_seconds, n_samples, endpoint=False)

    rows: list[list[float | str]] = []
    base = [
        "time",
        *[f"EEG_CH{idx + 1}" for idx in range(channels)],
        "event_label",
    ]
    rows.append(base)

    for idx, t in enumerate(time):
        signal_row = [float(t)]
        for ch in range(channels):
            freq = 2.0 + ch * 0.8
            alpha = 9.0 + ch * 0.5
            beta = 18.0 + ch * 0.9
            noise = rng.normal(0.0, 0.15)
            value = (
                np.sin(2 * np.pi * freq * t)
                + 0.7 * np.sin(2 * np.pi * alpha * t)
                + 0.5 * np.sin(2 * np.pi * beta * t)
                + noise
            )
            signal_row.append(float(value))

        label = "rest" if idx % 2 == 0 else "move"
        signal_row.append(label)
        rows.append(signal_row)

    return rows


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = generate_demo_data()
    with OUTPUT_PATH.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(rows)
    print(f"Synthetic EEG CSV generated at: {OUTPUT_PATH}")
    print(f"Rows: {len(rows)} | Columns: {len(rows[0])}")


if __name__ == "__main__":
    main()

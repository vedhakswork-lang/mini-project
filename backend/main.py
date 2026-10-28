from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from backend.modules.classification import run_classifier
    from backend.modules.evaluation import compute_metrics
    from backend.modules.feature_extraction import extract_features
    from backend.modules.frequency_analysis import welch_psd
    from backend.modules.preprocessing import apply_filter
    from backend.modules.report_export import build_report, report_to_markdown
    from backend.modules.validation import validate_dataframe, validate_filter_configuration
except ModuleNotFoundError:
    from modules.classification import run_classifier
    from modules.evaluation import compute_metrics
    from modules.feature_extraction import extract_features
    from modules.frequency_analysis import welch_psd
    from modules.preprocessing import apply_filter
    from modules.report_export import build_report, report_to_markdown
    from modules.validation import validate_dataframe, validate_filter_configuration

app = FastAPI(title="NeuroBridge", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.dataset = None
app.state.analysis = {}


def _dataset_response(dataset: dict[str, Any]) -> dict[str, Any]:
    return {**dataset, "data": np.asarray(dataset["data"]).tolist()}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "app": "NeuroBridge"}


@app.get("/api/dataset")
def get_dataset() -> dict[str, Any]:
    if app.state.dataset is None:
        return {"status": "empty", "dataset_type": "none"}
    return _dataset_response(app.state.dataset)


@app.post("/api/upload-csv")
async def upload_csv(
    file: UploadFile = File(...),
    time_col: str = Form("time"),
    channel_cols: str = Form(...),
    sampling_frequency: float = Form(...),
    amplitude_unit: str = Form("uV"),
    event_col: str | None = Form(None),
    participant_col: str | None = Form(None),
    trial_col: str | None = Form(None),
):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported in this upload flow.")
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    channels = [c.strip() for c in channel_cols.split(",") if c.strip()]
    if not channels:
        raise HTTPException(status_code=400, detail="At least one EEG channel column must be selected.")
    validate_dataframe(df, time_col if time_col else None, channels, float(sampling_frequency), amplitude_unit, event_col)
    data = df[channels].to_numpy(dtype=float).T
    labels = df[event_col].tolist() if event_col and event_col in df.columns else []
    app.state.dataset = {
        "name": file.filename,
        "dataset_type": "recorded",
        "source": "CSV upload",
        "n_samples": int(data.shape[1]),
        "n_channels": int(data.shape[0]),
        "channels": channels,
        "data": data,
        "sampling_frequency": float(sampling_frequency),
        "amplitude_unit": amplitude_unit,
        "event_labels": labels,
        "time_col": time_col,
        "validation_status": "valid",
        "recorded_status": "recorded",
        "synthetic_banner": "",
    }
    app.state.analysis = {}
    return {"status": "ok", "dataset": _dataset_response(app.state.dataset)}


@app.post("/api/generate-synthetic")
async def generate_synthetic(
    num_channels: int = Form(8),
    sampling_frequency: float = Form(256.0),
    duration: float = Form(10.0),
    snr: float = Form(20.0),
    random_seed: int = Form(42),
):
    np.random.seed(random_seed)
    n_samples = int(sampling_frequency * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    data = np.zeros((num_channels, n_samples), dtype=float)
    for ch in range(num_channels):
        low = 1.5 + 0.8 * ch
        alpha = 10 + 0.5 * ch
        beta = 22 + 0.7 * ch
        noise = np.random.randn(n_samples) / (snr / 10.0)
        data[ch] = (
            np.sin(2 * np.pi * low * t)
            + 0.6 * np.sin(2 * np.pi * alpha * t)
            + 0.4 * np.sin(2 * np.pi * beta * t)
            + noise
        )
    channels = [f"EEG_CH_{idx + 1}" for idx in range(num_channels)]
    app.state.dataset = {
        "name": "synthetic_demo_eeg",
        "dataset_type": "synthetic",
        "source": "Demonstration generator",
        "n_samples": n_samples,
        "n_channels": num_channels,
        "channels": channels,
        "data": data,
        "sampling_frequency": float(sampling_frequency),
        "amplitude_unit": "uV",
        "event_labels": [],
        "time_col": "time",
        "validation_status": "synthetic demonstration",
        "recorded_status": "synthetic",
        "synthetic_banner": "SYNTHETIC DEMONSTRATION DATA — NOT A REAL EEG RECORDING.",
    }
    app.state.analysis = {}
    return {"status": "ok", "dataset": _dataset_response(app.state.dataset)}


@app.post("/api/process-signal")
async def process_signal(
    filter_type: str = Form("bandpass"),
    low_cut: float | None = Form(None),
    high_cut: float | None = Form(None),
    notch: float | None = Form(None),
    order: int = Form(4),
):
    dataset = app.state.dataset
    if dataset is None:
        raise HTTPException(status_code=400, detail="No dataset is loaded.")
    data = np.asarray(dataset["data"], dtype=float)
    fs = float(dataset["sampling_frequency"])
    validate_filter_configuration(fs, low_cut, high_cut, notch, filter_type)
    filtered = apply_filter(data, fs, filter_type, low_cut, high_cut, notch, order)
    app.state.analysis["processed"] = {
        "filter_type": filter_type,
        "low_cut": low_cut,
        "high_cut": high_cut,
        "notch": notch,
        "order": order,
        "filtered": filtered.tolist(),
    }
    return {"status": "ok", "filtered": filtered.tolist(), "config": app.state.analysis["processed"]}


@app.post("/api/frequency-analysis")
async def frequency_analysis():
    dataset = app.state.dataset
    if dataset is None:
        raise HTTPException(status_code=400, detail="No dataset is loaded.")
    data = np.asarray(dataset["data"], dtype=float)
    fs = float(dataset["sampling_frequency"])
    results = []
    for channel_index in range(data.shape[0]):
        channel_data = data[channel_index]
        result = welch_psd(channel_data, fs=fs)
        results.append({"channel": dataset["channels"][channel_index], "freqs": result["freqs"].tolist(), "psd": result["psd"][0].tolist(), "bands": result["bands"]})
    app.state.analysis["frequency"] = {"results": results, "fs": fs}
    return {"status": "ok", "results": results}


@app.post("/api/feature-extraction")
async def feature_extraction():
    dataset = app.state.dataset
    if dataset is None:
        raise HTTPException(status_code=400, detail="No dataset is loaded.")
    features = extract_features(np.asarray(dataset["data"], dtype=float), float(dataset["sampling_frequency"]), dataset["channels"])
    app.state.analysis["features"] = features
    return {"status": "ok", "features": features}


@app.post("/api/classification")
async def classification(payload: dict[str, Any]):
    dataset = app.state.dataset
    if dataset is None:
        raise HTTPException(status_code=400, detail="No dataset is loaded.")
    labels = payload.get("labels")
    if not labels:
        raise HTTPException(status_code=400, detail="Classification requires a valid list of labels.")
    if len(labels) < 2:
        raise HTTPException(status_code=400, detail="At least two label values are required.")
    features = extract_features(np.asarray(dataset["data"], dtype=float), float(dataset["sampling_frequency"]), dataset["channels"])
    channel_values = list(features["channels"].values())
    X = []
    for item in channel_values:
        row = []
        for key in ["mean", "variance", "rms", "hjorth_activity", "hjorth_mobility", "hjorth_complexity"]:
            row.append(float(item[key]))
        X.append(row)
    X = np.asarray(X, dtype=float)
    y = np.asarray(labels[: X.shape[0]], dtype=object)
    if X.shape[0] != y.shape[0]:
        raise HTTPException(status_code=400, detail="Label count does not match the available samples.")
    result = run_classifier(X, y, model_name=payload.get("model_name", "lda"), test_size=float(payload.get("test_size", 0.25)))
    app.state.analysis["classification"] = result
    return {"status": "ok", "result": result}


@app.post("/api/evaluate")
async def evaluate(payload: dict[str, Any]):
    y_true = payload.get("y_true")
    y_pred = payload.get("y_pred")
    if not y_true or not y_pred:
        raise HTTPException(status_code=400, detail="Actual and predicted labels are required.")
    metrics = compute_metrics(y_true, y_pred)
    app.state.analysis["evaluation"] = metrics
    return {"status": "ok", "metrics": metrics}


@app.post("/api/export-report")
async def export_report(payload: dict[str, Any] | None = None):
    dataset = app.state.dataset or {}
    report_dataset = {key: value for key, value in dataset.items() if key != "data"}
    report = jsonable_encoder(build_report(report_dataset, app.state.analysis))
    report_payload = {
        "status": "ok",
        "report": report,
        "markdown": report_to_markdown(report),
        "json": json.dumps(report, indent=2),
    }
    return report_payload


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)

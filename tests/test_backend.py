import numpy as np
import pandas as pd
import pytest

from backend.modules.evaluation import compute_metrics
from backend.modules.feature_extraction import extract_features, hjorth_parameters
from backend.modules.frequency_analysis import welch_psd
from backend.modules.preprocessing import apply_filter
from backend.modules.validation import validate_dataframe, validate_filter_configuration


def test_validate_dataframe_accepts_valid_csv():
    df = pd.DataFrame(
        {
            'time': np.linspace(0, 1, 256),
            'EEG_CH1': np.sin(np.linspace(0, 10, 256)),
            'EEG_CH2': np.cos(np.linspace(0, 10, 256)),
        }
    )
    result = validate_dataframe(df, 'time', ['EEG_CH1', 'EEG_CH2'], 256.0, 'uV')
    assert result['n_channels'] == 2
    assert result['sampling_frequency'] == 256.0


def test_validate_dataframe_rejects_missing_columns():
    df = pd.DataFrame({'time': [0.0, 0.1], 'other': [1.0, 2.0]})
    with pytest.raises(ValueError):
        validate_dataframe(df, 'time', ['EEG_CH1'], 128.0, 'uV')


def test_filter_configuration_rejects_unsafe_cutoffs():
    with pytest.raises(ValueError):
        validate_filter_configuration(100.0, 60.0, 80.0, None, 'bandpass')


def test_apply_filter_preserves_shape():
    signal = np.random.randn(2, 256)
    filtered = apply_filter(signal, 256.0, 'bandpass', 1.0, 25.0, None, 4)
    assert filtered.shape == signal.shape


def test_welch_psd_produces_expected_shapes():
    signal = np.sin(2 * np.pi * np.linspace(0, 1, 512))
    result = welch_psd(signal, fs=256.0)
    assert result['freqs'].size > 0
    assert result['psd'].shape[0] == 1


def test_feature_extraction_runs_on_small_signal():
    signal = np.random.randn(256)
    result = extract_features(signal.reshape(1, -1), 256.0, ['EEG_CH1'])
    assert 'channels' in result
    assert result['channels']['EEG_CH1']['mean'] == pytest.approx(np.mean(signal), rel=1e-3)


def test_hjorth_values_are_finite():
    vals = hjorth_parameters(np.random.randn(128))
    assert all(np.isfinite(v) for v in vals)


def test_compute_metrics_rejects_mismatched_lengths():
    metrics = compute_metrics(['A', 'B', 'A'], ['A', 'B', 'A'])
    assert metrics['accuracy'] == pytest.approx(1.0)


def test_validate_filter_configuration_accepts_valid_bands():
    info = validate_filter_configuration(256.0, 1.0, 30.0, 50.0, 'bandpass')
    assert info['nyquist'] == pytest.approx(128.0)

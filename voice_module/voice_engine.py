# voice_module/voice_engine.py

import io
import math
import numpy as np
import librosa
import soundfile as sf


def _extract_voice_features_from_bytes(raw_bytes: bytes):
    """
    Low-level feature extractor.
    Returns (sr, rms_mean, rms_std, pitch_std) or None on failure.
    """
    try:
        data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
    except Exception:
        return None

    # Stereo -> mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Limit duration to 10 seconds
    max_duration = 10
    max_samples = sr * max_duration
    if data.shape[0] > max_samples:
        data = data[:max_samples]

    # Too short? (e.g., < 0.4s)
    if data.shape[0] < sr * 0.4:
        return None

    # Loudness features
    rms = librosa.feature.rms(y=data)[0]
    rms_mean = float(np.mean(rms))        # overall loudness
    rms_std = float(np.std(rms))          # loudness variability

    # Pitch (F0) features
    try:
        f0 = librosa.yin(data, fmin=80, fmax=400, sr=sr)
        f0 = f0[~np.isnan(f0)]
        if f0.size > 0:
            pitch_std = float(np.std(f0))     # pitch variability
        else:
            pitch_std = 0.0
    except Exception:
        pitch_std = 0.0

    return sr, rms_mean, rms_std, pitch_std


def compute_baseline_from_bytes(raw_bytes: bytes) -> dict:
    """
    Compute baseline voice features from a calm recording.

    This should be called once when the user records their calm / neutral voice.
    The returned 'baseline' dict should be stored per user/session.

    Output:
        {
          "ok": True/False,
          "baseline": {
              "rms_mean": ...,
              "rms_std": ...,
              "pitch_std": ...
          }  # present only when ok=True
        }
    """
    feats = _extract_voice_features_from_bytes(raw_bytes)
    if feats is None:
        return {
            "ok": False,
            "error": "Invalid or too short baseline recording."
        }

    _, rms_mean, rms_std, pitch_std = feats

    return {
        "ok": True,
        "baseline": {
            "rms_mean": rms_mean,
            "rms_std": rms_std,
            "pitch_std": pitch_std
        }
    }


def analyze_voice_stress_from_bytes(raw_bytes: bytes, baseline: dict = None) -> dict:
    """
    Analyze stress level from audio bytes, optionally using a per-user baseline.

    baseline: dict with keys "rms_mean", "rms_std", "pitch_std".
              If None, a generic baseline is used (less personalized).

    Output example:
        {
            "stress_score": 72.57,   # 0–100
            "state": "High Stress",  # Calm / Mild / Moderate / High
            "info": "Computed relative to baseline voice using loudness + pitch variability. Audio is not stored."
        }
    """
    feats = _extract_voice_features_from_bytes(raw_bytes)
    if feats is None:
        return {
            "stress_score": None,
            "state": "Too short",
            "info": "Recording too short or invalid to analyze. Please record at least 1–2 seconds."
        }

    sr, rms_mean, rms_std, pitch_std = feats

    # Basic noisy environment check
    if rms_mean > 0.18 and pitch_std > 90:
        return {
            "stress_score": None,
            "state": "Environment too noisy",
            "info": "Noisy/mixed speech detected. Please record again in a quieter place."
        }

    # -------------------------
    # If no baseline is provided, use a generic calm-ish baseline
    # -------------------------
    if baseline is None:
        baseline = {
            "rms_mean": 0.03,   # typical calm speech loudness
            "rms_std": 0.008,   # typical calm loudness variability
            "pitch_std": 20.0   # typical calm pitch variability
        }

    base_rms_mean = max(float(baseline.get("rms_mean", 0.03)), 1e-5)
    base_rms_std = max(float(baseline.get("rms_std", 0.008)), 1e-5)
    base_pitch_std = max(float(baseline.get("pitch_std", 20.0)), 1e-5)

    # -------------------------
    # Relative changes compared to baseline
    # -------------------------
    # (current - baseline) / baseline
    delta_loud_level = (rms_mean - base_rms_mean) / base_rms_mean
    delta_loud_var = (rms_std - base_rms_std) / base_rms_std
    delta_pitch_var = (pitch_std - base_pitch_std) / base_pitch_std

    # Clamp strong negative values (being quieter than baseline is not “negative stress”)
    delta_loud_level = max(delta_loud_level, -0.5)
    delta_loud_var = max(delta_loud_var, -0.5)
    delta_pitch_var = max(delta_pitch_var, -0.5)

    # -------------------------
    # Combine into a single stress index
    # -------------------------
    # Weights: pitch change (0.5), loudness variation change (0.3), loudness level change (0.2)
    stress_index = (
        0.5 * delta_pitch_var +
        0.3 * delta_loud_var +
        0.2 * delta_loud_level
    )

    # -------------------------
    # Map stress_index -> 0–100 using a sigmoid (logistic) function
    # -------------------------
    # Around baseline (stress_index ~ 0), score is moderate (~40).
    # As stress_index grows, score increases towards 100.
    k = 1.4   # slope
    x0 = 0.3  # center point of sigmoid
    sig = 1 / (1 + math.exp(-k * (stress_index - x0)))  # 0–1

    stress_score = float(sig * 100.0)
    stress_score = max(0.0, min(stress_score, 100.0))

    # -------------------------
    # Map numeric score to stress levels (tuned to current range ~33–51)
    # -------------------------
    if stress_score < 36:
        state = "Calm"
    elif stress_score < 42:
        state = "Mild Stress"
    elif stress_score < 48:
        state = "Moderate Stress"
    else:
        state = "High Stress"


    return {
        "stress_score": round(stress_score, 2),
        "state": state,
        "info": "Computed relative to baseline voice using loudness + pitch variability. Audio is not stored."
    }

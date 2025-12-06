# typing_module/typing_features.py

from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
from pathlib import Path

import joblib
import pandas as pd


@dataclass
class KeyEvent:
    """
    Represents a single keyboard event.

    timestamp : float (seconds)
    key       : str   (e.g., "a", "b", "BACKSPACE")
    event_type: str   ("down" or "up")
    """
    timestamp: float
    key: str
    event_type: str  # "down" or "up"


# ---------------------------
#  Load ML model (if present)
# ---------------------------

_TSI_MODEL = None
_THIS_DIR = Path(__file__).resolve().parent
_MODEL_PATH = _THIS_DIR / "tsi_model.pkl"

if _MODEL_PATH.exists():
    try:
        _TSI_MODEL = joblib.load(_MODEL_PATH)
        print("[TypingModule] Loaded ML TSI model from", _MODEL_PATH)
    except Exception as e:
        print("[TypingModule] Failed to load ML model:", e)
        _TSI_MODEL = None


# ---------------------------
#  Basic feature computations
# ---------------------------

def compute_typing_speed(events: List[KeyEvent]) -> float:
    """
    Compute typing speed in characters per minute (CPM),
    based on key DOWN events (excluding BACKSPACE).
    """
    key_down_events = [e for e in events if e.event_type == "down"]
    if len(key_down_events) < 2:
        return 0.0

    start_time = key_down_events[0].timestamp
    end_time = key_down_events[-1].timestamp
    duration_sec = max(end_time - start_time, 1e-6)  # avoid division by zero

    chars_typed = len([e for e in key_down_events if e.key != "BACKSPACE"])
    cpm = chars_typed / duration_sec * 60.0
    return cpm


def compute_backspace_rate(events: List[KeyEvent]) -> float:
    """
    Compute backspaces per 100 key DOWN events.
    """
    key_down_events = [e for e in events if e.event_type == "down"]
    if not key_down_events:
        return 0.0

    total_keys = len(key_down_events)
    backspaces = len([e for e in key_down_events if e.key == "BACKSPACE"])
    return (backspaces / total_keys) * 100.0


def compute_pause_stats(events: List[KeyEvent]) -> Tuple[float, float]:
    """
    Compute average and maximum pause (in seconds)
    between consecutive key DOWN events.
    """
    key_down_events = [e for e in events if e.event_type == "down"]
    if len(key_down_events) < 2:
        return 0.0, 0.0

    timestamps = [e.timestamp for e in key_down_events]
    gaps = [
        timestamps[i + 1] - timestamps[i]
        for i in range(len(timestamps) - 1)
    ]

    avg_pause = sum(gaps) / len(gaps)
    max_pause = max(gaps)
    return avg_pause, max_pause


def extract_typing_features(events: List[KeyEvent]) -> Dict[str, float]:
    """
    High-level wrapper that returns all relevant typing features
    in a single dictionary. This will later be passed to the
    Typing Stress Index (TSI) logic.
    """
    cpm = compute_typing_speed(events)
    backspace_rate = compute_backspace_rate(events)
    avg_pause, max_pause = compute_pause_stats(events)

    return {
        "typing_speed_cpm": cpm,
        "backspace_rate_per_100": backspace_rate,
        "avg_pause_sec": avg_pause,
        "max_pause_sec": max_pause,
    }


# ---------------------------
#  Pause pattern classification
# ---------------------------

def classify_pause_pattern(features: Dict[str, float]) -> str:
    """
    Heuristic classification of pause behavior, based on
    max pause, average pause and backspace rate.
    """
    max_pause = features.get("max_pause_sec", 0.0)
    avg_pause = features.get("avg_pause_sec", 0.0)
    backspace_rate = features.get("backspace_rate_per_100", 0.0)

    if max_pause >= 8.0:
        return "external_distraction"

    if max_pause >= 3.0 and backspace_rate >= 15.0:
        return "possible_micro_stress"

    if max_pause >= 3.0:
        return "normal_pause"

    if avg_pause <= 0.7:
        return "continuous_typing"

    return "regular_typing"


# ---------------------------
#  Heuristic TSI (rule-based)
# ---------------------------

def compute_tsi(features: Dict[str, float]) -> float:
    """
    Compute a simple Typing Stress Index (TSI) in [0, 100].
    Higher score = more stressed pattern (heuristic-based).
    """
    speed = features.get("typing_speed_cpm", 0.0)
    backspace_rate = features.get("backspace_rate_per_100", 0.0)
    avg_pause = features.get("avg_pause_sec", 0.0)
    max_pause = features.get("max_pause_sec", 0.0)

    score = 40.0  # baseline

    # 1) Typing speed
    if speed < 80:
        score += 20
    elif speed < 120:
        score += 10
    else:
        score -= 5

    # 2) Backspace rate
    if backspace_rate > 25:
        score += 25
    elif backspace_rate > 10:
        score += 10

    # 3) Pauses
    if avg_pause > 3.0:
        score += 20
    elif avg_pause > 1.5:
        score += 10

    if max_pause > 8.0:
        score += 5
    elif max_pause > 4.0:
        score += 10

    score = max(0.0, min(100.0, score))
    return score


def classify_tsi_level(score: float) -> str:
    """
    Map TSI numeric score to a qualitative label that is
    user-friendly (non-panicky language).
    """
    if score < 30:
        return "Relaxed"
    if score < 55:
        return "Stable"
    if score < 75:
        return "Needs a short break"
    if score < 90:
        return "Overloaded moment"
    return "Highly overwhelmed"


# ---------------------------
#  ML-based TSI
# ---------------------------

def compute_tsi_ml(features: Dict[str, float]) -> float:
    """
    Compute TSI using trained ML model if available.
    """
    global _TSI_MODEL
    if _TSI_MODEL is None:
        return -1.0

    # Use a DataFrame with the same column names as training
    X = pd.DataFrame([{
        "typing_speed_cpm": features.get("typing_speed_cpm", 0.0),
        "backspace_rate_per_100": features.get("backspace_rate_per_100", 0.0),
        "avg_pause_sec": features.get("avg_pause_sec", 0.0),
        "max_pause_sec": features.get("max_pause_sec", 0.0),
    }])

    tsi_pred = float(_TSI_MODEL.predict(X)[0])
    tsi_pred = max(0.0, min(100.0, tsi_pred))
    return tsi_pred


# ---------------------------
#  Conversion + High-level API
# ---------------------------

def dicts_to_keyevents(raw_events: List[Dict[str, Any]]) -> List[KeyEvent]:
    """
    Convert a list of dictionaries (e.g. from JSON) into KeyEvent objects.
    """
    events: List[KeyEvent] = []

    for item in raw_events:
        if not all(k in item for k in ("timestamp", "key", "event_type")):
            continue

        try:
            ev = KeyEvent(
                timestamp=float(item["timestamp"]),
                key=str(item["key"]),
                event_type=str(item["event_type"]),
            )
            events.append(ev)
        except (ValueError, TypeError):
            continue

    return events


def analyze_typing_session(events: List[KeyEvent]) -> Dict[str, Any]:
    """
    High-level wrapper called by backend/API.
    Produces a clean, JSON-serializable dictionary.
    """
    features = extract_typing_features(events)
    pause_label = classify_pause_pattern(features)

    tsi_ml = compute_tsi_ml(features)
    tsi_heuristic = compute_tsi(features)

    if tsi_ml >= 0.0:
        tsi_score = 0.7 * tsi_ml + 0.3 * tsi_heuristic
    else:
        tsi_score = tsi_heuristic

    tsi_label = classify_tsi_level(tsi_score)

    return {
        "features": features,
        "pause_pattern": pause_label,
        "tsi_score": tsi_score,
        "tsi_label": tsi_label,
    }


def predict_tsi(raw_events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Public entry point for Typing Stress Index prediction.
    Backend should call this function.
    """
    events = dicts_to_keyevents(raw_events)
    return analyze_typing_session(events)

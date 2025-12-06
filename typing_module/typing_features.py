# typing_module/typing_features.py

from dataclasses import dataclass
from typing import List, Dict, Tuple


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

    This is deliberately simple for hackathon use and can
    be refined later.
    """
    max_pause = features.get("max_pause_sec", 0.0)
    avg_pause = features.get("avg_pause_sec", 0.0)
    backspace_rate = features.get("backspace_rate_per_100", 0.0)

    # Very long pause -> likely context switch (talking to someone / phone / distraction)
    if max_pause >= 8.0:
        return "external_distraction"

    # Medium pauses with high backspace can signal micro-stress
    if max_pause >= 3.0 and backspace_rate >= 15.0:
        return "possible_micro_stress"

    # Medium pauses but low backspace -> normal thinking pauses
    if max_pause >= 3.0:
        return "normal_pause"

    # Very low pauses -> continuous typing / flow state
    if avg_pause <= 0.7:
        return "continuous_typing"

    return "regular_typing"


# ---------------------------
#  Typing Stress Index (TSI)
# ---------------------------

def compute_tsi(features: Dict[str, float]) -> float:
    """
    Compute a simple Typing Stress Index (TSI) in [0, 100].

    Higher score = more stressed pattern (heuristic-based).
    This is NOT a clinical score; just a behavioral indicator.
    """
    speed = features.get("typing_speed_cpm", 0.0)
    backspace_rate = features.get("backspace_rate_per_100", 0.0)
    avg_pause = features.get("avg_pause_sec", 0.0)
    max_pause = features.get("max_pause_sec", 0.0)

    score = 40.0  # baseline

    # 1) Typing speed: unusually low speed increases score
    if speed < 80:
        score += 20
    elif speed < 120:
        score += 10
    else:
        score -= 5  # good speed, slightly lower stress

    # 2) Backspace rate: more corrections => more friction
    if backspace_rate > 25:
        score += 25
    elif backspace_rate > 10:
        score += 10

    # 3) Pauses: long pauses + high variability suggest strain
    if avg_pause > 3.0:
        score += 20
    elif avg_pause > 1.5:
        score += 10

    if max_pause > 8.0:
        score += 5  # may be distraction, small penalty
    elif max_pause > 4.0:
        score += 10

    # Clamp to [0, 100]
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

def analyze_typing_session(events: List[KeyEvent]) -> Dict[str, object]:
    """
    High-level wrapper called by backend/API.
    Produces a clean, JSON-serializable dictionary.
    """
    features = extract_typing_features(events)
    pause_label = classify_pause_pattern(features)
    tsi_score = compute_tsi(features)
    tsi_label = classify_tsi_level(tsi_score)

    return {
        "features": features,
        "pause_pattern": pause_label,
        "tsi_score": tsi_score,
        "tsi_label": tsi_label
    }

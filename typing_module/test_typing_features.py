# typing_module/test_typing_features.py

from typing_features import (
    KeyEvent,
    extract_typing_features,
    classify_pause_pattern,
    compute_tsi,
    classify_tsi_level,
)


def build_dummy_events():
    """
    Simulate a small typing session with a few pauses and a backspace.
    Timestamps are in seconds.
    """
    events = [
        # typing "hel"
        KeyEvent(0.00, "h", "down"),
        KeyEvent(0.05, "h", "up"),
        KeyEvent(0.10, "e", "down"),
        KeyEvent(0.15, "e", "up"),
        KeyEvent(0.20, "l", "down"),
        KeyEvent(0.25, "l", "up"),

        # pause + backspace
        KeyEvent(0.90, "BACKSPACE", "down"),
        KeyEvent(0.95, "BACKSPACE", "up"),

        # continue typing "lo"
        KeyEvent(1.10, "l", "down"),
        KeyEvent(1.15, "l", "up"),
        KeyEvent(1.30, "o", "down"),
        KeyEvent(1.35, "o", "up"),
    ]
    return events


if __name__ == "__main__":
    events = build_dummy_events()

    features = extract_typing_features(events)
    print("=== Typing Features ===")
    for k, v in features.items():
        print(f"{k}: {v}")

    pause_pattern = classify_pause_pattern(features)
    print("\n=== Pause Pattern Classification ===")
    print(pause_pattern)

    tsi_score = compute_tsi(features)
    tsi_label = classify_tsi_level(tsi_score)

    print("\n=== Typing Stress Index (TSI) ===")
    print(f"TSI Score: {tsi_score}")
    print(f"TSI Level: {tsi_label}")

from typing_features import (
    KeyEvent,
    analyze_typing_session
)

if __name__ == "__main__":
    events = build_dummy_events()

    results = analyze_typing_session(events)

    print("\n=== Final Analysis Output ===")
    print(results)

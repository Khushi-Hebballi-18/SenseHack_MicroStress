# typing_module/test_typing_features.py

from typing_features import (
    KeyEvent,
    analyze_typing_session,
    dicts_to_keyevents,
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


def run_direct_test():
    events = build_dummy_events()
    results = analyze_typing_session(events)

    print("\n=== Direct KeyEvent Test ===")
    for k, v in results.items():
        print(f"{k}: {v}")


def run_json_like_test():
    raw = [
        {"timestamp": 0.0, "key": "h", "event_type": "down"},
        {"timestamp": 0.05, "key": "h", "event_type": "up"},
        {"timestamp": 0.10, "key": "e", "event_type": "down"},
        {"timestamp": 0.15, "key": "e", "event_type": "up"},
    ]
    events = dicts_to_keyevents(raw)
    results = analyze_typing_session(events)

    print("\n=== JSON-like Events Test ===")
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    run_direct_test()
    run_json_like_test()

# global_typing_listener.py

import time
from threading import Thread, Event
from typing import List

from pynput import keyboard
from typing_module.typing_features import KeyEvent, analyze_typing_session


events: List[KeyEvent] = []
stop_flag = Event()


def on_press(key):
    timestamp = time.time()
    key_name = "BACKSPACE" if key == keyboard.Key.backspace else "KEY"
    events.append(KeyEvent(timestamp=timestamp, key=key_name, event_type="down"))


def on_release(key):
    timestamp = time.time()
    key_name = "BACKSPACE" if key == keyboard.Key.backspace else "KEY"
    events.append(KeyEvent(timestamp=timestamp, key=key_name, event_type="up"))

    if key == keyboard.Key.esc:
        print("ESC pressed, stopping listener...")
        stop_flag.set()
        return False


def analysis_loop(window_seconds: int = 20, min_events: int = 8):
    """
    Every `window_seconds`, analyze the last window of typing.
    """
    print(f"[Analyzer] Running every {window_seconds} seconds. Press ESC to stop.\n")
    while not stop_flag.is_set():
        time.sleep(window_seconds)

        now = time.time()
        window_start = now - window_seconds
        window_events = [e for e in events if e.timestamp >= window_start]

        if len(window_events) < min_events:
            print(
                f"[Analyzer] Not enough typing in last {window_seconds}s "
                f"({len(window_events)} events). Skipping.\n"
            )
            continue

        result = analyze_typing_session(window_events)

        print("\n========== Typing Analysis (Last "
              f"{window_seconds}s) ==========")
        print(f"TSI Score   : {result['tsi_score']:.1f}")
        print(f"TSI Label   : {result['tsi_label']}")
        print(f"Pause type  : {result['pause_pattern']}")
        print(f"Features    : {result['features']}")
        print("============================================\n")


def main():
    analyzer_thread = Thread(
        target=analysis_loop,
        kwargs={"window_seconds": 20, "min_events": 8},  # 20 seconds
        daemon=True,
    )
    analyzer_thread.start()

    print("Global typing listener started.")
    print("Type anywhere (Notes, WhatsApp Web, etc.).")
    print("Every 20s your typing in that period will be analyzed.")
    print("Press ESC to stop.\n")

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

    print("Listener stopped. Exiting...")


if __name__ == "__main__":
    main()

# typing_module/demo_predict_tsi.py

from typing_module.typing_features import predict_tsi

def main():
    # Fake events like what frontend/backend will send
    raw_events = [
        {"timestamp": 0.00, "key": "h", "event_type": "down"},
        {"timestamp": 0.05, "key": "h", "event_type": "up"},
        {"timestamp": 0.10, "key": "e", "event_type": "down"},
        {"timestamp": 0.15, "key": "e", "event_type": "up"},
        {"timestamp": 0.20, "key": "l", "event_type": "down"},
        {"timestamp": 0.25, "key": "l", "event_type": "up"},
        {"timestamp": 0.30, "key": "l", "event_type": "down"},
        {"timestamp": 0.35, "key": "l", "event_type": "up"},
        {"timestamp": 0.40, "key": "o", "event_type": "down"},
        {"timestamp": 0.45, "key": "o", "event_type": "up"},
    ]

    result = predict_tsi(raw_events)

    print("=== Demo predict_tsi output ===")
    for k, v in result.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()

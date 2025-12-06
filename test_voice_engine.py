# test_voice_engine.py

from voice_module.voice_engine import (
    compute_baseline_from_bytes,
    analyze_voice_stress_from_bytes,
)


def load_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def test_with_baseline(baseline_path: str, sample_paths: list[str]):
    # 1. Load and compute baseline
    baseline_bytes = load_bytes(baseline_path)
    baseline_result = compute_baseline_from_bytes(baseline_bytes)

    if not baseline_result.get("ok"):
        print("Baseline error:", baseline_result.get("error"))
        return

    baseline = baseline_result["baseline"]
    print("Baseline computed from:", baseline_path)
    print("  rms_mean =", baseline["rms_mean"])
    print("  rms_std  =", baseline["rms_std"])
    print("  pitch_std=", baseline["pitch_std"])
    print()

    # 2. Analyze each sample vs this baseline
    for path in sample_paths:
        raw = load_bytes(path)
        result = analyze_voice_stress_from_bytes(raw, baseline=baseline)
        print(path, "=>", result)


if __name__ == "__main__":
    # Choose one calm / neutral file as baseline
    # You can change "baseline.wav" to any calm sample file you have.
    baseline_path = "Sample1.wav"  # for example: your calmest recording

    sample_paths = [
        "sample.wav",
        "sample1.1.wav",
        "sample1.2.wav",
        "sample1.3.wav",
        "sample1.4.wav",
        "Sample1.wav",
        "sample1.5.wav",
        "sample1.6.wav",
        "sample1.7.wav",
        "sample1.9.wav",
        "short2.0.wav",
        "sample2.1.wav",
        "sample2.2.wav",
        "sample2.3.wav",
        "shouting1.wav",
    ]

    test_with_baseline(baseline_path, sample_paths)

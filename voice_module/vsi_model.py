from .voice_features import extract_voice_features
import numpy as np

class VoiceStressModel:
    """
    Simple heuristic model to convert extracted voice features into a stress score.
    Designed for hackathon: fast, lightweight, no training required.
    """

    def compute_vsi(self, features: np.ndarray) -> float:
        # Unpack features
        (
            rms_mean, rms_std,
            pitch_mean, pitch_std,
            spec_centroid_mean, spec_centroid_std,
            zcr_mean, zcr_std
        ) = features

        # Normalize values (rough)
        energy_component = min(rms_std / 0.05, 1.0)
        pitch_component = min(pitch_std / 20.0, 1.0)
        brightness_component = min(spec_centroid_mean / 4000.0, 1.0)
        noise_component = min(zcr_mean / 0.2, 1.0)

        # Weighted sum
        stress_raw = (
            0.35 * energy_component +
            0.35 * pitch_component +
            0.15 * brightness_component +
            0.15 * noise_component
        )

        vsi_score = float(stress_raw * 100.0)
        return max(0.0, min(vsi_score, 100.0))

    def state_label(self, vsi: float) -> str:
        if vsi < 25:
            return "Relaxed"
        elif vsi < 50:
            return "Focused"
        elif vsi < 70:
            return "Needs a Short Break"
        else:
            return "Overwhelmed"

    def predict_from_file(self, file_path: str) -> dict:
        features = extract_voice_features(file_path)
        vsi = self.compute_vsi(features)
        label = self.state_label(vsi)

        return {
            "vsi_score": round(vsi, 2),
            "state": label,
            "features": features.tolist()  # debug
        }

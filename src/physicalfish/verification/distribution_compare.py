"""Distribution comparison for synthetic vs real trajectories."""

import numpy as np

from physicalfish.models import FrameData, TrajectoryData


class DistributionComparer:
    """Compare synthetic vs real trajectory distributions."""

    def compare(self, synthetic: TrajectoryData, real: TrajectoryData) -> dict:
        """Compare feature distributions between synthetic and real data."""
        syn_features = self._extract_features(synthetic.frames)
        real_features = self._extract_features(real.frames)

        similarities = {}
        for key in syn_features:
            syn_mean = np.mean(syn_features[key])
            real_mean = np.mean(real_features[key])
            syn_std = np.std(syn_features[key])
            real_std = np.std(real_features[key])

            mean_diff = abs(syn_mean - real_mean) / (abs(real_mean) + 1e-6)
            std_diff = abs(syn_std - real_std) / (abs(real_std) + 1e-6)
            similarity = max(0.0, 1.0 - (mean_diff + std_diff) / 2)
            similarities[key] = float(similarity)

        return {
            "feature_similarities": similarities,
            "overall_similarity": float(np.mean(list(similarities.values()))),
        }

    def _extract_features(self, frames: list[FrameData]) -> dict[str, list[float]]:
        return {
            "velocity_magnitude": [f.speed for f in frames],
            "acceleration_magnitude": [float(np.linalg.norm(f.acceleration)) for f in frames],
            "height": [f.height for f in frames],
        }

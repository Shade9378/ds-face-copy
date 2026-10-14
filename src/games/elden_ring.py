"""First planned game calibration target."""

from games.base_adapter import GameAdapter, UnsupportedGenerationError


class EldenRingAdapter(GameAdapter):
    """Unavailable until calibration and slider schema have been verified."""

    def generate_sliders(self, features: dict[str, float]) -> dict[str, dict[str, int]]:
        """Refuse fabricated slider values."""
        raise UnsupportedGenerationError("Elden Ring requires verified calibration data.")

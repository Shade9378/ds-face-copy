from games.base_adapter import GameAdapter, UnsupportedGenerationError


class EldenRingAdapter(GameAdapter):
    def generate_sliders(self, features: dict[str, float]) -> dict[str, dict[str, int]]:
        raise UnsupportedGenerationError("Elden Ring requires verified calibration data.")

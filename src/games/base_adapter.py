"""Common future game calibration contract."""

from abc import ABC, abstractmethod


class UnsupportedGenerationError(RuntimeError):
    """Generation lacks verified calibration data."""


class GameAdapter(ABC):
    """Each game must implement its own verified mapping."""

    @abstractmethod
    def generate_sliders(self, features: dict[str, float]) -> dict[str, dict[str, int]]:
        """Generate category-grouped values or raise an unsupported error."""
        raise NotImplementedError

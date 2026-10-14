from abc import ABC, abstractmethod


class UnsupportedGenerationError(RuntimeError):
    pass


class GameAdapter(ABC):
    @abstractmethod
    def generate_sliders(self, features: dict[str, float]) -> dict[str, dict[str, int]]:
        raise NotImplementedError

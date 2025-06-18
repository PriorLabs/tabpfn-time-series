import numpy as np
from abc import abstractmethod


class CovariateGenerator:
    """Minimal base class for covariate generators.

    Each generator should override the generate method to implement
    the specific covariate generation logic.
    """

    @abstractmethod
    def generate(self, n_timesteps: int, seed: int = None, **kwargs) -> np.ndarray:
        """Generate a covariate signal of specified length.

        Args:
            n_timesteps: Number of timesteps to generate
            seed: Random seed for reproducible generation
            **kwargs: Generator-specific parameters

        Returns:
            Generated covariate signal as numpy array
        """
        raise NotImplementedError("Subclasses must implement generate method")

    @classmethod
    def generate_random_parameters(cls, seed: int = None) -> dict:
        """Generate random parameters for the generator."""
        rng = np.random.RandomState(seed)
        return cls._generate_random_parameters(rng)

    @staticmethod
    def _generate_random_parameters(rng: np.random.RandomState) -> dict:
        raise NotImplementedError(
            "Subclasses must implement _generate_random_parameters method"
        )

    def __call__(self, n_timesteps: int, seed: int = None, **kwargs) -> np.ndarray:
        """Make the generator callable."""
        return self.generate(n_timesteps, seed, **kwargs)

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return self.__str__()

    def _init_rng(self, seed: int = None) -> np.random.RandomState:
        """Initialize random number generator."""
        return np.random.RandomState(seed) if seed is not None else np.random

import numpy as np
import logging
from .base import CovariateGenerator
from .covariate_generators import ALL_COVARIATE_GENERATORS

logger = logging.getLogger(__name__)


class LinearTrend(CovariateGenerator):
    """Generates a linear trend with random slope."""

    def generate(
        self,
        n_timesteps: int,
        slope: float,
    ) -> np.ndarray:
        """Generate linear trend covariate.

        Args:
            n_timesteps: Number of timesteps to generate
            slope: Slope of the linear trend

        Returns:
            Generated linear trend signal as numpy array
        """
        return slope * np.arange(n_timesteps)

    @staticmethod
    def _generate_random_parameters(rng: np.random.RandomState) -> dict:
        slope = np.exp(rng.uniform(np.log(0.01), np.log(0.1)))
        return {
            "slope": slope if rng.random() < 0.5 else -slope,
        }


class AR1(CovariateGenerator):
    """Generates a stationary or near-unit-root AR(1) process."""

    def generate(
        self,
        n_timesteps: int,
        seed: int = None,
        phi: float = 0.95,
        sigma: float = 0.02,
        init: float = 0.0,
        scale: float = 10.0,
    ) -> np.ndarray:
        """Generate AR(1) process covariate.

        Args:
            n_timesteps: Number of timesteps to generate
            seed: Random seed for reproducible generation
            phi: Autoregressive parameter
            sigma: Noise standard deviation
            init: Initial value

        Returns:
            Generated AR(1) signal as numpy array
        """
        # Set random seed if provided
        rng = self._init_rng(seed)

        x = np.zeros(n_timesteps)
        x[0] = init
        logger.debug(f"AR1: phi={phi}, sigma={sigma}, init={init}")
        for t in range(1, n_timesteps):
            x[t] = phi * x[t - 1] + rng.normal(0, sigma)
        return scale * x

    @staticmethod
    def _generate_random_parameters(rng: np.random.RandomState) -> dict:
        return {
            "phi": rng.uniform(0.9, 0.99),
            "sigma": np.exp(rng.uniform(np.log(0.01), np.log(0.05))),
            "init": rng.uniform(-1, 1),
        }


class LogisticGrowthGenerator(CovariateGenerator):
    """Generates a logistic (sigmoid) growth curve with carrying capacity K."""

    def generate(
        self,
        n_timesteps: int,
        max_value: float = 1.0,
        growth_rate: float = 0.1,
        inflection_point_ratio: float = 0.5,
    ) -> np.ndarray:
        """Generate logistic growth curve covariate.

        Args:
            n_timesteps: Number of timesteps to generate
            seed: Random seed for reproducible generation (not used in this deterministic function)
            max_value: Maximum value (K parameter)
            growth_rate: Growth rate (r parameter)
            inflection_point_ratio: Ratio of inflection point to n_timesteps

        Returns:
            Generated logistic growth signal as numpy array
        """
        inflection_point = int(inflection_point_ratio * n_timesteps)

        t = np.arange(n_timesteps)
        return max_value / (1 + np.exp(-growth_rate * (t - inflection_point)))

    @staticmethod
    def _generate_random_parameters(rng: np.random.RandomState) -> dict:
        return {
            "max_value": np.exp(rng.uniform(np.log(0.5), np.log(1.5))),
            "growth_rate": np.exp(rng.uniform(np.log(0.01), np.log(0.05))),
            "inflection_point_ratio": rng.randint(50, 80) / 100.0,
        }


# Register all generators
ALL_COVARIATE_GENERATORS.update(
    {
        "linear_trend": LinearTrend(),
        "ar1": AR1(),
        "logistic_growth": LogisticGrowthGenerator(),
    }
)

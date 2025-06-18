import numpy as np
import logging

from .base import CovariateGenerator
from .covariate_generators import ALL_COVARIATE_GENERATORS


# Set up logging
logger = logging.getLogger(__name__)


class RandomWalk(CovariateGenerator):
    """Generates random walk covariates."""

    def generate(
        self,
        n_timesteps: int,
        seed: int = None,
        step_size: float = 0.5,
        interval: int = 10,
        interpolation_method: str = "linear",
    ) -> np.ndarray:
        """Generate random walk covariate signal.

        Args:
            n_timesteps: Number of timesteps to generate
            seed: Random seed for reproducible generation
            step_size_range: Tuple of (min_step_size, max_step_size) for random step size
            interval_range: Tuple of (min_interval, max_interval) for step intervals
            interpolation_method: Method for interpolation ("linear" or "constant")

        Returns:
            Generated random walk covariate signal as numpy array
        """
        # Set random seed if provided
        rng = self._init_rng(seed)

        # Generate random walk with steps every interval timesteps
        n_steps = (n_timesteps + interval - 1) // interval
        steps = rng.normal(0, step_size, n_steps)
        walk_values = np.cumsum(steps)

        # Interpolate to fill all timesteps
        if interpolation_method == "constant":
            result = np.zeros(n_timesteps)
            for i in range(n_timesteps):
                step_index = min(i // interval, len(walk_values) - 1)
                result[i] = walk_values[step_index]
        elif interpolation_method == "linear":
            interval_points = np.arange(0, n_timesteps, interval)
            if len(interval_points) > len(walk_values):
                walk_values = np.pad(
                    walk_values,
                    (0, len(interval_points) - len(walk_values)),
                    mode="edge",
                )
            elif len(interval_points) < len(walk_values):
                walk_values = walk_values[: len(interval_points)]

            if interval_points[-1] < n_timesteps - 1:
                interval_points = np.append(interval_points, n_timesteps - 1)
                walk_values = np.append(walk_values, walk_values[-1])

            result = np.interp(np.arange(n_timesteps), interval_points, walk_values)
        else:
            raise ValueError(f"Unknown interpolation method: {interpolation_method}")

        return result

    @staticmethod
    def _generate_random_parameters(rng: np.random.RandomState) -> dict:
        return {
            "step_size": np.exp(rng.uniform(np.log(0.5), np.log(3.0))),
            "interval": rng.randint(5, 15),
            "interpolation_method": rng.choice(["linear", "constant"]),
        }


# Register the generator
ALL_COVARIATE_GENERATORS["random_walk"] = RandomWalk()

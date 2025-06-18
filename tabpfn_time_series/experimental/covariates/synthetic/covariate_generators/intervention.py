import numpy as np
import logging

from .base import CovariateGenerator
from .covariate_generators import ALL_COVARIATE_GENERATORS

logger = logging.getLogger(__name__)


class Pulses(CovariateGenerator):
    """Generates pulse events at specified timesteps."""

    def generate(
        self,
        n_timesteps: int,
        event_time: list[int] | list[float],
        amplitude: float = 5.0,
    ) -> np.ndarray:
        signal = np.zeros(n_timesteps)

        # Convert to numpy array once
        event_time_array = np.array(event_time)

        # Handle float conversion if needed
        if event_time_array.dtype.kind == "f":  # More efficient float check
            assert np.all(
                (event_time_array >= 0) & (event_time_array <= 1)
            ), "Event times must be between 0 and 1"
            event_time_array = (event_time_array * n_timesteps).astype(int)

        # Filter valid times and set amplitudes in one step
        valid_times = (event_time_array >= 0) & (event_time_array < n_timesteps)
        valid_event_indices = event_time_array[valid_times]
        signal[valid_event_indices] = amplitude
        return signal

    @staticmethod
    def _generate_random_parameters(rng: np.random.RandomState) -> dict:
        # Generate uniformly distributed pulse times
        n_pulses = rng.randint(8, 15)

        # Create uniform base spacing and add small random jitter
        base_spacing = 0.9 / (n_pulses + 1)  # Leave 10% buffer at start/end
        event_times = []

        for i in range(n_pulses):
            # Base position with small random jitter (±25% of base spacing)
            base_position = (i + 1) * base_spacing + 0.05  # Start at 5%
            jitter = rng.uniform(-0.25 * base_spacing, 0.25 * base_spacing)
            time = max(0.01, min(0.99, base_position + jitter))  # Keep within bounds
            event_times.append(time)

        return {"event_time": sorted(event_times)}


class Steps(CovariateGenerator):
    """Generates multiple step events that turn on and off at specified times."""

    def generate(
        self,
        n_timesteps: int,
        start_times: list[int] | list[float],
        end_times: list[int] | list[float],
        amplitudes: list[float] | float = 5.0,
    ) -> np.ndarray:
        signal = np.zeros(n_timesteps)

        # Convert to numpy arrays
        start_times_array = np.array(start_times)
        end_times_array = np.array(end_times)

        # Handle float conversion if needed (assume normalized to [0, 1])
        if start_times_array.dtype.kind == "f":
            assert np.all(
                (start_times_array >= 0) & (start_times_array <= 1)
            ), "Start times must be between 0 and 1"
            start_times_array = (start_times_array * n_timesteps).astype(int)
        if end_times_array.dtype.kind == "f":
            assert np.all(
                (end_times_array >= 0) & (end_times_array <= 1)
            ), "End times must be between 0 and 1"
            end_times_array = (end_times_array * n_timesteps).astype(int)

        # Handle amplitudes (can be single value or list)
        if isinstance(amplitudes, (int, float)):
            amplitudes = [amplitudes] * len(start_times_array)

        amplitudes_array = np.array(amplitudes)

        # Ensure all arrays have the same length
        assert (
            len(start_times_array) == len(end_times_array) == len(amplitudes_array)
        ), "All parameter arrays must have the same length"

        # Generate each step
        for start_time, end_time, amplitude in zip(
            start_times_array, end_times_array, amplitudes_array
        ):
            start = max(0, start_time)
            end = min(n_timesteps, end_time)
            if start < end:
                # Take maximum value at each timestep (no overlapping addition)
                signal[start:end] = np.maximum(signal[start:end], amplitude)

        return signal

    @staticmethod
    def _generate_random_parameters(rng: np.random.RandomState) -> dict:
        # Generate non-overlapping step intervals
        n_steps = rng.randint(3, 7)
        intervals = []

        # Generate non-overlapping intervals
        current_time = 0.0
        for _ in range(n_steps):
            if current_time >= 0.9:  # Leave some space at the end
                break

            # Random gap before this step
            gap = rng.uniform(0.02, 0.1)
            start = current_time + gap

            # Random duration for this step
            max_duration = min(0.15, 0.9 - start)
            if max_duration <= 0.02:
                break
            duration = rng.uniform(0.02, max_duration)
            end = start + duration

            intervals.append((start, end))
            current_time = end

        if len(intervals) == 0:
            # Fallback: single step
            intervals = [(0.1, 0.3)]

        start_times, end_times = zip(*intervals)
        return {
            "start_times": list(start_times),
            "end_times": list(end_times),
        }


class Ramps(CovariateGenerator):
    """Generates multiple ramp-up or ramp-down events over time intervals."""

    def generate(
        self,
        n_timesteps: int,
        start_times: list[int] | list[float],
        end_times: list[int] | list[float],
        start_values: list[float] | float = 0.0,
        end_values: list[float] | float = 5.0,
    ) -> np.ndarray:
        signal = np.zeros(n_timesteps)

        # Convert to numpy arrays
        start_times_array = np.array(start_times)
        end_times_array = np.array(end_times)

        # Handle float conversion if needed (assume normalized to [0, 1])
        if start_times_array.dtype.kind == "f":
            assert np.all(
                (start_times_array >= 0) & (start_times_array <= 1)
            ), "Start times must be between 0 and 1"
            start_times_array = (start_times_array * n_timesteps).astype(int)
        if end_times_array.dtype.kind == "f":
            assert np.all(
                (end_times_array >= 0) & (end_times_array <= 1)
            ), "End times must be between 0 and 1"
            end_times_array = (end_times_array * n_timesteps).astype(int)

        # Handle start_values and end_values (can be single values or lists)
        if isinstance(start_values, (int, float)):
            start_values = [start_values] * len(start_times_array)
        if isinstance(end_values, (int, float)):
            end_values = [end_values] * len(end_times_array)

        start_values_array = np.array(start_values)
        end_values_array = np.array(end_values)

        # Ensure all arrays have the same length
        assert (
            len(start_times_array)
            == len(end_times_array)
            == len(start_values_array)
            == len(end_values_array)
        ), "All parameter arrays must have the same length"

        # Generate each ramp
        for start_time, end_time, start_value, end_value in zip(
            start_times_array, end_times_array, start_values_array, end_values_array
        ):
            if start_time < end_time:
                start = max(0, start_time)
                end = min(n_timesteps, end_time)
                length = end - start
                if length > 0:
                    # Take maximum value at each timestep (no overlapping addition)
                    ramp_values = np.linspace(start_value, end_value, length)
                    signal[start:end] = np.maximum(signal[start:end], ramp_values)

        return signal

    @staticmethod
    def _generate_random_parameters(rng: np.random.RandomState) -> dict:
        # Generate non-overlapping ramp intervals
        n_ramps = rng.randint(2, 5)
        intervals = []

        # Generate non-overlapping intervals
        current_time = 0.0
        for _ in range(n_ramps):
            if current_time >= 0.9:  # Leave some space at the end
                break

            # Random gap before this ramp
            gap = rng.uniform(0.05, 0.15)
            start = current_time + gap

            # Random duration for this ramp
            max_duration = min(0.2, 0.9 - start)
            if max_duration <= 0.05:
                break
            duration = rng.uniform(0.05, max_duration)
            end = start + duration

            intervals.append((start, end))
            current_time = end

        if len(intervals) == 0:
            # Fallback: single ramp
            intervals = [(0.1, 0.4)]

        start_times, end_times = zip(*intervals)
        return {
            "start_times": list(start_times),
            "end_times": list(end_times),
        }


ALL_COVARIATE_GENERATORS.update(
    {
        "pulses": Pulses(),
        "steps": Steps(),
        "ramps": Ramps(),
    }
)

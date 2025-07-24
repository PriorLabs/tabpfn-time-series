from typing import Tuple
import numpy as np


def generate_sinusoid_dataset(
    num_samples: int,
    num_points: int,
    min_periods: float = 2.0,
    max_periods: float = 4.0,
    amplitude: float = 1.0,
    noise_std: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a dataset of sinusoids with random frequencies and phase shifts
    using vectorized operations.
    """
    # 1. Generate random parameters for each sinusoid
    # Shape: (num_samples,)
    num_periods = np.random.uniform(min_periods, max_periods, size=num_samples)
    phase_shifts = np.random.uniform(0, 2 * np.pi, size=num_samples)
    time_starts = np.random.randint(0, 500, size=num_samples)

    # 2. Create time vectors
    # Base time for y-generation and for shifting, shape: (num_points,)
    base_time = np.arange(num_points)

    # Create shifted time for X features, shape: (num_samples, num_points)
    # Using broadcasting: time_starts[:, None] is (num_samples, 1), base_time is (num_points,)
    all_X_flat = time_starts[:, np.newaxis] + base_time

    # 3. Generate y values (sinusoids)
    # Reshape parameters for broadcasting with base_time.
    # num_periods[:, np.newaxis] -> (num_samples, 1)
    # base_time -> (num_points,)
    # Broadcasting happens over the second dimension of the first term and first of second term.
    # Resulting shape: (num_samples, num_points)
    sinusoids = amplitude * np.sin(
        2 * np.pi * num_periods[:, np.newaxis] * base_time / num_points
        + phase_shifts[:, np.newaxis]
    )

    # 4. Add noise
    # Shape: (num_samples, num_points)
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, size=(num_samples, num_points))
        all_y = (sinusoids + noise).astype(np.float32)
    else:
        all_y = sinusoids.astype(np.float32)

    # 5. Reshape X to final shape (num_samples, num_points, 1)
    all_X = all_X_flat[:, :, np.newaxis]

    return all_X, all_y


def generate_line_dataset(
    num_samples: int,
    num_points: int,
    min_slope: float = -2.0,
    max_slope: float = 2.0,
    min_intercept: float = -5.0,
    max_intercept: float = 5.0,
    noise_std: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a dataset of lines with random slopes and intercepts
    using vectorized operations.
    """
    # 1. Generate random parameters for each line
    # Shape: (num_samples,)
    slopes = np.random.uniform(min_slope, max_slope, size=num_samples)
    intercepts = np.random.uniform(min_intercept, max_intercept, size=num_samples)
    time_starts = np.random.randint(0, 500, size=num_samples)

    # 2. Create time vectors
    # Base time for y-generation, shape: (num_points,)
    base_time = np.arange(num_points)

    # Create shifted time for X features, shape: (num_samples, num_points)
    all_X_flat = time_starts[:, np.newaxis] + base_time

    # 3. Generate y values (lines)
    # y = m*x + c where x is base_time
    # slopes[:, np.newaxis] -> (num_samples, 1)
    # base_time -> (num_points,)
    # intercepts[:, np.newaxis] -> (num_samples, 1)
    # Resulting shape: (num_samples, num_points)
    lines = slopes[:, np.newaxis] * base_time + intercepts[:, np.newaxis]

    # 4. Add noise
    # Shape: (num_samples, num_points)
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, size=(num_samples, num_points))
        all_y = (lines + noise).astype(np.float32)
    else:
        all_y = lines.astype(np.float32)

    # 5. Reshape X to final shape (num_samples, num_points, 1)
    all_X = all_X_flat[:, :, np.newaxis]

    return all_X, all_y


def generate_sinusoid(
    num_points: int = 200,
    amplitude: float = 1.0,
    num_periods: float = 2.0,
    noise_std: float = 0.0,
    phase_shift: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a time series with a sinusoidal pattern and Gaussian noise.

    Args:
        num_points: The total number of points in the generated time series.
        amplitude: The amplitude of the sinusoid.
        num_periods: The number of full cycles over the series length.
        noise_std: The standard deviation of the Gaussian noise.
        phase_shift: The phase shift in radians.

    Returns:
        A tuple of (time_indices, series_values).
    """
    time = np.arange(num_points)
    sinusoid = amplitude * np.sin(
        2 * np.pi * num_periods * time / num_points + phase_shift
    )
    noise = np.random.normal(0, noise_std, size=num_points)
    series = (sinusoid + noise).astype(np.float32)
    return time.reshape(-1, 1), series


def generate_ood_sinusoid(
    num_points: int = 200,
    amplitude: float = 1.0,
    num_periods: float = 2.0,
    noise_std: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a time series with a square wave pattern and Gaussian noise.
    The time series is multiplied by the time index to create an OOD signal.
    """
    time = np.arange(num_points)
    # Generate square wave
    sinusoid = generate_sinusoid(num_points, amplitude, num_periods, noise_std)[1]
    sinusoid[sinusoid > 0] += 1
    square_wave = amplitude * np.sign(sinusoid)

    # Add noise
    noise = np.random.normal(0, noise_std, size=num_points)
    series = (square_wave + noise).astype(np.float32)

    return time.reshape(-1, 1), series

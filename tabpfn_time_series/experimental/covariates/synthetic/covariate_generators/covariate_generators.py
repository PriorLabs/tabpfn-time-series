#!/usr/bin/env python3
"""
Covariate Generators

This module contains various covariate generator classes for creating
different types of synthetic covariates for time series analysis.
"""

import numpy as np
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import CovariateGenerator

# Set up logging
logger = logging.getLogger(__name__)


# Registry of covariate generator instances
ALL_COVARIATE_GENERATORS = {}


def get_covariate_generator(covariate_type: str) -> "CovariateGenerator":
    """Get a covariate generator instance by type name.

    Args:
        covariate_type: Name of the covariate type

    Returns:
        Covariate generator instance

    Raises:
        ValueError: If covariate_type is not recognized
    """
    if covariate_type not in ALL_COVARIATE_GENERATORS:
        available_types = list(ALL_COVARIATE_GENERATORS.keys())
        logger.error(
            f"Unknown covariate type: {covariate_type}. Available types: {available_types}"
        )
        raise ValueError(
            f"Unknown covariate type: {covariate_type}. "
            f"Available types: {available_types}"
        )

    logger.debug(f"Retrieved covariate generator: {covariate_type}")
    return ALL_COVARIATE_GENERATORS[covariate_type]


def generate_covariate(
    covariate_config: dict, n_timesteps: int, seed: int = None
) -> np.ndarray:
    """Generate a covariate from a configuration dictionary.

    Args:
        covariate_config: Dictionary with 'type' and optional 'params' keys
        n_timesteps: Number of timesteps to generate
        seed: Random seed for reproducible generation

    Returns:
        Generated covariate signal as numpy array

    Example:
        config = {
            'type': 'random_walk',
            'params': {'step_size_range': (0.1, 1.0), 'interval_range': (3, 10)}
        }
        covariate = generate_covariate(config, n_timesteps=100, seed=42)
    """
    covariate_type = covariate_config.get("type")
    if not covariate_type:
        raise ValueError("Covariate config must include 'type' field")

    params = covariate_config.get("params", {})
    generator = get_covariate_generator(covariate_type)

    return generator.generate(n_timesteps=n_timesteps, seed=seed, **params)


def get_available_covariate_types():
    """Get list of available covariate types.

    Returns:
        List of available covariate type names
    """
    return list(ALL_COVARIATE_GENERATORS.keys())

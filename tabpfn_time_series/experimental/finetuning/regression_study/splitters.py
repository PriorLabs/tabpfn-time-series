from typing import Tuple
import numpy as np
from functools import partial


def normalize_target_based_on_train(
    y_train: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalizes the target based on the mean and std of the training set.
    """
    y_train_mean = y_train.mean()
    y_train_std = y_train.std()

    print(
        f"Before normalizing: \n"
        f"\ty_train (mean, std) = ({y_train_mean}, {y_train_std}) \n"
        f"\ty_val (mean, std) = ({y_val.mean()}, {y_val.std()})"
    )

    y_train = (y_train - y_train_mean) / y_train_std
    y_val = (y_val - y_train_mean) / y_train_std

    print(
        f"After normalizing: \n"
        f"\ty_train (mean, std) = ({y_train.mean()}, {y_train.std()}) \n"
        f"\ty_val (mean, std) = ({y_val.mean()}, {y_val.std()})"
    )

    return y_train, y_val


def random_chunk_splitter(
    X: np.ndarray,
    y: np.ndarray,
    num_chunks: int = 5,
    chunk_len: int = 10,
    random_state=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits data by masking a number of non-overlapping, randomly placed chunks.

    Args:
        X: Feature array (time indices).
        y: Target array (time series values).
        num_chunks: The number of chunks to mask.
        chunk_len: The length of each individual chunk.
        random_state: Seed for the random number generator.

    Returns:
        A tuple (X_train, X_test, y_train, y_test).
    """
    n_total = len(y)
    mask = np.zeros(n_total, dtype=bool)
    masked_count = 0
    target_masked = num_chunks * chunk_len

    rng = np.random.default_rng(random_state)
    possible_starts = np.arange(n_total - chunk_len + 1)
    rng.shuffle(possible_starts)

    for start in possible_starts:
        if masked_count >= target_masked:
            break
        chunk_indices = np.arange(start, start + chunk_len)
        if not np.any(mask[chunk_indices]):
            mask[chunk_indices] = True
            masked_count += chunk_len

    X_train, y_train = X[~mask], y[~mask]
    X_test, y_test = X[mask], y[mask]

    return X_train, X_test, y_train, y_test


def random_sample_splitter(
    X: np.ndarray,
    y: np.ndarray,
    mask_frac: float = 0.2,
    random_state=None,  # to match sklearn's splitter signature
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    A custom splitter that splits data by masking a random fraction of samples.

    Args:
        X: Feature array (expected to be time indices).
        y: Target array (time series values).
        mask_frac: The fraction of samples to mask for the test set.
        random_state: Seed for the random number generator for reproducibility.

    Returns:
        A tuple (X_train, X_test, y_train, y_test).
    """
    n_total = len(y)
    n_masked = int(n_total * mask_frac)

    # Use a local random state to not affect the global one
    rng = np.random.default_rng(random_state)
    shuffled_indices = rng.permutation(n_total)
    test_indices = shuffled_indices[:n_masked]

    # Create a boolean mask from these indices
    mask = np.zeros(n_total, dtype=bool)
    mask[test_indices] = True

    X_train, y_train = X[~mask], y[~mask]
    X_test, y_test = X[mask], y[mask]

    return X_train, X_test, y_train, y_test


def time_series_splitter(
    X: np.ndarray,
    y: np.ndarray,
    mask_frac: float = 0.2,
    random_state=None,  # to match sklearn's splitter signature
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    A custom splitter that splits data by masking the future (according to chronological order).
    The mask is applied to the last n_masked points.
    """
    n_total = len(y)
    n_masked = int(n_total * mask_frac)
    X_train, y_train = X[:-n_masked], y[:-n_masked]
    X_test, y_test = X[-n_masked:], y[-n_masked:]
    return X_train, X_test, y_train, y_test


def interpolation_splitter(
    X: np.ndarray,
    y: np.ndarray,
    mask_start_frac: float = 0.4,
    mask_len_frac: float = 0.2,
    random_state=None,  # to match sklearn's splitter signature
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    A custom splitter that splits data for an interpolation task.
    It masks a middle segment of the data, using the outer parts for training
    and the middle part for testing.

    Args:
        X: Feature array (expected to be time indices).
        y: Target array (time series values).
        mask_start_frac: The fraction of the series where the mask begins.
        mask_len_frac: The fraction of the series to mask.

    Returns:
        A tuple (X_train, X_test, y_train, y_test).
    """
    n_total = len(y)
    mask_start = int(n_total * mask_start_frac)
    mask_len = int(n_total * mask_len_frac)
    mask_end = mask_start + mask_len

    mask = np.zeros(n_total, dtype=bool)
    mask[mask_start:mask_end] = True

    X_train, y_train = X[~mask], y[~mask]
    X_test, y_test = X[mask], y[mask]

    return X_train, X_test, y_train, y_test


def get_splitter(masking_strategy: str, **kwargs) -> callable:
    """
    Returns a splitter function based on the specified strategy.

    Args:
        masking_strategy: The name of the masking strategy.
        **kwargs: Additional arguments for the splitter function.

    Returns:
        A callable splitter function.
    """
    if masking_strategy == "block":
        return partial(
            interpolation_splitter,
            mask_start_frac=kwargs.get("mask_start_frac", 0.4),
            mask_len_frac=kwargs.get("mask_len_frac", 0.2),
        )
    elif masking_strategy == "random":
        return partial(
            random_sample_splitter,
            mask_frac=kwargs.get("random_mask_frac", 0.2),
            random_state=kwargs.get("random_state"),
        )
    elif masking_strategy == "chunk":
        return partial(
            random_chunk_splitter,
            num_chunks=kwargs.get("num_chunks", 5),
            chunk_len=kwargs.get("chunk_len", 10),
            random_state=kwargs.get("random_state"),
        )
    elif masking_strategy == "future":
        return partial(
            time_series_splitter,
            mask_frac=kwargs.get("future_mask_frac", 0.2),
            random_state=kwargs.get("random_state"),
        )
    else:
        raise ValueError(f"Unknown masking strategy: {masking_strategy}")

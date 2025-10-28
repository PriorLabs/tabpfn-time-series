"""TabPFN-specific parallel worker with retry logic for API calls."""

import contextvars
import backoff
import pandas as pd
from typing import Callable

from tabpfn_time_series.ts_dataframe import TimeSeriesDataFrame
from tabpfn_time_series.worker.parallel_workers.cpu_worker import CPUParallelWorker
from tabpfn_common_utils.telemetry import set_extension, track_model_call


class TabPFNRetryHandler:
    """Handles retry logic for TabPFN client API calls.

    This class manages:
    - Retry attempt counting using context variables (thread-safe)
    - Detection of specific error types (e.g., GCS 429 errors)
    - Giveup logic based on error type and retry count
    """

    # Per-call attempt counter, isolated per thread & task
    _retry_attempts = contextvars.ContextVar("predict_attempts", default=0)

    @classmethod
    def reset_attempts(cls):
        """Reset the attempt counter."""
        cls._retry_attempts.set(0)

    @classmethod
    def increment_attempts(cls):
        """Increment and return the current attempt count."""
        current = cls._retry_attempts.get()
        cls._retry_attempts.set(current + 1)
        return current + 1

    @classmethod
    def is_gcs_429_error(cls, exc: Exception) -> bool:
        """Check if an error is a GCS 429 rate limit error from TabPFN API.

        Args:
            exc: The exception to check

        Returns:
            True if the error is a GCS 429 rate limit error
        """
        markers = (
            "TooManyRequests: 429",
            "rateLimitExceeded",
            "cloud.google.com/storage/docs/gcs429",
        )
        return any(m in str(exc) for m in markers)

    @classmethod
    def should_giveup(cls, exc: Exception) -> bool:
        """Determine whether to give up retrying based on error type and attempt count.

        Args:
            exc: The exception that was raised

        Returns:
            True if retrying should stop, False to continue retrying
        """
        # For GCS 429 errors, keep retrying up to max_tries
        if cls.is_gcs_429_error(exc):
            return False

        # For other errors, stop after first retry (2 attempts total)
        return cls._retry_attempts.get() >= 2


class TabPFNClientCPUParallelWorker(CPUParallelWorker):
    """CPU parallel worker for TabPFN Client with automatic retry logic.

    This worker extends CPUParallelWorker with automatic retry handling
    for TabPFN client API calls, with special treatment for rate limit errors.
    """

    def __init__(
        self,
        inference_routine: Callable,
        num_workers: int = CPUParallelWorker._DEFAULT_NUM_WORKERS,
    ):
        """Initialize TabPFN client worker.

        Args:
            inference_routine: Callable that performs inference on a single time series
            num_workers: Number of parallel workers to use (default: 8)
        """
        super().__init__(inference_routine, num_workers)

    @backoff.on_exception(
        backoff.expo,
        Exception,
        base=1,
        factor=2,
        max_tries=5,
        jitter=backoff.full_jitter,
        giveup=lambda exc: TabPFNRetryHandler.should_giveup(exc),
        on_success=lambda details: TabPFNRetryHandler.reset_attempts(),
    )
    @set_extension("time-series")
    @track_model_call(
        model_method="predict", param_names=["single_train_tsdf", "single_test_tsdf"]
    )
    def _prediction_routine(
        self,
        item_id: str,
        single_train_tsdf: TimeSeriesDataFrame,
        single_test_tsdf: TimeSeriesDataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """Execute prediction with retry logic for API failures.

        Args:
            item_id: Identifier for the time series
            single_train_tsdf: Training data for a single time series
            single_test_tsdf: Test data for a single time series
            **kwargs: Additional arguments passed to inference routine

        Returns:
            DataFrame with predictions for the single time series
        """
        TabPFNRetryHandler.increment_attempts()
        return super()._prediction_routine(
            item_id,
            single_train_tsdf,
            single_test_tsdf,
            **kwargs,
        )

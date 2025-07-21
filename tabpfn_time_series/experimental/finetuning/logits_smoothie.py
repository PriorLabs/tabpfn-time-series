from __future__ import annotations

import torch
from torch import nn


def create_gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    """Creates a 1D Gaussian kernel.

    Args:
        kernel_size: The size of the kernel. Must be an odd number.
        sigma: The standard deviation of the Gaussian distribution.

    Returns:
        A 1D tensor representing the Gaussian kernel.
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd.")
    x = torch.arange(kernel_size) - (kernel_size - 1) / 2.0
    kernel = torch.exp(-(x**2) / (2 * sigma**2))
    return kernel / kernel.sum()


class LogitSmoothieMaker(nn.Conv1d):
    """A convolution layer to smoothen logits using a static Gaussian kernel."""

    def __init__(
        self,
        kernel_size: int,
        sigma: float = 1.0,
    ) -> None:
        """Initializes the LogitSmoother.

        Args:
            kernel_size: The size of the Gaussian kernel. Must be an odd number.
            sigma: The standard deviation of the Gaussian kernel. A good rule of
                   thumb is to have the kernel span about 3 standard deviations
                   on each side, so sigma should be scaled with kernel_size.
                   For a kernel_size of 101, a sigma of ~15 is a good start.
        """
        super().__init__(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding="same",
            bias=False,
            padding_mode="replicate",
        )

        self.weight.requires_grad = False
        kernel = create_gaussian_kernel(kernel_size, sigma)
        self.weight.data = kernel.view(1, 1, kernel_size)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Applies Gaussian smoothing to the input logits.

        Args:
            logits: A tensor of logits, with shape (..., num_bars).

        Returns:
            A tensor of smoothed logits with the same shape as the input.
        """
        original_shape = logits.shape
        num_bars = original_shape[-1]

        # Reshape to (N, 1, L) for Conv1d
        logits_reshaped = logits.reshape(-1, 1, num_bars)

        smoothed_logits = super().forward(logits_reshaped)

        # Reshape back to original shape
        return smoothed_logits.reshape(original_shape)

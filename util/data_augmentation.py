import numpy as np
import torch

# reduplicated from SensorLM data augmentation  
class TimeSeriesAugmentor:
    def __init__(self,
                 jitter_std_range=(0.0, 0.5),
                 scale_range=(1.1, 1.5),
                 p=0.5):
        """
        Args:
          jitter_std_range: (min, max) range for Gaussian noise standard deviation
          scale_range: (min, max) range for multiplicative scaling factor
          p: probability of applying each augmentation independently
        """
        self.jitter_std_range = jitter_std_range
        self.scale_range = scale_range
        self.p = p

    def jitter(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise: mean = 0, std sampled uniformly in jitter_std_range.
        Args:
          x: Tensor of shape [C, L]
        Returns:
          augmented tensor (same shape)
        """
        std = np.random.uniform(self.jitter_std_range[0], self.jitter_std_range[1])
        noise = torch.randn_like(x) * std
        return x + noise

    def scale(self, x: torch.Tensor) -> torch.Tensor:
        """
        Multiply the series by a scalar factor sampled uniformly in scale_range.
        Args:
          x: Tensor of shape [C, L]
        Returns:
          augmented tensor
        """
        factor = np.random.uniform(self.scale_range[0], self.scale_range[1])
        return x * factor

    def time_flip(self, x: torch.Tensor) -> torch.Tensor:
        """
        Flip the time axis: reverse the order of time steps.
        Args:
          x: Tensor of shape [C, L]
        Returns:
          augmented tensor
        """
        return x.flip(dims=[-1])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations independently with probability p each.
        Args:
          x: Tensor of shape [C, L]
        Returns:
          augmented tensor of shape [C, L]
        """
        x = torch.as_tensor(x)
        
        if np.random.rand() < self.p:
            x = self.jitter(x)
        if np.random.rand() < self.p:
            x = self.scale(x)
        if np.random.rand() < self.p:
            x = self.time_flip(x)
        return x
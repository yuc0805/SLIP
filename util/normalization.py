# reference: https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos_bolt.py

import torch
import torch.nn as nn

class InstanceNorm(nn.Module):
    """
    Apply standardization along the last dimension and optionally apply arcsinh after standardization.
    """

    def __init__(self, 
                 eps: float = 1e-5, 
                 use_arcsinh: bool = False) -> None:
        super().__init__()
        self.eps = eps
        self.use_arcsinh = use_arcsinh

    def forward(
        self, 
        x: torch.Tensor, 
        loc_scale: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if loc_scale is None:
            loc = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.0)
            scale = torch.nan_to_num((x - loc).square().nanmean(dim=-1, keepdim=True).sqrt(), nan=1.0)
            scale = torch.where(scale == 0, self.eps, scale)
        else:
            loc, scale = loc_scale

        scaled_x = (x - loc) / scale

        if self.use_arcsinh:
            scaled_x = torch.arcsinh(scaled_x)

        return scaled_x.to(orig_dtype), (loc, scale)

    def inverse(self, x: torch.Tensor, 
                loc_scale: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        loc, scale = loc_scale

        if self.use_arcsinh:
            x = torch.sinh(x)

        x = x * scale + loc

        return x.to(orig_dtype)
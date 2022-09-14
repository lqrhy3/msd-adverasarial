from typing import Dict, Optional, Union

import torch


class CompositeLoss(torch.nn.Module):
    def __init__(self, loss_dict: Dict[torch.nn.Module, float]):
        super().__init__()
        self.loss_dict = CompositeLoss._remove_redundant_losses(loss_dict)

    def forward(
            self,
            input: Dict[str, torch.Tensor],
            target: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        sum_loss = 0.

        for loss_fn, weight in self.loss_dict.items():
            loss_value = loss_fn(input, target)

            sum_loss += weight * torch.mean(loss_value)

        return sum_loss

    @staticmethod
    def _remove_redundant_losses(loss_dict: Dict[torch.nn.Module, float])\
            -> [torch.nn.Module, float]:
        new_loss_dict = {
            loss_fn: weight for loss_fn, weight in loss_dict.items()
            if weight != 0.
        }
        return new_loss_dict

    def to(self, device: Optional[Union[int, torch.device]] = ..., dtype: Optional[Union[torch.dtype, str]] = ...,
           non_blocking: bool = ...):
        super().to(device)
        for loss_fn in self.loss_dict:
            loss_fn.to(device)

        return self

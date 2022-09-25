from typing import Optional
import math

import numpy as np
import torch
from torch import nn


def sigmoid_inv(x):
    if isinstance(x, (int, float)):
        log = math.log
    elif isinstance(x, torch.Tensor):
        log = torch.log
    elif isinstance(x, np.ndarray):
        log = np.log
    else:
        raise ValueError

    return log(x / (1 - x))


class CrossEntropyLossAdv(nn.Module):
    def __init__(self, confidence_thr: Optional[float] = None):
        super(CrossEntropyLossAdv, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.confidence_thr = confidence_thr

    def __call__(self, input: torch.Tensor, target: torch.Tensor):
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        if n_pred_ch == n_target_ch:
            # target is in the one-hot format, convert to BH[WD] format to calculate ce loss
            target = torch.argmax(target, dim=1)
        else:
            target = torch.squeeze(target, dim=1)
        target = target.long()

        loss = self.cross_entropy(input, target)
        if self.confidence_thr is not None:
            logit_thr = sigmoid_inv(self.confidence_thr)
            mask = input < logit_thr
            loss = loss * mask

        loss = loss.mean()
        return loss

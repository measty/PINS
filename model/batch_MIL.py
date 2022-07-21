
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
from typing import Optional
from torch import Tensor
import numpy as np
from cv2 import threshold, THRESH_OTSU

#custom loss, only considers loss of most positive patch in mini-batch. 'minibatch MIL'
class BatchMIL_CE_Loss(_WeightedLoss):

    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean') -> None:
        super(BatchMIL_CE_Loss, self).__init__(weight, size_average, reduce, reduction='none')
        self.ignore_index = ignore_index

    def forward(self, input: Tensor, target: Tensor) -> Tensor:

        losses=F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction='none')

        
        if target[0]==1:
            #batch from positive slide, so most positive example in batch has min loss ->backprop
            #min loss only
            return torch.min(losses)
        else:
            #batch from negative slide, so most positive example in batch has max loss ->backprop
            #max loss value
            return torch.max(losses)


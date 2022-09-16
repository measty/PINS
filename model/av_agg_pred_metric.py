# from unshuffled pred from output of engine and sizes to give
# blocks corr. to slides, aggregate with av and calc metric
from typing import Any, Callable, Tuple
from numpy.core.fromnumeric import _size_dispatcher
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)
from ignite.metrics import EpochMetric
from functools import partial
import torch.nn.functional as F
import warnings
from typing import Callable, List, Sequence, Tuple, Union, cast
import ignite.distributed as idist
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced


def ap_prc_compute_fn(y_pred, y):

    ap = average_precision_score(y.cpu(), y_pred.cpu(), pos_label=1)
    prec, recall, thresholds = precision_recall_curve(
        y.cpu(), y_pred.cpu(), pos_label=1
    )
    return ap  # , prec, recall, thresholds


def auc_roc_compute_fn(y_pred, y):
    auc = roc_auc_score(y.cpu(), y_pred.cpu())
    fpr, tpr, thresholds = roc_curve(y.cpu(), y_pred.cpu(), pos_label=1)
    return auc  # , fpr, tpr, thresholds


def mean_topN(x, N):
    inds = torch.argsort(x, 0, True)
    return torch.mean(x[inds[0:N]])


class av_aggregation_metric(EpochMetric):
    # class to store patch level predictions and aggregate them per slide to get
    # slide level labels when needed, to calculate auc/ap on slide level
    def __init__(
        self,
        sizes,
        agg_type="mean",
        compute_fn: Callable = auc_roc_compute_fn,
        output_transform: Callable = lambda x: x,
    ):
        super(av_aggregation_metric, self).__init__(
            compute_fn, output_transform=output_transform, check_compute_fn=False
        )
        self.sizes = sizes
        self.agg_type = agg_type

    def aggregate_slide(self, y_pred, y):
        csum_sizes = np.cumsum(self.sizes)

        if self.agg_type == "max":
            agg_fn = lambda x: torch.max(x)
        elif self.agg_type == "mean":
            agg_fn = lambda x: torch.mean(x)
        else:
            agg_fn = lambda x: mean_topN(x, self.agg_type)

        y_pred = F.softmax(y_pred, 1)
        # y_pred_out, y_out=torch.zeros([0,2], dtype=y_pred.dtype, device='cpu'), torch.unsqueeze(y[0],0)
        y_pred_out, y_out = (
            torch.zeros([0], dtype=y_pred.dtype, device="cpu"),
            torch.unsqueeze(y[0], 0),
        )
        for i, size in enumerate(self.sizes):
            if i == 0:
                # y_pred_out=torch.cat((y_pred_out,torch.unsqueeze(torch.max(y_pred[0:csum_sizes[i],1]),0)),0)
                y_pred_out = torch.cat(
                    (
                        y_pred_out,
                        torch.unsqueeze(agg_fn(y_pred[0 : csum_sizes[i], 1]), 0),
                    ),
                    0,
                )
            else:
                # y_pred_out=torch.cat((y_pred_out,torch.unsqueeze(torch.max(y_pred[csum_sizes[i-1]:csum_sizes[i],1]),0)),0)
                y_pred_out = torch.cat(
                    (
                        y_pred_out,
                        torch.unsqueeze(
                            agg_fn(y_pred[csum_sizes[i - 1] : csum_sizes[i], 1]), 0
                        ),
                    ),
                    0,
                )
                y_out = torch.cat((y_out, torch.unsqueeze(y[csum_sizes[i] - 1], 0)), 0)

        # y_pred_out=F.softmax(y_pred_out,1)[:,1]
        return y_pred_out, y_out

    def get_data(self, agg=False):
        _prediction_tensor = torch.cat(self._predictions, dim=0)
        _target_tensor = torch.cat(self._targets, dim=0)

        if agg:
            _prediction_tensor, _target_tensor = self.aggregate_slide(
                _prediction_tensor, _target_tensor
            )

        return _prediction_tensor, _target_tensor

    def compute(self) -> float:
        if len(self._predictions) < 1 or len(self._targets) < 1:
            raise NotComputableError(
                "EpochMetric must have at least one example before it can be computed."
            )

        _prediction_tensor = torch.cat(self._predictions, dim=0)
        _target_tensor = torch.cat(self._targets, dim=0)

        ws = idist.get_world_size()

        if ws > 1 and not self._is_reduced:
            # All gather across all processes
            _prediction_tensor = cast(
                torch.Tensor, idist.all_gather(_prediction_tensor)
            )
            _target_tensor = cast(torch.Tensor, idist.all_gather(_target_tensor))
        self._is_reduced = True

        # get slide aggregate predictions
        _prediction_tensor, _target_tensor = self.aggregate_slide(
            _prediction_tensor, _target_tensor
        )

        result = 0.0
        if idist.get_rank() == 0:
            # Run compute_fn on zero rank only
            result = self.compute_fn(_prediction_tensor, _target_tensor)

        if ws > 1:
            # broadcast result to all processes
            result = cast(float, idist.broadcast(result, src=0))

        return result

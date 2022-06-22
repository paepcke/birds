'''
Created on Feb 14, 2021

@author: paepcke

NOT FINISHED!

The following methods are from
https://pytorch-lightning.readthedocs.io/en/stable/_modules/pytorch_lightning/metrics/functional/precision_recall_curve.html
with license from there:

_binary_clf_curve
_precision_recall_curve_update
_precision_recall_curve_compute
precision_recall_curve
'''

from typing import Optional, Sequence, Tuple, List, Union

from logging_service.logging_service import LoggingService
from torch import Tensor
import torch
from torch import t_argsort
from torch import cumsum as t_cumsum
from torch import long as t_long

import torch.nn.functional as F


class MultiClassPRGrapher(object):
    '''
    classdocs
    '''

    def __init__(self, logger=None):
        '''
        Constructor
        '''
        if logger is None:
            self.log = LoggingService()
        else:
            self.log = logger

    def _binary_clf_curve(
            self,
            preds: Tensor,
            target: Tensor,
            sample_weights: Optional[Sequence] = None,
            pos_label: int = 1.,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        adapted from https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/metrics/_ranking.py
        """
        if sample_weights is not None and not isinstance(sample_weights, Tensor):
            sample_weights = torch.tensor(sample_weights, device=preds.device, dtype=torch.float)
    
        # remove class dimension if necessary
        if preds.ndim > target.ndim:
            preds = preds[:, 0]
        desc_score_indices = t_argsort(preds, descending=True)
    
        preds = preds[desc_score_indices]
        target = target[desc_score_indices]
    
        if sample_weights is not None:
            weight = sample_weights[desc_score_indices]
        else:
            weight = 1.
    
        # pred typically has many tied values. Here we extract
        # the indices associated with the distinct values. We also
        # concatenate a value for the end of the curve.
        distinct_value_indices = torch.where(preds[1:] - preds[:-1])[0]
        threshold_idxs = F.pad(distinct_value_indices, (0, 1), value=target.size(0) - 1)
        target = (target == pos_label).to(t_long)
        tps = t_cumsum(target * weight, dim=0)[threshold_idxs]
    
        if sample_weights is not None:
            # express fps as a cumsum to ensure fps is increasing even in
            # the presence of floating point errors
            fps = t_cumsum((1 - target) * weight, dim=0)[threshold_idxs]
        else:
            fps = 1 + threshold_idxs - tps
    
        return fps, tps, preds[threshold_idxs]
    
    
    def _precision_recall_curve_update(
            self,
            preds: Tensor,
            target: Tensor,
            num_classes: Optional[int] = None,
            pos_label: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, int, int]:
        if not (len(preds.shape) == len(target.shape) or len(preds.shape) == len(target.shape) + 1):
            raise ValueError(
                "preds and target must have same number of dimensions, or one additional dimension for preds"
            )
        # single class evaluation
        if len(preds.shape) == len(target.shape):
            if num_classes is not None and num_classes != 1:
                raise ValueError('Preds and target have equal shape, but number of classes is different from 1')
            num_classes = 1
            if pos_label is None:
                self.log.warn('`pos_label` automatically set 1.')
                pos_label = 1
            preds = preds.flatten()
            target = target.flatten()
    
        # multi class evaluation
        if len(preds.shape) == len(target.shape) + 1:
            if pos_label is not None:
                self.log.warn('Argument `pos_label` should be `None` when running'
                               f'multiclass precision recall curve. Got {pos_label}')
            if num_classes != preds.shape[1]:
                raise ValueError(f'Argument `num_classes` was set to {num_classes} in'
                                 f'metric `precision_recall_curve` but detected {preds.shape[1]}'
                                 'number of classes from predictions')
            preds = preds.transpose(0, 1).reshape(num_classes, -1).transpose(0, 1)
            target = target.flatten()
    
        return preds, target, num_classes, pos_label

    def _precision_recall_curve_compute(
            self,
            preds: Tensor,
            target: Tensor,
            num_classes: int,
            pos_label: int,
            sample_weights: Optional[Sequence] = None,
    ) -> Union[Tuple[Tensor, Tensor, Tensor],
               Tuple[List[Tensor], List[Tensor], List[Tensor]]]:
    
        if num_classes == 1:
            fps, tps, thresholds = self._binary_clf_curve(
                preds=preds,
                target=target,
                sample_weights=sample_weights,
                pos_label=pos_label
            )
    
            precision = tps / (tps + fps)
            recall = tps / tps[-1]
    
            # stop when full recall attained
            # and reverse the outputs so recall is decreasing
            last_ind = torch.where(tps == tps[-1])[0][0]
            sl = slice(0, last_ind.item() + 1)
    
            # need to call reversed explicitly, since including that to slice would
            # introduce negative strides that are not yet supported in pytorch
            precision = torch.cat([reversed(precision[sl]),
                                   torch.ones(1, dtype=precision.dtype,
                                              device=precision.device)])
    
            recall = torch.cat([reversed(recall[sl]),
                                torch.zeros(1, dtype=recall.dtype,
                                            device=recall.device)])
    
            thresholds = reversed(thresholds[sl]).clone()
    
            return precision, recall, thresholds
    
        # Recursively call per class
        precision, recall, thresholds = [], [], []
        for c in range(num_classes):
            preds_c = preds[:, c]
            res = self.precision_recall_curve(
                preds=preds_c,
                target=target,
                num_classes=1,
                pos_label=c,
                sample_weights=sample_weights,
            )
            precision.append(res[0])
            recall.append(res[1])
            thresholds.append(res[2])
    
        return precision, recall, thresholds
    
    
    def precision_recall_curve(
            self,
            preds: Tensor,
            target: Tensor,
            num_classes: Optional[int] = None,
            pos_label: Optional[int] = None,
            sample_weights: Optional[Sequence] = None,
    ) -> Union[Tuple[Tensor, Tensor, Tensor],
               Tuple[List[Tensor], List[Tensor], List[Tensor]]]:
        """
        Computes precision-recall pairs for different thresholds.
    
        Args:
            num_classes: integer with number of classes. Not nessesary to provide
                for binary problems.
            pos_label: integer determining the positive class. Default is ``None``
                which for binary problem is translate to 1. For multiclass problems
                this argument should not be set as we iteratively change it in the
                range [0,num_classes-1]
            sample_weight: sample weights for each data point
    
        Returns: 3-element tuple containing
    
            precision:
                tensor where element i is the precision of predictions with
                score >= thresholds[i] and the last element is 1.
                If multiclass, this is a list of such tensors, one for each class.
            recall:
                tensor where element i is the recall of predictions with
                score >= thresholds[i] and the last element is 0.
                If multiclass, this is a list of such tensors, one for each class.
            thresholds:
                Thresholds used for computing precision/recall scores
    
        Example (binary case):
    
            >>> pred = torch.tensor([0, 1, 2, 3])
            >>> target = torch.tensor([0, 1, 1, 0])
            >>> precision, recall, thresholds = precision_recall_curve(pred, target, pos_label=1)
            >>> precision
            tensor([0.6667, 0.5000, 0.0000, 1.0000])
            >>> recall
            tensor([1.0000, 0.5000, 0.0000, 0.0000])
            >>> thresholds
            tensor([1, 2, 3])
    
        Example (multiclass case):
    
            >>> pred = torch.tensor([[0.75, 0.05, 0.05, 0.05, 0.05],
            ...                      [0.05, 0.75, 0.05, 0.05, 0.05],
            ...                      [0.05, 0.05, 0.75, 0.05, 0.05],
            ...                      [0.05, 0.05, 0.05, 0.75, 0.05]])
            >>> target = torch.tensor([0, 1, 3, 2])
            >>> precision, recall, thresholds = precision_recall_curve(pred, target, num_classes=5)
            >>> precision
            [tensor([1., 1.]), tensor([1., 1.]), tensor([0.2500, 0.0000, 1.0000]), tensor([0.2500, 0.0000, 1.0000]), tensor([0., 1.])]
            >>> recall
            [tensor([1., 0.]), tensor([1., 0.]), tensor([1., 0., 0.]), tensor([1., 0., 0.]), tensor([nan, 0.])]
            >>> thresholds
            [tensor([0.7500]), tensor([0.7500]), tensor([0.0500, 0.7500]), tensor([0.0500, 0.7500]), tensor([0.0500])]
    
        """
        preds, target, num_classes, pos_label = self._precision_recall_curve_update(preds, target,
                                                                               num_classes, pos_label)
        return self._precision_recall_curve_compute(preds, target, num_classes, pos_label, sample_weights)


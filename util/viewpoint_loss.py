"""
Multi-class Geometric Viewpoint Aware Loss

A PyTorch implmentation of the geometric-aware softmax view loss as described in
RenderForCNN (link: https://arxiv.org/pdf/1505.05641.pdf)

Caffe implmentation:
https://github.com/charlesq34/caffe-render-for-cnn/blob/view_prediction/
"""

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from IPython import embed

class ViewpointLoss(nn.Module):
    def __init__(self, num_classes = 12, class_period = 360, mean = False, weights = None):
        weights = None
        super(ViewpointLoss, self).__init__()
        self.num_classes = num_classes
        self.class_period = class_period
        self.mean = mean
        # self.weights = np.ones(num_classes) if weights is None else weights

    def forward(self, preds, labels, obj_classes):
        """
        :param preds:   Angle predictions (batch_size, 360 x num_classes)
        :param targets: Angle labels (batch_size, 360 x num_classes)
        :return: Loss. Loss is a variable which may have a backward pass performed.
        Apply softmax over the preds, and then apply geometrics loss
        """
        # Set absolute minimum for numerical stability (assuming float16 - 6x10^-5)
        # preds = F.softmax(preds.float())
        # _min_float  = 1e-6
        # preds       = preds.clamp( min = _min_float)
        preds       = preds.float()
        labels      = labels.float()
        loss        = torch.zeros(1)
        # weights     = torch.from_numpy(self.weights).float()

        if torch.cuda.is_available():
            loss    = loss.cuda()
            # weights = weights.cuda()

        loss    = torch.autograd.Variable(loss)
        # weights = torch.autograd.Variable(weights)
        obj_classes = obj_classes.data.cpu().numpy()


        for inst_id in range(preds.size(0)):
            start_index = obj_classes[inst_id] * self.class_period
            end_index   = start_index + self.class_period
            # if self.mean:
            #     # loss += (labels[inst_id, start_index:end_index] * F.log_softmax(preds[inst_id, start_index:end_index] / preds[inst_id, start_index:end_index].abs().sum())).mean()
            #     # loss += weights[obj_classes[inst_id]] * (labels[inst_id, start_index:end_index] * F.log_softmax(preds[inst_id, start_index:end_index], dim=-1)).mean()
            #     loss -= (labels[inst_id, start_index:end_index] * F.log_softmax(preds[inst_id, start_index:end_index], dim=-1)).mean()
            # else:
            #     # loss += (labels[inst_id, start_index:end_index] * F.log_softmax(preds[inst_id, start_index:end_index] / preds[inst_id, start_index:end_index].abs().sum())).sum()
            #     # loss += weights[obj_classes[inst_id]] * (labels[inst_id, start_index:end_index] * F.log_softmax(preds[inst_id, start_index:end_index], dim=-1)).sum()

            loss -= (labels[inst_id, start_index:end_index] * F.log_softmax(preds[inst_id, start_index:end_index], dim=-1) ).mean()
        # print(loss)


        return loss / preds.size(0)

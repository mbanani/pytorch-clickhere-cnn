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

class ViewpointLoss(nn.Module):
    def __init__(self, num_classes = 12, class_period = 360, mean = False):
        super(ViewpointLoss, self).__init__()
        self.num_classes = num_classes
        self.class_period = class_period
        self.mean = mean

    def forward(self, preds, labels, obj_classes):
        """
        :param preds:   Angle predictions (batch_size, 360 x num_classes)
        :param targets: Angle labels (batch_size, 360 x num_classes)
        :return: Loss. Loss is a variable which may have a backward pass performed.
        Apply softmax over the preds, and then apply geometrics loss
        """
        # Set absolute minimum for numerical stability (assuming float16 - 6x10^-5)
        # preds = F.softmax(preds.float())
        labels = labels.float()
        batch_size = preds.size(0)
        loss     = torch.zeros(1)

        if torch.cuda.is_available():
            loss = loss.cuda()
        loss = torch.autograd.Variable(loss)

        for inst_id in range(batch_size):
            start_index = int(obj_classes[inst_id].data[0]) * self.class_period
            end_index   = start_index + self.class_period
            if self.mean:
                loss -= (labels[inst_id, start_index:end_index] * F.log_softmax(preds[inst_id, start_index:end_index] / preds[inst_id, start_index:end_index].abs().sum())).mean()
            else:
                loss -= (labels[inst_id, start_index:end_index] * F.log_softmax(preds[inst_id, start_index:end_index] / preds[inst_id, start_index:end_index].abs().sum())).sum()
                # loss -= (labels[inst_id, start_index:end_index] * preds[inst_id, start_index:end_index].log()).mean() / preds[inst_id, start_index:end_index].sum()

        # loss = loss / batch_size
        return loss

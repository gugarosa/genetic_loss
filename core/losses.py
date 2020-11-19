import torch
from torch import nn


class PartialLoss(nn.Module):
    def __init__(self):
        super(PartialLoss, self).__init__()

        self.id = torch.round(torch.rand(1))

    def __str__(self):
        if self.id == 0:
            return 'preds'
        elif self.id == 1:
            return 'y'

    def forward(self, preds, y):
        # print(y.requires_grad, preds.requires_grad)
        if self.id == 0:
            # return torch.nn.functional.softmax(y, dim=0).gather(1, preds.unsqueeze(1))
            return nn.LogSoftmax()(preds)
        elif self.id == 1:
            return torch.nn.functional.one_hot(y)

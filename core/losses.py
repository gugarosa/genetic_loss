import torch
from torch import nn


class PartialLoss(nn.Module):
    def __init__(self):
        super(PartialLoss, self).__init__()

        self.id = torch.round(torch.rand(1))

    def __str__(self):
        if self.id == 0:
            return 'y'
        elif self.id == 1:
            return 'y_pred'

    def forward(self, y, y_pred):
        # print(y.requires_grad, y_pred.requires_grad)
        if self.id == 0:
            return torch.nn.functional.softmax(y, dim=0).gather(1, y_pred.unsqueeze(1))
        elif self.id == 1:
            return y_pred

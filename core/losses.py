import torch
from torch import nn


class LabelLoss(nn.Module):
    """
    """

    def __init__(self):
        """Initialization method.

        """

        # Overrides the parent class
        super(LabelLoss, self).__init__()

    def __str__(self):
        """String representation.

        """

        return 'y'

    def forward(self, y):
        """Forward pass.

        Args:
            y (torch.Tensor): True labels.

        """
        
        return y

    
class PredLoss(nn.Module):
    """
    """

    def __init__(self):
        """Initialization method.

        """

        # Overrides the parent class
        super(PredLoss, self).__init__()

    def __str__(self):
        """String representation.

        """

        return 'y_pred'

    def forward(self, y_pred):
        """Forward pass.

        Args:
            y_pred (torch.Tensor): Predictions.

        """
        
        return y_pred
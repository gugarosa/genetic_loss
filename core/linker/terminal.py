import torch
from torch import nn


class Terminal(nn.Module):
    """Terminal wrapper around the Opytimizer's nodes.

    """

    def __init__(self, n_classes=10):
        """Initialization method.

        Args:
            n_classes (int): Number of classes.

        """

        # Overrides the parent class
        super(Terminal, self).__init__()

        # Defines the number of classes for compatibility with one-hot encoding
        self.n_classes = n_classes

        # Defines an identifier for further selection
        # It ranges between 0, 1 and 2
        self.id = torch.randint(0, 3, (1,))

        # Defines a random uniform value between `0` and `1`
        self.value = torch.tensor(torch.rand(1), requires_grad=True)

    def __str__(self):
        """String representation.

        """

        # If it is the first identifier
        if self.id == 0:
            return 'preds'

        # If it is the second identifier
        elif self.id == 1:
            return 'y'

        # If it is the third identifier
        elif self.id == 2:
            return str(round(self.value.item(), 4))

    def forward(self, preds, y):
        """Forward pass.

        Args:
            preds (torch.Tensor): Predictions.
            y (torch.Tensor): True labels.

        Returns:
            The parameter based on the pre-initialized identifier.

        """

        # If it is the first identifier
        if self.id == 0:
            return preds

        # If it is the second identifier
        elif self.id == 1:
            return torch.nn.functional.one_hot(y, num_classes=self.n_classes).float().requires_grad_(True)

        # If it is the third identifier
        elif self.id == 2:
            return self.value

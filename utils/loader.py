import torch
import torchvision as tv

# Constant used to hold a dictionary of possible datasets
DATASETS = {
    'mnist': tv.datasets.MNIST,
    'fmnist': tv.datasets.FashionMNIST,
    'kmnist': tv.datasets.KMNIST
}


def load_dataset(name='mnist', val_split=0.25, seed=0):
    """Loads a dataset.

    Args:
        name (str): Name of dataset to be loaded.
        val_split (float): Percentage of split for the validation set.
        seed (int): Randomness seed.

    Returns:
        Training, validation and testing sets of loaded dataset.

    """

    # Defining the torch seed
    torch.manual_seed(seed)

    # Loads the training data
    train = DATASETS[name](root='./data', train=True, download=True,
                           transform=tv.transforms.ToTensor())

    # Splitting the training data into training/validation
    train, val = torch.utils.data.random_split(train, [int(len(train) * (1 - val_split)),
                                                       int(len(train) * val_split)])

    # Loads the testing data
    test = DATASETS[name](root='./data', train=False, download=True,
                          transform=tv.transforms.ToTensor())

    return train, val, test

import argparse

import torch
from opytimizer.utils.history import History
from torch.utils.data import DataLoader

import utils.loader as l
import utils.object as o
import utils.target as t
import utils.wrapper as w


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(usage='Evaluates an optimized loss.')

    parser.add_argument('dataset', help='Dataset identifier', choices=['mnist', 'fmnist', 'kmnist'])

    parser.add_argument('model', help='Model identifier', choices=['mlp', 'resnet'])

    parser.add_argument('input_file', help='Input history file .pkl identifier', type=str)

    parser.add_argument('-batch_size', help='Batch size', type=int, default=128)

    parser.add_argument('-n_input', help='Number of input units', type=int, default=784)

    parser.add_argument('-n_hidden', help='Number of hidden units', type=int, default=128)

    parser.add_argument('-n_classes', help='Number of classes', type=int, default=10)

    parser.add_argument('-lr', help='Learning rate', type=float, default=0.001)

    parser.add_argument('-epochs', help='Number of training epochs', type=int, default=1)

    parser.add_argument('-device', help='CPU or GPU usage', choices=['cpu', 'cuda'])

    parser.add_argument('-seed', help='Seed identifier', type=int, default=0)

    parser.add_argument('--shuffle', help='Whether data should be shuffled or not', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Common arguments
    dataset = args.dataset
    input_file = args.input_file
    seed = args.seed
    shuffle = args.shuffle

    # Model arguments
    name = args.model
    batch_size = args.batch_size
    n_input = args.n_input
    n_hidden = args.n_hidden
    n_classes = args.n_classes
    epochs = args.epochs
    lr = args.lr
    device = args.device
    
    # Loads the optimization history
    h = History()
    h.load(input_file)

    # Loads the data
    train, _, test = l.load_dataset(name=dataset, seed=seed)

    # Creates the iterators
    train_iterator = DataLoader(train, batch_size=batch_size, shuffle=shuffle)
    test_iterator = DataLoader(test, batch_size=batch_size, shuffle=shuffle)

    # Defining the torch seed
    torch.manual_seed(seed)

    # Gathers the model object
    model_obj = o.get_model(name).obj
    model = model_obj(n_input=n_input, n_hidden=n_hidden, n_classes=n_classes,
                      lr=lr, init_weights=None, device=device)

    # Gathers the loss function
    model.loss = h.best_tree[-1]

    # Fits the model
    model.fit(train_iterator, epochs)

    # Evaluates the model
    _, acc = model.evaluate(test_iterator)

    print(acc)

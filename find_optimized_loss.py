import argparse

import torch
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
    parser = argparse.ArgumentParser(usage='Finds an optimized loss using Genetic Programming.')

    parser.add_argument('dataset', help='Dataset identifier', choices=['mnist', 'fmnist', 'kmnist'])

    parser.add_argument('model', help='Model identifier', choices=['mlp', 'resnet'])

    parser.add_argument('output_file', help='Output history file .pkl identifier', type=str)

    parser.add_argument('-batch_size', help='Batch size', type=int, default=128)

    parser.add_argument('-n_input', help='Number of input units', type=int, default=784)

    parser.add_argument('-n_hidden', help='Number of hidden units', type=int, default=128)

    parser.add_argument('-n_classes', help='Number of classes', type=int, default=10)

    parser.add_argument('-lr', help='Learning rate', type=float, default=0.001)

    parser.add_argument('-epochs', help='Number of training epochs', type=int, default=1)

    parser.add_argument('-n_agents', help='Number of meta-heuristic agents', type=int, default=5)

    parser.add_argument('-n_iter', help='Number of meta-heuristic iterations', type=int, default=10)

    parser.add_argument('-min_depth', help='Minimum depth of trees', type=int, default=1)

    parser.add_argument('-max_depth', help='Maximum depth of trees', type=int, default=5)

    parser.add_argument('-init_loss_prob', help='Probability of initial standard losses', type=float, default=0.0)

    parser.add_argument('-device', help='CPU or GPU usage', choices=['cpu', 'cuda'])

    parser.add_argument('-seed', help='Seed identifier', type=int, default=0)

    parser.add_argument('--shuffle', help='Whether data should be shuffled or not', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Common arguments
    dataset = args.dataset
    output_file = args.output_file
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

    # Optimization arguments
    n_agents = args.n_agents
    n_iterations = args.n_iter
    min_depth = args.min_depth
    max_depth = args.max_depth
    init_loss_prob = args.init_loss_prob

    # Loads the data
    train, val, _ = l.load_dataset(name=dataset, seed=seed)

    # Creates the iterators
    train_iterator = DataLoader(train, batch_size=batch_size, shuffle=shuffle)
    val_iterator = DataLoader(val, batch_size=batch_size, shuffle=shuffle)

    # Defining the torch seed
    torch.manual_seed(seed)

    # Gathers the model object
    model = o.get_model(name).obj

    # Defining the optimization task
    opt_fn = t.validate_losses(train_iterator, val_iterator, model, n_input, n_hidden, n_classes, lr, epochs, device)

    # Running the optimization task
    history = w.run(opt_fn, n_trees=n_agents, n_terminals=3, n_iterations=n_iterations, n_classes=n_classes,
                    min_depth=min_depth, max_depth=max_depth, functions=['MUL', 'LOG_SOFTMAX'],
                    init_loss_prob=init_loss_prob)

    # Saving optimization history
    history.save(output_file)

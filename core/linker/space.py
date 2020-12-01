import copy

import opytimizer.math.random as r
import opytimizer.utils.constants as c
from opytimizer.core.space import Space

from core.linker.node import LossNode
from core.linker.terminal import Terminal

# When using Genetic Programming, each function node needs an unique number of arguments,
# which is defined by this dictionary
N_ARGS_FUNCTION = {
    'SUM': 2,
    'SUB': 2,
    'MUL': 2,
    'DIV': 2,
    'ABS': 1,
    'SQRT': 1,
    'EXP': 1,
    'LOG': 1,
    'COS': 1,
    'SIN': 1,
    'TAN': 1,
    'RELU': 1,
    'SIGMOID': 1,
    'SOFTPLUS': 1,
    'SOFTMAX': 1,
    'TANH': 1,
    'LOG_SIGMOID': 1,
    'LOG_SOFTMAX': 1
}


class LossTreeSpace:
    """LossTreeSpace implements a loss-based version of the tree search space.

    """

    def __init__(self, n_trees=1, n_terminals=1, n_iterations=10, n_classes=10,
                 min_depth=1, max_depth=3, functions=None, init_loss_prob=0.0):
        """Initialization method.

        Args:
            n_trees (int): Number of trees.
            n_terminals (int): Number of terminal nodes.
            n_iterations (int): Number of iterations.
            n_classes (int): Number of classes.
            min_depth (int): Minimum depth of the trees.
            max_depth (int): Maximum depth of the trees.
            functions (list): Functions nodes.
            init_loss_prob (float): Probability of trees instanciated with standard losses.

        """

        # Number of trees
        self.n_trees = n_trees

        # List of fitness
        self.fits = [c.FLOAT_MAX for _ in range(n_trees)]

        # Best fitness value
        self.best_fit = c.FLOAT_MAX

        # Number of terminal nodes
        self.n_terminals = n_terminals

        # Number of iterations
        self.n_iterations = n_iterations

        # Number of classes
        self.n_classes = n_classes

        # Minimum depth of the trees
        self.min_depth = min_depth

        # Maximum depth of the trees
        self.max_depth = max_depth

        # List of functions nodes
        self.functions = functions

        # Probability of trees that should use initial standard losses
        self.init_loss_prob = init_loss_prob

        # Creating the trees
        self._create_trees()

        # Defining flag for later use
        self.built = True

    def _replace_with_standard_loss(self, trees):
        """Replaces a set of trees with standard loss functions.

        Args:
            trees (list): List of trees to be replaced.

        Returns:
            List of trees that were replaced.

        """

        # Creates a set of terminals
        t1 = Terminal(self.n_classes)
        t2 = Terminal(self.n_classes)

        # Replaces their identifiers
        t1.id = 0
        t2.id = 1

        # Creates nodes based on the terminals
        preds = LossNode(str(t1), 'TERMINAL', t1)
        y = LossNode(str(t2), 'TERMINAL', t2)

        # Creates a set of function nodes
        log_softmax = LossNode('LOG_SOFTMAX', 'FUNCTION')
        mul = LossNode('MUL', 'FUNCTION')

        # Creates the Cross Entropy tree
        preds.parent = log_softmax
        log_softmax.left = preds
        log_softmax.parent = mul
        y.parent = mul
        mul.left = log_softmax
        mul.right = y

        # Calculates the number of trees to be replaced
        n_replacement_trees = int(self.init_loss_prob * self.n_trees)

        # Iterates over every replacement tree
        for i in range(n_replacement_trees):
            # Makes a deepcopy over the cross entropy loss function
            trees[i] = copy.deepcopy(mul)

        return trees

    def _create_trees(self):
        """Creates a list of random trees using `GROW` algorithm.

        Args:
            algorithm (str): Algorithm's used to create the initial trees.

        Returns:
            The created trees.

        """

        # Creates a list of random trees
        trees = [self.grow(self.min_depth, self.max_depth)
                      for _ in range(self.n_trees)]

        # Replaces a set of trees with standard loss functions
        self.trees = self._replace_with_standard_loss(trees)

        # Applies the first tree as the best one
        self.best_tree = copy.deepcopy(self.trees[0])

    def grow(self, min_depth=1, max_depth=3):
        """It creates a random tree based on the GROW algorithm.

        References:
            S. Luke. Two Fast Tree-Creation Algorithms for Genetic Programming.
            IEEE Transactions on Evolutionary Computation (2000).

        Args:
            min_depth (int): Minimum depth of the tree.
            max_depth (int): Maximum depth of the tree.

        Returns:
            A random tree based on the GROW algorithm.

        """

        # If minimum depth equals the maximum depth
        if min_depth == max_depth:
            # Creates a terminal-based instance
            terminal = Terminal(n_classes=self.n_classes)

            # Return the terminal node with its id and corresponding loss
            return LossNode(str(terminal), 'TERMINAL', terminal)

        # Generates a node identifier
        node_id = r.generate_integer_random_number(
            0, len(self.functions) + self.n_terminals)

        # If the identifier is a terminal
        if node_id >= len(self.functions):
            # Creates a terminal-based instance
            terminal = Terminal(n_classes=self.n_classes)

            # Return the terminal node with its id and corresponding loss
            return LossNode(str(terminal), 'TERMINAL', terminal)

        # Generates a new function node
        function_node = LossNode(self.functions[node_id], 'FUNCTION')

        # For every possible function argument
        for i in range(N_ARGS_FUNCTION[self.functions[node_id]]):
            # Calls recursively the grow function and creates a temporary node
            node = self.grow(min_depth + 1, max_depth)

            # If it is not the root
            if not i:
                # The left child receives the temporary node
                function_node.left = node

            # If it is the first node
            else:
                # The right child receives the temporary node
                function_node.right = node

                # Flag to identify whether the node is a left child
                node.flag = False

            # The parent of the temporary node is the function node
            node.parent = function_node

        return function_node

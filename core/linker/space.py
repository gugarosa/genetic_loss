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
                 min_depth=1, max_depth=3, functions=None):
        """Initialization method.

        Args:
            n_trees (int): Number of trees.
            n_terminals (int): Number of terminal nodes.
            n_iterations (int): Number of iterations.
            n_classes (int): Number of classes.
            min_depth (int): Minimum depth of the trees.
            max_depth (int): Maximum depth of the trees.
            functions (list): Functions nodes.

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

        # Creating the trees
        self._create_trees()

        # Defining flag for later use
        self.built = True

    def _create_trees(self):
        """Creates a list of random trees using `GROW` algorithm.

        Args:
            algorithm (str): Algorithm's used to create the initial trees.

        Returns:
            The created trees.

        """

        # Creates a list of random trees
        self.trees = [self.grow(self.min_depth, self.max_depth)
                      for _ in range(self.n_trees)]

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

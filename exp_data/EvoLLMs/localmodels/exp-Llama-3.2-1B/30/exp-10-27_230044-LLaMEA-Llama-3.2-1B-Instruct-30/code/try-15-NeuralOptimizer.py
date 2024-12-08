import numpy as np
import random
import math

class NeuralOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.weights = None
        self.bias = None

    def __call__(self, func):
        """
        Optimize the black box function using Neural Optimizer.

        Args:
            func (function): The black box function to optimize.

        Returns:
            float: The optimized value of the function.
        """
        # Initialize weights and bias using a neural network
        self.weights = np.random.rand(self.dim)
        self.bias = np.random.rand(1)
        self.weights = np.vstack((self.weights, [0]))
        self.bias = np.append(self.bias, 0)

        # Define the neural network architecture
        self.nn = {
            'input': self.dim,
            'hidden': self.dim,
            'output': 1
        }

        # Define the optimization function
        def optimize(x):
            # Forward pass
            y = np.dot(x, self.weights) + self.bias
            # Backward pass
            dy = np.dot(self.nn['output'].reshape(-1, 1), (y - func(x)))
            # Update weights and bias
            self.weights -= 0.1 * dy * x
            self.bias -= 0.1 * dy
            return y

        # Run the optimization algorithm
        for _ in range(self.budget):
            # Generate a random input
            x = np.random.rand(self.dim)
            # Optimize the function
            y = optimize(x)
            # Check if the optimization is successful
            if np.allclose(y, func(x)):
                return y
        # If the optimization fails, return None
        return None

    def mutate(self, individual):
        """
        Randomly mutate the individual's weights and bias.

        Args:
            individual (list): The individual to mutate.
        """
        # Select two random indices
        idx1, idx2 = random.sample(range(len(individual)), 2)
        # Swap the weights and bias at the selected indices
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

# Example usage:
def func(x):
    return x**2 + 2*x + 1

optimizer = NeuralOptimizer(budget=100, dim=5)
individual = [1, 2, 3, 4, 5]
optimized_value = optimizer(individual)
print(optimized_value)

# Mutate the individual
mutated_individual = optimizer.mutate(individual)
print(mutated_individual)

# Run the optimization algorithm again
optimized_value = optimizer(mutated_individual)
print(optimized_value)
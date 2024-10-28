# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import numpy as np
import random
import math
import copy

class NeuralOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.weights = None
        self.bias = None
        self.population = []

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
            # Create a copy of the current population
            new_individual = copy.deepcopy(self.population)
            # Optimize the function
            y = optimize(x)
            # Check if the optimization is successful
            if np.allclose(y, func(x)):
                # Refine the strategy by changing the line of the selected solution
                if random.random() < 0.3:
                    new_individual[0] = x
                    new_individual[1] = x
                    new_individual[2] = x
                return y

        # If the optimization fails, return None
        return None

def evaluate_fitness(individual, func, budget):
    """
    Evaluate the fitness of an individual using the given function.

    Args:
        individual (list): The individual to evaluate.
        func (function): The function to evaluate the individual with.
        budget (int): The number of function evaluations allowed.

    Returns:
        float: The fitness of the individual.
    """
    for _ in range(budget):
        # Evaluate the function
        y = func(individual)
        # Check if the evaluation is successful
        if np.allclose(y, func(individual)):
            return y
    # If the evaluation fails, return None
    return None

# Test the algorithm
def test_optimization():
    # Define the function to optimize
    def func(x):
        return x[0]**2 + x[1]**2

    # Define the population and budget
    population = []
    budget = 100

    # Run the optimization algorithm
    optimizer = NeuralOptimizer(budget, 2)
    for _ in range(budget):
        individual = [random.uniform(-5.0, 5.0) for _ in range(2)]
        fitness = evaluate_fitness(individual, func, budget)
        if fitness is not None:
            population.append(individual)
            optimizer.population.append(population)

    # Print the final population
    print("Final population:")
    for individual in optimizer.population:
        print(individual)

test_optimization()
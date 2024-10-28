import numpy as np
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the BlackBoxOptimizer.

        Parameters:
        budget (int): The number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Optimize the black box function using the given budget.

        Parameters:
        func (function): The black box function to optimize.

        Returns:
        float: The optimized value of the function.
        """
        # Initialize the search space with random values
        x = np.random.uniform(-5.0, 5.0, self.dim)
        
        # Perform the given number of function evaluations
        for _ in range(self.budget):
            # Evaluate the function at the current point
            y = func(x)
            
            # Update the search space if the current function value is better
            if y > x[-1]:
                x = stgd(x, func, 0.1, 0.01)  # Refine the search space using stochastic gradient descent
            # Refine the search space using genetic algorithm
            # x = self.fitness(x, func)
        
        # Return the optimized value of the function
        return x[-1]

def fitness(individual, func):
    """
    Evaluate the fitness of an individual.

    Parameters:
    individual (list): The individual to evaluate.
    func (function): The black box function to optimize.

    Returns:
    float: The fitness of the individual.
    """
    return func(individual)

def stgd(x, func, epsilon, learning_rate):
    """
    Iteratively refine the search space using Stochastic Gradient Descent.

    Parameters:
    x (numpy array): The current point in the search space.
    func (function): The black box function to optimize.
    epsilon (float): The step size for the gradient descent update.
    learning_rate (float): The step size for the gradient descent update.

    Returns:
    numpy array: The updated point in the search space.
    """
    y = func(x)
    grad = (y - x[-1]) / epsilon
    x = x[:-1] + np.random.uniform(-epsilon, epsilon, self.dim)
    return x

def func(x):
    return x**2

# Evaluate the fitness of the initial individual
individual = [-2, -1, 0, 1, 2]
optimized_x = BlackBoxOptimizer(1000, 10).func(individual)
print(optimized_x)

# Refine the search space using stochastic gradient descent
optimized_x = stgd(optimized_x, func, 0.1, 0.01)
print(optimized_x)
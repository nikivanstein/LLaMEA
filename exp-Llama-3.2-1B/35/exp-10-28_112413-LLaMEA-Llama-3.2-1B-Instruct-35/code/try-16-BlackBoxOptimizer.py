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
        self.iterations = 0
        self.best_individual = None

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
                x = stgd(x, func, 0.1, 0.01)
        
        # Return the optimized value of the function
        return x[-1]

    def mutate(self, individual):
        """
        Mutate the current individual by changing one gene at a time.

        Parameters:
        individual (numpy array): The current individual.

        Returns:
        numpy array: The mutated individual.
        """
        if self.best_individual is None or self.iterations % 100 == 0:
            # Select a random gene to mutate
            idx = random.randint(0, self.dim - 1)
            
            # Mutate the gene by changing it with a random value from a uniform distribution
            individual[idx] = random.uniform(-5.0, 5.0)
        
        self.iterations += 1
        return individual

# One-line description with the main idea
# "Novel Metaheuristic for Solving Black Box Optimization Problems: Iteratively Refining the Search Space using Stochastic Gradient Descent"

# Code
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

def mutate_func(x):
    return stgd(x, func, 0.1, 0.01)

optimizer = BlackBoxOptimizer(1000, 10)
optimized_x = optimizer(func, 0, 1)
print(optimized_x)

# Refine the search space using mutation
optimized_x = mutate_func(optimized_x)
print(optimized_x)
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

    def __call__(self, func, epsilon=0.1, learning_rate=0.01):
        """
        Optimize the black box function using the given budget.

        Parameters:
        func (function): The black box function to optimize.
        epsilon (float): The step size for the gradient descent update. Defaults to 0.1.
        learning_rate (float): The step size for the gradient descent update. Defaults to 0.01.

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
                # Refine the search space using Stochastic Gradient Descent
                x = stgd(x, func, epsilon, learning_rate)
        
        # Return the optimized value of the function
        return x[-1]

def stgd(x, func, epsilon=0.1, learning_rate=0.01):
    """
    Iteratively refine the search space using Stochastic Gradient Descent.

    Parameters:
    x (numpy array): The current point in the search space.
    func (function): The black box function to optimize.
    epsilon (float): The step size for the gradient descent update. Defaults to 0.1.
    learning_rate (float): The step size for the gradient descent update. Defaults to 0.01.

    Returns:
    numpy array: The updated point in the search space.
    """
    y = func(x)
    grad = (y - x[-1]) / epsilon
    x = x[:-1] + np.random.uniform(-epsilon, epsilon, self.dim)
    return x

# One-line description with the main idea
# "Novel Metaheuristic for Solving Black Box Optimization Problems: Iteratively Refining the Search Space using Stochastic Gradient Descent"

# Code
def func(x):
    return x**2

optimizer = BlackBoxOptimizer(1000, 10)
optimized_x = optimizer(func, epsilon=0.01, learning_rate=0.001)
print(optimized_x)

# Updated solution: Refine the search space using Stochastic Gradient Descent to improve the fitness
def refiner(x, func, epsilon=0.1, learning_rate=0.01):
    """
    Refine the search space using Stochastic Gradient Descent.

    Parameters:
    x (numpy array): The current point in the search space.
    func (function): The black box function to optimize.
    epsilon (float): The step size for the gradient descent update. Defaults to 0.1.
    learning_rate (float): The step size for the gradient descent update. Defaults to 0.01.

    Returns:
    numpy array: The updated point in the search space.
    """
    y = func(x)
    grad = (y - x[-1]) / epsilon
    x = x[:-1] + np.random.uniform(-epsilon, epsilon, self.dim)
    return x

optimizer = BlackBoxOptimizer(1000, 10)
optimized_x = optimizer(func, epsilon=0.01, learning_rate=0.001)
refined_x = refiner(optimized_x, func)
print(refined_x)

# Updated solution: Refine the search space using Stochastic Gradient Descent with a different initial point
def refiner2(x, func, epsilon=0.1, learning_rate=0.01):
    """
    Refine the search space using Stochastic Gradient Descent.

    Parameters:
    x (numpy array): The current point in the search space.
    func (function): The black box function to optimize.
    epsilon (float): The step size for the gradient descent update. Defaults to 0.1.
    learning_rate (float): The step size for the gradient descent update. Defaults to 0.01.

    Returns:
    numpy array: The updated point in the search space.
    """
    y = func(x)
    grad = (y - x[-1]) / epsilon
    x = x[:-1] + np.random.uniform(-epsilon, epsilon, self.dim)
    return x

optimizer = BlackBoxOptimizer(1000, 10)
optimized_x = optimizer(func, epsilon=0.01, learning_rate=0.001)
refined_x = refiner2(optimized_x, func)
print(refined_x)
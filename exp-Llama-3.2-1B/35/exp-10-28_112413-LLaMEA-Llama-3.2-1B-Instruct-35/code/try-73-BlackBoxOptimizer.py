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
        self.best_solution = None
        self.best_score = float('-inf')

    def __call__(self, func, iterations=100):
        """
        Optimize the black box function using the given budget.

        Parameters:
        func (function): The black box function to optimize.
        iterations (int): The number of iterations to perform.

        Returns:
        float: The optimized value of the function.
        """
        # Initialize the search space with random values
        x = np.random.uniform(-5.0, 5.0, self.dim)
        
        # Initialize the best solution and score
        self.best_solution = x
        self.best_score = func(x)

        # Perform the given number of function evaluations
        for _ in range(iterations):
            # Evaluate the function at the current point
            y = func(x)
            
            # Update the search space if the current function value is better
            if y > self.best_score:
                self.best_score = y
                self.best_solution = x
        
        # Return the optimized value of the function
        return self.best_score

    def mutate(self, func, iterations):
        """
        Mutate the current solution using the given number of iterations.

        Parameters:
        func (function): The black box function to optimize.
        iterations (int): The number of iterations to perform.

        Returns:
        float: The mutated value of the function.
        """
        # Initialize the search space with random values
        x = np.random.uniform(-5.0, 5.0, self.dim)
        
        # Perform the given number of function evaluations
        for _ in range(iterations):
            # Evaluate the function at the current point
            y = func(x)
            
            # Update the search space if the current function value is better
            if y > x[-1]:
                x = x[:-1] + np.random.uniform(-0.1, 0.1, self.dim)
        
        # Return the mutated value of the function
        return x

# One-line description with the main idea
# "Novel Metaheuristic for Solving Black Box Optimization Problems: Iteratively Refining the Search Space using Stochastic Gradient Descent with Mutation"

def stgd(x, func, epsilon, learning_rate, mutation_rate, iterations):
    """
    Iteratively refine the search space using Stochastic Gradient Descent with mutation.

    Parameters:
    x (numpy array): The current point in the search space.
    func (function): The black box function to optimize.
    epsilon (float): The step size for the gradient descent update.
    learning_rate (float): The step size for the gradient descent update.
    mutation_rate (float): The probability of mutation.
    iterations (int): The number of iterations to perform.

    Returns:
    numpy array: The updated point in the search space.
    """
    y = func(x)
    grad = (y - x[-1]) / epsilon
    x = x[:-1] + np.random.uniform(-epsilon, epsilon, self.dim)
    if random.random() < mutation_rate:
        x = stgd(x, func, epsilon, learning_rate, mutation_rate, iterations)
    return x

def func(x):
    return x**2

optimizer = BlackBoxOptimizer(1000, 10)
optimized_x = optimizer(func, 100)
print(optimized_x)

optimizer = BlackBoxOptimizer(1000, 10)
optimized_x = optimizer.func(np.random.uniform(-5.0, 5.0, 10))
print(optimized_x)

optimizer = BlackBoxOptimizer(1000, 10)
optimized_x = optimizer.mutate(func, 100)
print(optimized_x)
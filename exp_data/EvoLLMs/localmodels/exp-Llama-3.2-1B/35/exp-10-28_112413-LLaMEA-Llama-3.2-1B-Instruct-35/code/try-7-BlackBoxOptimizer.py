import numpy as np

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
        self.best_individual = None
        self.best_fitness = float('inf')

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
        
        # Evaluate the fitness of the best individual found so far
        if self.best_fitness > y:
            self.best_individual = x
            self.best_fitness = y

        # Return the optimized value of the function
        return y

# One-line description with the main idea
# "Novel Metaheuristic for Solving Black Box Optimization Problems: Iteratively Refining the Search Space using Stochastic Gradient Descent"

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

# Code
def func(x):
    return x**2

optimizer = BlackBoxOptimizer(1000, 10)
optimized_x = optimizer(func, 0, 1)
print(optimized_x)

# Evaluate the fitness of the best individual found so far
if optimizer.best_fitness > func(optimized_x):
    optimizer.best_individual = optimized_x
    optimizer.best_fitness = func(optimized_x)

# Print the best individual found so far
print("Best Individual:", optimizer.best_individual)
print("Best Fitness:", optimizer.best_fitness)
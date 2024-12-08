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

    def __call__(self, func, learning_rate=0.1, epsilon=0.01):
        """
        Optimize the black box function using the given budget.

        Parameters:
        func (function): The black box function to optimize.
        learning_rate (float, optional): The step size for the gradient descent update. Defaults to 0.1.
        epsilon (float, optional): The step size for the gradient descent update. Defaults to 0.01.

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
                new_x = stgd(x, func, epsilon, learning_rate)
                x = new_x

        # Return the optimized value of the function
        return x[-1]

def stgd(x, func, epsilon, learning_rate):
    """
    Refine the search space using Stochastic Gradient Descent.

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

# One-line description with the main idea
# "Novel Metaheuristic for Solving Black Box Optimization Problems: Iteratively Refining the Search Space using Stochastic Gradient Descent with Adaptive Learning Rate"

# Code
optimizer = BlackBoxOptimizer(1000, 10)
optimized_x = optimizer(func, learning_rate=0.1, epsilon=0.01)
print(optimized_x)

# Exception occurred: Traceback (most recent call last):
#  File "/root/LLaMEA/llamea/llamea.py", line 187, in initialize_single
#     new_individual = self.evaluate_fitness(new_individual)
#  File "/root/LLaMEA/mutation_exp.py", line 32, in evaluateBBOB
#     exec(code, globals())
#  File "<string>", line 67, in <module>
#     TypeError: BlackBoxOptimizer.__call__() takes 2 positional arguments but 4 were given
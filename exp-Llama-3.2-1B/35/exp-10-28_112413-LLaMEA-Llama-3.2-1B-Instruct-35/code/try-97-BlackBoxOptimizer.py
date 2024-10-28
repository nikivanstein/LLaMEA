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
                # Refine the search space using Stochastic Gradient Descent
                x = stgd(x, func, 0.1, 0.01)
        
        # Return the optimized value of the function
        return x[-1]

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

optimizer = BlackBoxOptimizer(1000, 10)
optimized_x = optimizer(func, 0, 1)
print(optimized_x)

# Exception occurred: Traceback (most recent call last):
#   File "/root/LLaMEA/llamea/llamea.py", line 187, in initialize_single
#     new_individual = self.evaluate_fitness(new_individual)
#   File "/root/LLaMEA/mutation_exp.py", line 32, in evaluateBBOB
#     exec(code, globals())
#   File "<string>", line 67, in <module>
#     TypeError: BlackBoxOptimizer.__call__() takes 2 positional arguments but 4 were given
#
# Selected solution to update:
# 
# With code:
# ```python
# import numpy as np
# import random
#
# class BlackBoxOptimizer:
#     def __init__(self, budget, dim):
#         """
#         Initialize the BlackBoxOptimizer.

#         Parameters:
#         budget (int): The number of function evaluations allowed.
#         dim (int): The dimensionality of the search space.
#         """
#         self.budget = budget
#         self.dim = dim
#         self.iterations = 0
#
#     def __call__(self, func):
#         """
#         Optimize the black box function using the given budget.

#         Parameters:
#         func (function): The black box function to optimize.

#         Returns:
#         float: The optimized value of the function.
#         """
#         # Initialize the search space with random values
#         x = np.random.uniform(-5.0, 5.0, self.dim)
#         
#         # Perform the given number of function evaluations
#         for _ in range(self.budget):
#             # Evaluate the function at the current point
#             y = func(x)
#             
#             # Update the search space if the current function value is better
#             if y > x[-1]:
#                 # Refine the search space using Stochastic Gradient Descent
#                 x = stgd(x, func, 0.1, 0.01)
#         
#         # Return the optimized value of the function
#         return x[-1]
# 
# # One-line description with the main idea
# # "Novel Metaheuristic for Solving Black Box Optimization Problems: Iterative Refinement of the Search Space using Stochastic Gradient Descent"
# # 
# # Code
# def stgd(x, func, epsilon, learning_rate):
#     """
#     Iteratively refine the search space using Stochastic Gradient Descent.

#     Parameters:
#     x (numpy array): The current point in the search space.
#     func (function): The black box function to optimize.
#     epsilon (float): The step size for the gradient descent update.
#     learning_rate (float): The step size for the gradient descent update.

#     Returns:
#     numpy array: The updated point in the search space.
#     """
#     y = func(x)
#     grad = (y - x[-1]) / epsilon
#     x = x[:-1] + np.random.uniform(-epsilon, epsilon, self.dim)
#     return x
#
# def func(x):
#     return x**2
#
# optimizer = BlackBoxOptimizer(1000, 10)
# optimized_x = optimizer(func, 0, 1)
# print(optimized_x)

# Exception occurred: Traceback (most recent call last):
#   File "/root/LLaMEA/llamea/llamea.py", line 187, in initialize_single
#     new_individual = self.evaluate_fitness(new_individual)
#   File "/root/LLaMEA/mutation_exp.py", line 32, in evaluateBBOB
#     exec(code, globals())
#   File "<string>", line 67, in <module>
#     TypeError: BlackBoxOptimizer.__call__() takes 2 positional arguments but 4 were given
#
# Selected solution to update:
# 
# With code:
# ```python
# import numpy as np
# import random
#
# class BlackBoxOptimizer:
#     def __init__(self, budget, dim):
#         """
#         Initialize the BlackBoxOptimizer.

#         Parameters:
#         budget (int): The number of function evaluations allowed.
#         dim (int): The dimensionality of the search space.
#         """
#         self.budget = budget
#         self.dim = dim
#         self.iterations = 0
#         self.current_individual = None
#
#     def __call__(self, func):
#         """
#         Optimize the black box function using the given budget.

#         Parameters:
#         func (function): The black box function to optimize.

#         Returns:
#         float: The optimized value of the function.
#         """
#         # Initialize the search space with random values
#         self.current_individual = np.random.uniform(-5.0, 5.0, self.dim)
#
#         # Perform the given number of function evaluations
#         for _ in range(self.budget):
#             # Evaluate the function at the current point
#             y = func(self.current_individual)
#
#             # Update the current individual if the current function value is better
#             if y > self.current_individual[-1]:
#                 # Refine the current individual using Stochastic Gradient Descent
#                 self.current_individual = stgd(self.current_individual, func, 0.1, 0.01)
#
#         # Return the optimized value of the function
#         return self.current_individual[-1]
#
# def stgd(x, func, epsilon, learning_rate):
#     """
#     Iteratively refine the search space using Stochastic Gradient Descent.

#     Parameters:
#     x (numpy array): The current point in the search space.
#     func (function): The black box function to optimize.
#     epsilon (float): The step size for the gradient descent update.
#     learning_rate (float): The step size for the gradient descent update.

#     Returns:
#     numpy array: The updated point in the search space.
#     """
#     y = func(x)
#     grad = (y - x[-1]) / epsilon
#     x = x[:-1] + np.random.uniform(-epsilon, epsilon, self.dim)
#     return x
#
# def func(x):
#     return x**2
#
# optimizer = BlackBoxOptimizer(1000, 10)
# optimized_x = optimizer(func, 0, 1)
# print(optimized_x)
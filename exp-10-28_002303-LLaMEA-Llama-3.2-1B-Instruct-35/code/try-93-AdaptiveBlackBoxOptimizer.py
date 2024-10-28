import random
import numpy as np

class AdaptiveBlackBoxOptimizer:
    """
    An optimization algorithm that uses adaptive search strategies to find the optimal solution.
    
    Parameters:
    budget (int): The maximum number of function evaluations allowed.
    dim (int): The dimensionality of the search space.
    learning_rate (float): The initial learning rate for the optimizer.
    """

    def __init__(self, budget, dim, learning_rate):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.learning_rate = learning_rate

    def __call__(self, func):
        """
        Optimize the black box function using the given budget for function evaluations.
        
        Parameters:
        func (function): The black box function to optimize.
        
        Returns:
        tuple: A tuple containing the optimal solution and its cost.
        """
        # Initialize the search space
        lower_bound = -5.0
        upper_bound = 5.0
        
        # Initialize the best solution and its cost
        best_solution = None
        best_cost = float('inf')
        
        # Perform the given number of function evaluations
        for _ in range(self.budget):
            # Initialize the current solution
            new_individual = np.random.uniform(-5.0, 5.0, self.dim)
            
            # Evaluate the function at the current solution
            cost = func(new_individual)
            
            # If the current solution is better than the best solution found so far, update the best solution
            if cost < best_cost:
                best_solution = new_individual
                best_cost = cost
        
        # Return the optimal solution and its cost
        return best_solution, best_cost

# Description: Adaptive Black Box Optimization Algorithm with Adaptive Learning Rate
# Code: 
# ```python
# import numpy as np
# import random
# import time
# 
# def black_box_optimizer(budget, dim, learning_rate):
#     optimizer = AdaptiveBlackBoxOptimizer(budget, dim, learning_rate)
#     func_evals = 0
#     best_solution = None
#     best_cost = float('inf')
# 
#     while True:
#         # Optimize the function using the optimizer
#         solution, cost = optimizer(func)
# 
#         # Increment the number of function evaluations
#         func_evals += 1
# 
#         # If the number of function evaluations exceeds the budget, break the loop
#         if func_evals > budget:
#             break
        
#         # Update the best solution and its cost
#         if cost < best_cost:
            # best_solution = solution
            # best_cost = cost
#     return best_solution, best_cost
# 
# def main():
#     budget = 1000
#     dim = 10
#     learning_rate = 0.1
#     best_solution, best_cost = black_box_optimizer(budget, dim, learning_rate)
#     print("Optimal solution:", best_solution)
#     print("Optimal cost:", best_cost)
# 
# if __name__ == "__main__":
#     main()

# Description: Adaptive Black Box Optimization Algorithm with Adaptive Learning Rate and Exploration
# Code: 
# ```python
# import numpy as np
# import random
# import time
# 
# class AdaptiveBlackBoxOptimizer:
#     def __init__(self, budget, dim, learning_rate):
#         self.budget = budget
#         self.dim = dim
#         self.func_evals = 0
#         self.learning_rate = learning_rate
#         self.explore_rate = 0.1
#         self.explore_threshold = 100
#         self.explore_history = []
# 
#     def __call__(self, func):
#         """
#         Optimize the black box function using the given budget for function evaluations.
#         """
#         # Initialize the search space
#         lower_bound = -5.0
#         upper_bound = 5.0
        
#         # Initialize the best solution and its cost
#         best_solution = None
#         best_cost = float('inf')
        
#         # Perform the given number of function evaluations
#         for _ in range(self.budget):
#             # Initialize the current solution
#             new_individual = np.random.uniform(-5.0, 5.0, self.dim)
            
#             # Evaluate the function at the current solution
#             cost = func(new_individual)
            
#             # If the current solution is better than the best solution found so far, update the best solution
#             if cost < best_cost:
#                 best_solution = new_individual
#                 best_cost = cost
        
#         # Return the optimal solution and its cost
#         return best_solution, best_cost

# Description: Adaptive Black Box Optimization Algorithm with Adaptive Learning Rate and Exploration
# Code: 
# ```python
# import numpy as np
# import random
# import time
# 
# def black_box_optimizer(budget, dim, learning_rate, exploration_rate):
#     optimizer = AdaptiveBlackBoxOptimizer(budget, dim, learning_rate)
#     func_evals = 0
#     best_solution = None
#     best_cost = float('inf')
# 
#     while True:
#         # Optimize the function using the optimizer
#         solution, cost = optimizer(func)
        
#         # Increment the number of function evaluations
#         func_evals += 1
        
#         # If the number of function evaluations exceeds the budget, break the loop
#         if func_evals > budget:
#             break
        
#         # Explore the search space
#         if np.random.rand() < exploration_rate:
            # Introduce some randomness in the exploration
            exploration_cost = random.uniform(0, 1)
            best_solution = optimizer.func_evals * exploration_cost
            best_cost = cost
        # Update the best solution and its cost
        if cost < best_cost:
            best_solution = solution
            best_cost = cost
        
        # Return the optimal solution and its cost
        return best_solution, best_cost

# Description: Adaptive Black Box Optimization Algorithm with Adaptive Learning Rate and Exploration
# Code: 
# ```python
# import numpy as np
# import random
# import time
# 
# def black_box_optimizer(budget, dim, learning_rate, exploration_rate):
#     optimizer = AdaptiveBlackBoxOptimizer(budget, dim, learning_rate)
#     func_evals = 0
#     best_solution = None
#     best_cost = float('inf')
# 
#     while True:
#         # Optimize the function using the optimizer
#         solution, cost = optimizer(func)
        
#         # Increment the number of function evaluations
#         func_evals += 1
        
#         # If the number of function evaluations exceeds the budget, break the loop
#         if func_evals > budget:
#             break
        
#         # Explore the search space
#         if np.random.rand() < exploration_rate:
            # Introduce some randomness in the exploration
            exploration_cost = random.uniform(0, 1)
            best_solution = optimizer.func_evals * exploration_cost
            best_cost = cost
        # Update the best solution and its cost
        if cost < best_cost:
            best_solution = solution
            best_cost = cost
        
        # Return the optimal solution and its cost
        return best_solution, best_cost

# Description: Adaptive Black Box Optimization Algorithm with Adaptive Learning Rate and Exploration
# Code: 
# ```python
# import numpy as np
# import random
# import time
# 
# def black_box_optimizer(budget, dim, learning_rate, exploration_rate):
#     optimizer = AdaptiveBlackBoxOptimizer(budget, dim, learning_rate)
#     func_evals = 0
#     best_solution = None
#     best_cost = float('inf')
# 
#     while True:
#         # Optimize the function using the optimizer
#         solution, cost = optimizer(func)
        
#         # Increment the number of function evaluations
#         func_evals += 1
        
#         # If the number of function evaluations exceeds the budget, break the loop
#         if func_evals > budget:
#             break
        
#         # Explore the search space
#         if np.random.rand() < exploration_rate:
            # Introduce some randomness in the exploration
            exploration_cost = random.uniform(0, 1)
            best_solution = optimizer.func_evals * exploration_cost
            best_cost = cost
        # Update the best solution and its cost
        if cost < best_cost:
            best_solution = solution
            best_cost = cost
        
        # Return the optimal solution and its cost
        return best_solution, best_cost

# Description: Adaptive Black Box Optimization Algorithm with Adaptive Learning Rate and Exploration
# Code: 
# ```python
# import numpy as np
# import random
# import time
# 
# def black_box_optimizer(budget, dim, learning_rate, exploration_rate):
#     optimizer = AdaptiveBlackBoxOptimizer(budget, dim, learning_rate)
#     func_evals = 0
#     best_solution = None
#     best_cost = float('inf')
# 
#     while True:
#         # Optimize the function using the optimizer
#         solution, cost = optimizer(func)
        
#         # Increment the number of function evaluations
#         func_evals += 1
        
#         # If the number of function evaluations exceeds the budget, break the loop
#         if func_evals > budget:
#             break
        
#         # Explore the search space
#         if np.random.rand() < exploration_rate:
            # Introduce some randomness in the exploration
            exploration_cost = random.uniform(0, 1)
            best_solution = optimizer.func_evals * exploration_cost
            best_cost = cost
        # Update the best solution and its cost
        if cost < best_cost:
            best_solution = solution
            best_cost = cost
        
        # Return the optimal solution and its cost
        return best_solution, best_cost

# Description: Adaptive Black Box Optimization Algorithm with Adaptive Learning Rate and Exploration
# Code: 
# ```python
# import numpy as np
# import random
# import time
# 
# def black_box_optimizer(budget, dim, learning_rate, exploration_rate):
#     optimizer = AdaptiveBlackBoxOptimizer(budget, dim, learning_rate)
#     func_evals = 0
#     best_solution = None
#     best_cost = float('inf')
# 
#     while True:
#         # Optimize the function using the optimizer
#         solution, cost = optimizer(func)
        
#         # Increment the number of function evaluations
#         func_evals += 1
        
#         # If the number of function evaluations exceeds the budget, break the loop
#         if func_evals > budget:
#             break
        
#         # Explore the search space
#         if np.random.rand() < exploration_rate:
            # Introduce some randomness in the exploration
            exploration_cost = random.uniform(0, 1)
            best_solution = optimizer.func_evals * exploration_cost
            best_cost = cost
        # Update the best solution and its cost
        if cost < best_cost:
            best_solution = solution
            best_cost = cost
        
        # Return the optimal solution and its cost
        return best_solution, best_cost

# Description: Adaptive Black Box Optimization Algorithm with Adaptive Learning Rate and Exploration
# Code: 
# ```python
# import numpy as np
# import random
# import time
# 
# def black_box_optimizer(budget, dim, learning_rate, exploration_rate):
#     optimizer = AdaptiveBlackBoxOptimizer(budget, dim, learning_rate)
#     func_evals = 0
#     best_solution = None
#     best_cost = float('inf')
# 
#     while True:
#         # Optimize the function using the optimizer
#         solution, cost = optimizer(func)
        
#         # Increment the number of function evaluations
#         func_evals += 1
        
#         # If the number of function evaluations exceeds the budget, break the loop
#         if func_evals > budget:
#             break
        
#         # Explore the search space
#         if np.random.rand() < exploration_rate:
            # Introduce some randomness in the exploration
            exploration_cost = random.uniform(0, 1)
            best_solution = optimizer.func_evals * exploration_cost
            best_cost = cost
        # Update the best solution and its cost
        if cost < best_cost:
            best_solution = solution
            best_cost = cost
        
        # Return the optimal solution and its cost
        return best_solution, best_cost

# Description: Adaptive Black Box Optimization Algorithm with Adaptive Learning Rate and Exploration
# Code: 
# ```python
# import numpy as np
# import random
# import time
# 
# def black_box_optimizer(budget, dim, learning_rate, exploration_rate):
#     optimizer = AdaptiveBlackBoxOptimizer(budget, dim, learning_rate)
#     func_evals = 0
#     best_solution = None
#     best_cost = float('inf')
# 
#     while True:
#         # Optimize the function using the optimizer
#         solution, cost = optimizer(func)
        
#         # Increment the number of function evaluations
#         func_evals += 1
        
#         # If the number of function evaluations exceeds the budget, break the loop
#         if func_evals > budget:
#             break
        
#         # Explore the search space
#         if np.random.rand() < exploration_rate:
            # Introduce some randomness in the exploration
            exploration_cost = random.uniform(0, 1)
            best_solution = optimizer.func_evals * exploration_cost
            best_cost = cost
        # Update the best solution and its cost
        if cost < best_cost:
            best_solution = solution
            best_cost = cost
        
        # Return the optimal solution and its cost
        return best_solution, best_cost

# Description: Adaptive Black Box Optimization Algorithm with Adaptive Learning Rate and Exploration
# Code: 
# ```python
# import numpy as np
# import random
# import time
# 
# def black_box_optimizer(budget, dim, learning_rate, exploration_rate):
#     optimizer = AdaptiveBlackBoxOptimizer(budget, dim, learning_rate)
#     func_evals = 0
#     best_solution = None
#     best_cost = float('inf')
# 
#     while True:
#         # Optimize the function using the optimizer
#         solution, cost = optimizer(func)
        
#         # Increment the number of function evaluations
#         func_evals += 1
        
#         # If the number of function evaluations exceeds the budget, break the loop
#         if func_evals > budget:
#             break
        
#         # Explore the search space
#         if np.random.rand() < exploration_rate:
            # Introduce some randomness in the exploration
            exploration_cost = random.uniform(0, 1)
            best_solution = optimizer.func_evals * exploration_cost
            best_cost = cost
        # Update the best solution and its cost
        if cost < best_cost:
            best_solution = solution
            best_cost = cost
        
        # Return the optimal solution and its cost
        return best_solution, best_cost

# Description: Adaptive Black Box Optimization Algorithm with Adaptive Learning Rate and Exploration
# Code: 
# ```python
# import numpy as np
# import random
# import time
# 
# def black_box_optimizer(budget, dim, learning_rate, exploration_rate):
#     optimizer = AdaptiveBlackBoxOptimizer(budget, dim, learning_rate)
#     func_evals = 0
#     best_solution = None
#     best_cost = float('inf')
# 
#     while True:
#         # Optimize the function using the optimizer
#         solution, cost = optimizer(func)
        
#         # Increment the number of function evaluations
#         func_evals += 1
        
#         # If the number of function evaluations exceeds the budget, break the loop
#         if func_evals > budget:
#             break
        
#         # Explore the search space
#         if np.random.rand() < exploration_rate:
            # Introduce some randomness in the exploration
            exploration_cost = random.uniform(0, 1)
            best_solution = optimizer.func_evals * exploration_cost
            best_cost = cost
        # Update the best solution and its cost
        if cost < best_cost:
            best_solution = solution
            best_cost = cost
        
        # Return the optimal solution and its cost
        return best_solution, best_cost

# Description: Adaptive Black Box Optimization Algorithm with Adaptive Learning Rate and Exploration
# Code: 
# ```python
# import numpy as np
# import random
# import time
# 
# def black_box_optimizer(budget, dim, learning_rate, exploration_rate):
#     optimizer = AdaptiveBlackBoxOptimizer(budget, dim, learning_rate)
#     func_evals = 0
#     best_solution = None
#     best_cost = float('inf')
# 
#     while True:
#         # Optimize the function using the optimizer
#         solution, cost = optimizer(func)
        
#         # Increment the number of function evaluations
#         func_evals += 1
        
#         # If the number of function evaluations exceeds the budget, break the loop
#         if func_evals > budget:
#             break
        
#         # Explore the search space
#         if np.random.rand() < exploration_rate:
            # Introduce some randomness in the exploration
            exploration_cost = random.uniform(0, 1)
            best_solution = optimizer.func_evals * exploration_cost
            best_cost = cost
        # Update the best solution and its cost
        if cost < best_cost:
            best_solution = solution
            best_cost = cost
        
        # Return the optimal solution and its cost
        return best_solution, best_cost
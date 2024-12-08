import numpy as np
import random
from scipy.optimize import differential_evolution

class MGDALR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.explore_rate = 0.1
        self.learning_rate = 0.01
        self.explore_count = 0
        self.max_explore_count = budget

    def __call__(self, func):
        def inner(x):
            return func(x)
        
        # Initialize x to the lower bound
        x = np.array([-5.0] * self.dim)
        
        for _ in range(self.budget):
            # Evaluate the function at the current x
            y = inner(x)
            
            # If we've reached the maximum number of iterations, stop exploring
            if self.explore_count >= self.max_explore_count:
                break
            
            # If we've reached the upper bound, stop exploring
            if x[-1] >= 5.0:
                break
            
            # Learn a new direction using gradient descent
            learning_rate = self.learning_rate * (1 - self.explore_rate / self.max_explore_count)
            dx = -np.dot(x - inner(x), np.gradient(y))
            x += learning_rate * dx
            
            # Update the exploration count
            self.explore_count += 1
            
            # If we've reached the upper bound, stop exploring
            if x[-1] >= 5.0:
                break
        
        return x

class Individual:
    def __init__(self, dim):
        self.dim = dim
        self.x = np.random.uniform(-5.0, 5.0, dim)
    
    def __call__(self):
        return self.x

class MGDALRWithProbability:
    def __init__(self, budget, dim, probability):
        self.budget = budget
        self.dim = dim
        self.probability = probability
        self.explore_rate = 0.1
        self.learning_rate = 0.01
        self.explore_count = 0
        self.max_explore_count = budget

    def __call__(self, func):
        def inner(x):
            return func(x)
        
        # Initialize x to the lower bound
        x = np.array([-5.0] * self.dim)
        
        for _ in range(self.budget):
            # Evaluate the function at the current x
            y = inner(x)
            
            # If we've reached the maximum number of iterations, stop exploring
            if self.explore_count >= self.max_explore_count:
                break
            
            # If we've reached the upper bound, stop exploring
            if x[-1] >= 5.0:
                break
            
            # Learn a new direction using gradient descent with probability
            if random.random() < self.probability:
                learning_rate = self.learning_rate * (1 - self.explore_rate / self.max_explore_count)
                dx = -np.dot(x - inner(x), np.gradient(y))
                x += learning_rate * dx
            
            # Update the exploration count
            self.explore_count += 1
            
            # If we've reached the upper bound, stop exploring
            if x[-1] >= 5.0:
                break
        
        return x

# Description: A novel metaheuristic algorithm for solving black box optimization problems.
# Code: 
# ```python
# import numpy as np
# import random
# import scipy.optimize as optimize
#
# class MGDALR:
#     def __init__(self, budget, dim):
#         self.budget = budget
#         self.dim = dim
#         self.explore_rate = 0.1
#         self.learning_rate = 0.01
#         self.explore_count = 0
#         self.max_explore_count = budget
#
#     def __call__(self, func):
#         def inner(x):
#             return func(x)
#         
#         # Initialize x to the lower bound
#         x = np.array([-5.0] * self.dim)
#         
#         for _ in range(self.budget):
#             # Evaluate the function at the current x
#             y = inner(x)
#             
#             # If we've reached the maximum number of iterations, stop exploring
#             if self.explore_count >= self.max_explore_count:
#                 break
#             
#             # If we've reached the upper bound, stop exploring
#             if x[-1] >= 5.0:
#                 break
#             
#             # Learn a new direction using gradient descent
#             learning_rate = self.learning_rate * (1 - self.explore_rate / self.max_explore_count)
#             dx = -np.dot(x - inner(x), np.gradient(y))
#             x += learning_rate * dx
#
#         return x
#
# 
# class MGDALRWithProbability:
#     def __init__(self, budget, dim, probability):
#         self.budget = budget
#         self.dim = dim
#         self.probability = probability
#         self.explore_rate = 0.1
#         self.learning_rate = 0.01
#         self.explore_count = 0
#         self.max_explore_count = budget
#
#     def __call__(self, func):
#         def inner(x):
#             return func(x)
#         
#         # Initialize x to the lower bound
#         x = np.array([-5.0] * self.dim)
#         
#         for _ in range(self.budget):
#             # Evaluate the function at the current x
#             y = inner(x)
#             
#             # If we've reached the maximum number of iterations, stop exploring
#             if self.explore_count >= self.max_explore_count:
#                 break
#             
#             # If we've reached the upper bound, stop exploring
#             if x[-1] >= 5.0:
#                 break
#             
#             # Learn a new direction using gradient descent with probability
#             if random.random() < self.probability:
#                 learning_rate = self.learning_rate * (1 - self.explore_rate / self.max_explore_count)
#                 dx = -np.dot(x - inner(x), np.gradient(y))
#                 x += learning_rate * dx
#
#         return x
#
# 
# # Description: A novel heuristic algorithm for solving black box optimization problems.
# # Code: 
# # ```python
# # import numpy as np
# # import random
# # import scipy.optimize as optimize
# #
# # class MGDALR:
# #     def __init__(self, budget, dim):
# #         self.budget = budget
# #         self.dim = dim
# #         self.explore_rate = 0.1
# #         self.learning_rate = 0.01
# #         self.explore_count = 0
# #         self.max_explore_count = budget
# #
# #     def __call__(self, func):
# #         def inner(x):
# #             return func(x)
#         
# #         # Initialize x to the lower bound
# #         x = np.array([-5.0] * self.dim)
#         
# #         for _ in range(self.budget):
# #             # Evaluate the function at the current x
# #             y = inner(x)
#             
# #             # If we've reached the maximum number of iterations, stop exploring
# #             if self.explore_count >= self.max_explore_count:
# #                 break
#             
# #             # If we've reached the upper bound, stop exploring
# #             if x[-1] >= 5.0:
# #                 break
#             
# #             # Learn a new direction using gradient descent
# #             learning_rate = self.learning_rate * (1 - self.explore_rate / self.max_explore_count)
# #             dx = -np.dot(x - inner(x), np.gradient(y))
# #             x += learning_rate * dx
#
# #         return x
#
# 
# class MGDALRWithProbability:
#     def __init__(self, budget, dim, probability):
#         self.budget = budget
#         self.dim = dim
#         self.probability = probability
#         self.explore_rate = 0.1
#         self.learning_rate = 0.01
#         self.explore_count = 0
#         self.max_explore_count = budget
#
#     def __call__(self, func):
#         def inner(x):
#             return func(x)
#         
#         # Initialize x to the lower bound
#         x = np.array([-5.0] * self.dim)
#         
#         for _ in range(self.budget):
#             # Evaluate the function at the current x
#             y = inner(x)
#             
#             # If we've reached the maximum number of iterations, stop exploring
#             if self.explore_count >= self.max_explore_count:
#                 break
#             
#             # If we've reached the upper bound, stop exploring
#             if x[-1] >= 5.0:
#                 break
#             
#             # Learn a new direction using gradient descent with probability
#             if random.random() < self.probability:
#                 learning_rate = self.learning_rate * (1 - self.explore_rate / self.max_explore_count)
#                 dx = -np.dot(x - inner(x), np.gradient(y))
#                 x += learning_rate * dx
#
#         return x
#
# # Description: A novel heuristic algorithm for solving black box optimization problems.
# # Code: 
# # ```python
# # import numpy as np
# # import random
# # import scipy.optimize as optimize
# #
# # def gradient(f, x, dx):
# #     return np.gradient(f(x), dx)
#
# # def differential_evolution(f, bounds, initial_guess, max_iter):
# #     result = optimize.differential_evolution(f, bounds, initial_guess, max_iter)
# #     return result.x
#
# # 
# # class MGDALRWithProbability:
# #     def __init__(self, budget, dim, probability):
# #         self.budget = budget
# #         self.dim = dim
# #         self.probability = probability
# #         self.explore_rate = 0.1
# #         self.learning_rate = 0.01
# #         self.explore_count = 0
# #         self.max_explore_count = budget
#
# #     def __call__(self, func):
# #         def inner(x):
# #             return func(x)
#         
# #         # Initialize x to the lower bound
# #         x = np.array([-5.0] * self.dim)
#         
# #         def gradient(f, x, dx):
# #             return np.gradient(f(x), dx)
# 
# #         # Learn a new direction using gradient descent with probability
# #         def differential_evolution(f, bounds, initial_guess, max_iter):
# #             result = optimize.differential_evolution(f, bounds, initial_guess, max_iter)
# #             return result.x
# 
# #         return MGDALR(differential_evolution, self.dim, 0.2)
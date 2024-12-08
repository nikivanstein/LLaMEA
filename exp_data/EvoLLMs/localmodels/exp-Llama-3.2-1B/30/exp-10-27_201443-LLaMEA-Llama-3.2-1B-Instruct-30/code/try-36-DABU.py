import numpy as np
from scipy.optimize import differential_evolution

class DABU:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
        return func_value

    def __str__(self):
        return f"DABU with {self.budget} function evaluations, {self.dim} dimensions"

class DABU2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.fitness_scores = []

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
        self.fitness_scores.append(func_value)

    def get_average_fitness(self):
        return np.mean(self.fitness_scores)

    def get_average_fitness_score(self):
        return np.std(self.fitness_scores)

# Description: A novel metaheuristic algorithm that combines the strengths of DABU and DABU2.
# Code: 
# ```python
# import numpy as np
# import scipy.optimize as optimize

# class DABU2:
#     def __init__(self, budget, dim):
#         self.budget = budget
#         self.dim = dim
#         self.search_space = np.linspace(-5.0, 5.0, dim)
#         self.func_evaluations = 0
#         self.fitness_scores = []

#     def __call__(self, func):
#         while self.func_evaluations < self.budget:
#             func_value = func(self.search_space)
#             if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
#                 break
#             self.func_evaluations += 1
#         self.fitness_scores.append(func_value)

#     def get_average_fitness(self):
#         return np.mean(self.fitness_scores)

#     def get_average_fitness_score(self):
#         return np.std(self.fitness_scores)

#     def __str__(self):
#         return f"DABU2 with {self.budget} function evaluations, {self.dim} dimensions"

# def optimize_function(func, search_space, budget):
#     return DABU2(budget, len(search_space))

# def test_function(x):
#     return np.exp(-x[0]**2 - x[1]**2)

# dabu2 = optimize_function(test_function, np.linspace(-5.0, 5.0, 2), 1000)
# print(dabu2.x)
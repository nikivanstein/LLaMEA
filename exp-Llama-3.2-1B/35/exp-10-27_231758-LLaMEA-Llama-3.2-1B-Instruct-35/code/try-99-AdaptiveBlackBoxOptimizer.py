import numpy as np

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None
        self.learning_rate = 0.01
        self.search_strategy = "uniform"

    def __call__(self, func):
        if self.func_values is None:
            self.func_evals = self.budget
            self.func_values = np.zeros(self.dim)
            for _ in range(self.func_evals):
                func(self.func_values)
        else:
            while self.func_evals > 0:
                if self.search_strategy == "uniform":
                    idx = np.argmin(np.abs(self.func_values))
                    self.func_values[idx] = func(self.func_values[idx])
                elif self.search_strategy == "adaptive":
                    if self.func_evals == 1:
                        self.learning_rate = 0.1
                    else:
                        self.learning_rate *= 0.9
                    idx = np.argmin(np.abs(self.func_values))
                    self.func_values[idx] = func(self.func_values[idx])
                elif self.search_strategy == "exp":
                    idx = np.argmin(np.abs(self.func_values))
                    self.func_values[idx] = func(self.func_values[idx])
                    self.func_evals -= 1
                    if self.func_evals == 0:
                        break
                else:
                    raise ValueError("Invalid search strategy")

# Description: Adaptive Black Box Optimization with Adaptive Learning Rate and Adaptive Search Strategy
# Code: 
# ```python
# ```python
# import numpy as np
# import random

# class AdaptiveBlackBoxOptimizer:
#     def __init__(self, budget, dim):
#         self.budget = budget
#         self.dim = dim
#         self.func_evals = 0
#         self.func_values = None
#         self.learning_rate = 0.01
#         self.search_strategy = "uniform"

#     def __call__(self, func):
#         if self.func_values is None:
#             self.func_evals = self.budget
#             self.func_values = np.zeros(self.dim)
#             for _ in range(self.func_evals):
#                 func(self.func_values)
#         else:
#             while self.func_evals > 0:
#                 if self.search_strategy == "uniform":
#                     idx = np.argmin(np.abs(self.func_values))
#                     self.func_values[idx] = func(self.func_values[idx])
#                 elif self.search_strategy == "adaptive":
#                     if self.func_evals == 1:
#                         self.learning_rate = 0.1
#                     else:
#                         self.learning_rate *= 0.9
#                     idx = np.argmin(np.abs(self.func_values))
#                     self.func_values[idx] = func(self.func_values[idx])
#                 elif self.search_strategy == "exp":
#                     idx = np.argmin(np.abs(self.func_values))
#                     self.func_values[idx] = func(self.func_values[idx])
#                     self.func_evals -= 1
#                     if self.func_evals == 0:
#                         break
#                 else:
#                     raise ValueError("Invalid search strategy")

# class AdaptiveBlackBoxOptimizerWithLearningRateAdaptation:
#     def __init__(self, budget, dim):
#         self.budget = budget
#         self.dim = dim
#         self.func_evals = 0
#         self.func_values = None
#         self.learning_rate = 0.01
#         self.search_strategy = "uniform"
#         self.learning_rate_adaptation = True

#     def __call__(self, func):
#         if self.func_values is None:
#             self.func_evals = self.budget
#             self.func_values = np.zeros(self.dim)
#             for _ in range(self.func_evals):
#                 func(self.func_values)
#         else:
#             while self.func_evals > 0:
#                 if self.search_strategy == "uniform":
#                     idx = np.argmin(np.abs(self.func_values))
#                     self.func_values[idx] = func(self.func_values[idx])
#                 elif self.search_strategy == "adaptive":
#                     if self.func_evals == 1:
#                         self.learning_rate = 0.1
#                     else:
#                         self.learning_rate *= 0.9
#                     idx = np.argmin(np.abs(self.func_values))
#                     self.func_values[idx] = func(self.func_values[idx])
#                 elif self.search_strategy == "exp":
#                     idx = np.argmin(np.abs(self.func_values))
#                     self.func_values[idx] = func(self.func_values[idx])
#                     self.func_evals -= 1
#                     if self.func_evals == 0:
#                         break
#                 else:
#                     raise ValueError("Invalid search strategy")

# class AdaptiveBlackBoxOptimizerWithLearningRateAdaptationAndExp:
#     def __init__(self, budget, dim):
#         self.budget = budget
#         self.dim = dim
#         self.func_evals = 0
#         self.func_values = None
#         self.learning_rate = 0.01
#         self.search_strategy = "uniform"
#         self.learning_rate_adaptation = True
#         self.exp_learning_rate = 0.1
#         self.exp_learning_rate_adaptation = True

#     def __call__(self, func):
#         if self.func_values is None:
#             self.func_evals = self.budget
#             self.func_values = np.zeros(self.dim)
#             for _ in range(self.func_evals):
#                 func(self.func_values)
#         else:
#             while self.func_evals > 0:
#                 if self.search_strategy == "uniform":
#                     idx = np.argmin(np.abs(self.func_values))
#                     self.func_values[idx] = func(self.func_values[idx])
#                 elif self.search_strategy == "adaptive":
#                     if self.func_evals == 1:
#                         self.learning_rate = 0.1
#                     else:
#                         self.learning_rate *= 0.9
#                     idx = np.argmin(np.abs(self.func_values))
#                     self.func_values[idx] = func(self.func_values[idx])
#                 elif self.search_strategy == "exp":
#                     idx = np.argmin(np.abs(self.func_values))
#                     self.func_values[idx] = func(self.func_values[idx])
#                     self.func_evals -= 1
#                     if self.func_evals == 0:
#                         break
#                 elif self.search_strategy == "exp":
#                     if self.exp_learning_rate_adaptation:
#                         self.exp_learning_rate *= 0.9
#                     idx = np.argmin(np.abs(self.func_values))
#                     self.func_values[idx] = func(self.func_values[idx])
#                     self.func_evals -= 1
#                     if self.func_evals == 0:
#                         break
#                 else:
#                     raise ValueError("Invalid search strategy")
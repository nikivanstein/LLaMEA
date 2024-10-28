import numpy as np

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None
        self.sample_size = 0
        self.sample_idx = 0

    def __call__(self, func):
        if self.func_values is None:
            self.func_evals = self.budget
            self.func_values = np.zeros(self.dim)
            for _ in range(self.func_evals):
                func(self.func_values)
        else:
            while self.func_evals > 0:
                idx = np.argmin(np.abs(self.func_values))
                self.func_values[idx] = func(self.func_values[idx])
                self.func_evals -= 1
                if self.func_evals == 0:
                    break

        # Adaptive sampling strategy
        if self.sample_size < self.budget:
            self.sample_size += 1
            self.sample_idx = np.random.randint(0, self.dim)
            while np.abs(self.func_values[self.sample_idx]) > 1e-6:
                self.sample_idx = np.random.randint(0, self.dim)

        return self.func_values

# One-line description: AdaptiveBlackBoxOptimizer uses adaptive sampling to balance exploration and exploitation.
# Code: 
# ```python
# AdaptiveBlackBoxOptimizer: AdaptiveBlackBoxOptimizer
# def __init__(self, budget, dim):
#     self.budget = budget
#     self.dim = dim
#     self.func_evals = 0
#     self.func_values = None
#     self.sample_size = 0
#     self.sample_idx = 0
# 
# def __call__(self, func):
#     if self.func_values is None:
#         self.func_evals = self.budget
#         self.func_values = np.zeros(self.dim)
#         for _ in range(self.func_evals):
#             func(self.func_values)
#     else:
#         while self.func_evals > 0:
#             idx = np.argmin(np.abs(self.func_values))
#             self.func_values[idx] = func(self.func_values[idx])
#             self.func_evals -= 1
#             if self.func_evals == 0:
#                 break
# 
#     # Adaptive sampling strategy
#     if self.sample_size < self.budget:
#         self.sample_size += 1
#         self.sample_idx = np.random.randint(0, self.dim)
#         while np.abs(self.func_values[self.sample_idx]) > 1e-6:
#             self.sample_idx = np.random.randint(0, self.dim)
# 
#     return self.func_values
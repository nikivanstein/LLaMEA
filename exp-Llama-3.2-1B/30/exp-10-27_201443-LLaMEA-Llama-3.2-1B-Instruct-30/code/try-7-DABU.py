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
        return f"DABU: Function evaluations: {self.func_evaluations}/{self.budget}"

class EMA_DABU(DABU):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.ema_coefficient = 0.1

    def __call__(self, func):
        ema_value = self.ema(self.search_space, self.budget, self.func_evaluations)
        func_value = func(self.search_space)
        new_value = ema_value + (1 - self.ema_coefficient) * (func_value - ema_value)
        return new_value

    def ema(self, x, budget, func_evaluations):
        mean = np.mean(x)
        variance = np.var(x)
        return mean + (1 - self.ema_coefficient) * variance / (1 + self.ema_coefficient * variance)

class L1_DABU(DABU):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.l1_coefficient = 0.1

    def __call__(self, func):
        def objective(x):
            return np.sum(np.abs(func(x)))
        return objective

    def __str__(self):
        return f"L1_DABU: Objective function: {self.l1_coefficient * np.sum(np.abs(self.search_space))}"

class L2_DABU(DABU):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.l2_coefficient = 0.1

    def __call__(self, func):
        def objective(x):
            return np.sum((func(x) - self.search_space)**2)
        return objective

    def __str__(self):
        return f"L2_DABU: Objective function: {self.l2_coefficient * np.sum((func(self.search_space) - self.search_space)**2)}"

# Description: Differential Evolution Algorithm with Adaptive EMA
# Code: 
# ```python
# import numpy as np
# import scipy.optimize as optimize
#
# class DAE_DABU(DABU):
#     def __init__(self, budget, dim):
#         super().__init__(budget, dim)
#         self.optimization_function = optimize.differential_evolution
#
#     def __call__(self, func):
#         try:
#             result = self.optimization_function(func, bounds=[(-5, 5), (-5, 5)], x0=[np.random.rand(self.dim)]*self.dim)
#         except Exception as e:
#             print(f"Error: {e}")
#         return result[0]
import numpy as np

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None

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

    def __str__(self):
        return f"AdaptiveBlackBoxOptimizer: Adaptive strategy with Area over the convergence curve (AOCC) score of {self.score:.4f}"

class AdaptiveBlackBoxOptimizerWithRefinement:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None
        self.refinement_threshold = 0.5
        self.refinement_iterations = 0

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

        if self.func_evals == 0:
            return None

        while True:
            idx = np.argmin(np.abs(self.func_values))
            self.func_values[idx] = func(self.func_values[idx])
            self.func_evals -= 1
            if self.func_evals == 0:
                break

            if np.abs(self.func_values[idx]) < self.refinement_threshold:
                self.refinement_iterations += 1
                if self.refinement_iterations >= 10:
                    break

            if self.func_evals == self.budget:
                break

        return self.func_values

# Description: Adaptive strategy with Area over the convergence curve (AOCC) score of 0.03
# Code: 
# ```python
# AdaptiveBlackBoxOptimizerWithRefinement: Adaptive strategy with Area over the convergence curve (AOCC) score of 0.03
# ```python
import numpy as np

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None
        self.alpha = 0.5  # probability of refining the individual lines
        self.beta = 0.3  # probability of switching to a new function

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

    def __refine_line(self, func, line):
        if np.random.rand() < self.alpha:
            idx = np.argmin(np.abs(func(line)))
            self.func_values[idx] = func(self.func_values[idx])
            self.func_evals -= 1
            if self.func_evals == 0:
                break

    def __switch_function(self):
        func_idx = np.argmin(np.abs(self.func_values))
        new_func = np.random.uniform(-5.0, 5.0, self.dim)
        while True:
            if np.random.rand() < self.beta:
                self.func_values[func_idx] = new_func
                self.func_evals -= 1
                if self.func_evals == 0:
                    break
            else:
                self.func_values[func_idx] = self.func_values[idx]
                self.func_evals -= 1
                if self.func_evals == 0:
                    break
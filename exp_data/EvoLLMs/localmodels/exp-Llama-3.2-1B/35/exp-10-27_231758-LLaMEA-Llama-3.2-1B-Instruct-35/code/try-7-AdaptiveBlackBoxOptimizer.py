import numpy as np
import random

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None
        self.search_strategy = "RandomSearch"
        self.refining_strategy = "AdaptiveRefining"
        self.score = 0

    def __call__(self, func):
        if self.func_values is None:
            self.func_evals = self.budget
            self.func_values = np.zeros(self.dim)
            for _ in range(self.func_evals):
                func(self.func_values)
        else:
            while self.func_evals > 0:
                if self.search_strategy == "RandomSearch":
                    idx = random.randint(0, self.dim - 1)
                elif self.search_strategy == "AdaptiveRefining":
                    idx = self.refine_search_idx()
                self.func_values[idx] = func(self.func_values[idx])
                self.func_evals -= 1
                if self.func_evals == 0:
                    break

    def refine_search_idx(self):
        # Refine the search index based on the function values
        # This strategy aims to balance exploration and exploitation
        idx = np.argmin(np.abs(self.func_values))
        # If the search index is close to the minimum, try to move it away
        if np.abs(idx) < 0.5:
            idx -= 1
        return idx

    def adaptive_refining(self):
        # Refine the search strategy based on the function values
        # This strategy aims to balance exploration and exploitation
        # It tries to move the search index away from the minimum and towards the maximum
        idx = np.argmin(np.abs(self.func_values))
        if np.abs(idx) < 0.5:
            idx += 1
        return idx
import numpy as np

class AdaptiveHyperband:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.explore_rate = 0.1
        self.explore_count = 0
        self.best_func = None
        self.best_func_val = None
        self.func_evals = []
        self.budget_count = 0
        self.reduction_factor = 0.3
        self.local_search = True
        self.deterministic = False
        self.explore_strategy = 'uniform'
        self.local_search_strategy = 'random'

    def __call__(self, func, max_evals=1000, max_iter=100):
        for _ in range(max_iter):
            if self.explore_strategy == 'uniform':
                func_evals = [func(np.random.uniform(-5.0, 5.0, self.dim)) for _ in range(max_evals)]
            elif self.explore_strategy == 'adaptive':
                if self.explore_count / self.budget_count < 0.3:
                    func_evals = [func(np.random.uniform(-5.0, 5.0, self.dim)) for _ in range(max_evals)]
                else:
                    func_evals = [func(np.random.uniform(-5.0, 5.0, self.dim)) for _ in range(max_evals)]
            elif self.explore_strategy == 'adaptive_exploitation':
                if self.explore_count / self.budget_count < 0.3:
                    func_evals = [func(np.random.uniform(-5.0, 5.0, self.dim)) for _ in range(max_evals)]
                else:
                    func_evals = [func(np.random.uniform(-5.0, 5.0, self.dim)) for _ in range(max_evals)]
            else:
                raise ValueError("Invalid exploration strategy. Choose from 'uniform', 'adaptive' or 'adaptive_exploitation'.")

            if self.best_func is None or np.abs(func_evals[-1] - self.best_func_val) > np.abs(func_evals[-1] - self.best_func):
                self.best_func = func_evals[-1]
                self.best_func_val = func_evals[-1]
            self.explore_count += 1
            self.budget_count += 1
            if self.budget_count > self.budget / 2:
                break
            if random.random() < self.reduction_factor:
                self.explore_count -= 1
            if random.random() < self.explore_rate:
                self.local_search = not self.local_search
        return self.best_func

# One-line description: Adaptive Hyperband Algorithm with Adaptive Exploration Strategy
# Code: 
import numpy as np

class Hyperband:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.explore_rate = 0.1
        self.explore_count = 0
        self.best_func = None
        self.best_func_val = None
        self.func_evals = []
        self.search_space = np.linspace(-5.0, 5.0, dim)

    def __call__(self, func):
        while self.explore_count < self.budget:
            if self.explore_count > 100:  # limit exploration to prevent infinite loop
                break
            func_eval = func(np.random.choice(self.search_space, size=self.dim))
            self.func_evals.append(func_eval)
            if np.abs(func_eval - self.best_func_val) > np.abs(func_eval - self.best_func):
                self.best_func = func_eval
                self.best_func_val = func_eval
            self.explore_count += 1
        return self.best_func

    def __str__(self):
        return f"Hyperband({self.dim})"

# One-line description: Hyperband is a novel metaheuristic that uses hyperband search to efficiently explore the search space of black box functions.

class HyperbandWithAOCB(Hyperband):
    def __init__(self, budget, dim, aocb_func):
        super().__init__(budget, dim)
        self.aocb_func = aocb_func

    def __call__(self, func):
        while self.explore_count < self.budget:
            if self.explore_count > 100:  # limit exploration to prevent infinite loop
                break
            func_eval = self.aocb_func(func)
            self.func_evals.append(func_eval)
            if np.abs(func_eval - self.best_func_val) > np.abs(func_eval - self.best_func):
                self.best_func = func_eval
                self.best_func_val = func_eval
            self.explore_count += 1
        return self.best_func

# One-line description: HyperbandWithAOCB is a novel metaheuristic that uses hyperband search and area over the convergence curve (AOCC) function to efficiently explore the search space of black box functions.

class AOCB:
    def __init__(self, func, budget, dim):
        self.func = func
        self.budget = budget
        self.dim = dim
        self.explore_rate = 0.1
        self.explore_count = 0
        self.best_func = None
        self.best_func_val = None
        self.func_evals = []
        self.search_space = np.linspace(-5.0, 5.0, dim)

    def __call__(self, func):
        while self.explore_count < self.budget:
            if self.explore_count > 100:  # limit exploration to prevent infinite loop
                break
            func_eval = np.random.uniform(self.search_space)
            self.func_evals.append(func_eval)
            if np.abs(func_eval - self.best_func_val) > np.abs(func_eval - self.best_func):
                self.best_func = func_eval
                self.best_func_val = func_eval
            self.explore_count += 1
        return self.best_func

# One-line description: AOCB is a novel metaheuristic that uses area over the convergence curve (AOCC) function to efficiently explore the search space of black box functions.

# Example usage:
# ```python
# AOCB func = lambda x: np.sin(x)
# HyperbandWithAOCB budget=1000, dim=10
# best_func = HyperbandWithAOCB(budget, dim, AOCB(func, budget, dim))
# print(best_func)
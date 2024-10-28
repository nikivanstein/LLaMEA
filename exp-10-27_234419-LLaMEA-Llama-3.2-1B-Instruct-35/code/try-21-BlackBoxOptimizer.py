import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func = None
        self.search_space = None
        self.sample_size = None
        self.sample_indices = None
        self.local_search = False
        self.budget_history = []
        self.best_individual = None
        self.best_score = -np.inf

    def __call__(self, func):
        if self.func is None:
            self.func = func
            self.search_space = np.random.uniform(-5.0, 5.0, self.dim)
            self.sample_size = 1
            self.sample_indices = None
            self.local_search = False
            self.budget_history = []

        if self.budget <= 0:
            raise ValueError("Budget is less than or equal to zero")

        for _ in range(self.budget):
            if self.sample_indices is None:
                self.sample_indices = np.random.choice(self.search_space, size=self.sample_size, replace=False)
            else:
                self.sample_indices = np.random.choice(self.sample_indices, size=self.sample_size, replace=False)
            self.local_search = False

            if self.local_search:
                best_func = func(self.sample_indices)
                if np.abs(best_func - func(self.sample_indices)) < np.abs(func(self.sample_indices) - func(self.sample_indices)):
                    self.sample_indices = None
                    self.local_search = False
                    self.sample_indices = np.random.choice(self.search_space, size=self.sample_size, replace=False)
                    self.sample_indices = self.sample_indices[:self.sample_size]
                else:
                    self.sample_indices = None
                    self.local_search = False

            if self.sample_indices is None:
                best_func = func(self.sample_indices)
                self.sample_indices = None
                self.local_search = False

            if np.abs(best_func - func(self.sample_indices)) < 1e-6:
                break

            self.budget_history.append(self.budget)
            self.best_individual = func(self.sample_indices)
            self.best_score = np.abs(best_func - func(self.sample_indices))
            self.budget_history.append(self.budget)
            if self.best_score < self.best_score:
                self.best_score = self.best_score
                self.best_individual = func(self.sample_indices)

        return func(self.sample_indices)

class AdaptiveBlackBoxOptimizer(BlackBoxOptimizer):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def __call__(self, func):
        best_individual = self.best_individual
        best_score = self.best_score

        while True:
            new_individual = func(best_individual)
            if np.abs(new_individual - best_individual) < 1e-6:
                break
            self.budget_history.append(self.budget)
            self.best_individual = new_individual
            self.best_score = np.abs(new_individual - best_individual)

        return new_individual
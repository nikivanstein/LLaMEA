import numpy as np

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None
        self.initial_individual = None
        self.initial_population = None
        self.initial_function = None
        self.best_individual = None
        self.best_function = None
        self.best_score = None
        self.best_population = None
        self.best_individual_history = None
        self.best_individual_history_fitness = None
        self.best_individual_history_fitness_history = None

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

        # Refine the strategy
        if self.best_individual is None or self.best_individual_fitness < self.best_score:
            self.best_individual = self.func_values
            self.best_individual_fitness = np.mean(np.abs(self.func_values))
            self.best_score = self.best_individual_fitness
        if self.best_function is None or self.best_individual_fitness < self.best_score:
            self.best_function = func
            self.best_individual_history = self.func_values
            self.best_individual_history_fitness = np.mean(np.abs(self.func_values))
            self.best_individual_history_fitness_history = self.best_individual_fitness

        return self.best_function

# Description: Adaptive Black Box Optimization
# Code: 
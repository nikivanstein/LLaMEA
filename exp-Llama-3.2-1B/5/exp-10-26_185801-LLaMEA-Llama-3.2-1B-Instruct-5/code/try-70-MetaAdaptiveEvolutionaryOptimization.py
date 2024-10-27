import numpy as np

class MetaAdaptiveEvolutionaryOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.best_individual = None
        self.best_fitness = -np.inf

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            func_value = func(self.search_space)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            if func_value < self.best_fitness:
                self.best_individual = self.search_space
                self.best_fitness = func_value
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func_value

    def mutate(self, individual):
        if np.random.rand() < 0.05:
            return np.random.uniform(self.search_space[0], self.search_space[-1]) + self.best_individual
        return individual

    def evaluateBBOB(self, func, num_evaluations):
        for _ in range(num_evaluations):
            func_value = func(self.search_space)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
        return func_value

# Initialize the Meta-Adaptive Evolutionary Optimization algorithm
meta_adaptive_optimization = MetaAdaptiveEvolutionaryOptimization(budget=100, dim=10)

# Example usage:
def func(x):
    return np.sin(x)

# Evaluate the function BBOB
num_evaluations = 100
func_value = meta_adaptive_optimization.evaluateBBOB(func, num_evaluations)
print(f"Function value: {func_value}")
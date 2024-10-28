import numpy as np
from scipy.optimize import minimize

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.population = self.initialize_population()

    def initialize_population(self):
        return [(func, np.random.uniform(-5.0, 5.0), np.random.uniform(-5.0, 5.0)) for func in self.search_space for _ in range(100)]

    def __call__(self, func):
        # Evaluate the function at a random point from the search space
        func_value = func(self.search_space[np.random.randint(0, len(self.search_space))])
        
        # Refine the search space based on the evaluation
        self.search_space = np.linspace(func_value - 1.5, func_value + 1.5, 100)
        
        # Limit the number of evaluations
        evaluations = np.random.uniform(1, self.budget, size=len(self.population))
        self.population = [(func, evaluation) for evaluation in evaluations if evaluation <= self.budget]
        
        # Find the best solution
        best_func, best_func_value, best_idx = self.population[0]
        
        # Refine the search space again
        self.search_space = np.linspace(best_func_value - 1.5, best_func_value + 1.5, 100)
        
        # Limit the number of evaluations again
        evaluations = np.random.uniform(1, self.budget, size=len(self.population))
        self.population = [(func, evaluation) for evaluation in evaluations if evaluation <= self.budget]
        
        # Return the best solution
        return best_func, best_func_value, best_idx

    def run(self):
        for func, evaluation in self.population:
            res = minimize(lambda x: -func(x), x0=np.random.uniform(-5.0, 5.0), args=(func, evaluation))
            if res.fun < 0:
                return func, res.fun, res.x